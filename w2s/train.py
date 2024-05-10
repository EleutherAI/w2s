from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from simple_parsing import Serializable, field, parse
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .ds_registry import load_and_process_dataset
from .knn import gather_hiddens, knn_average


@dataclass
class TrainConfig(Serializable):
    weak_name: str
    """Name of the weak model to use."""

    dataset: str = field(positional=True)
    """Name of the dataset to use."""


class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, torch.stack([1.0 - labels, labels], dim=-1)
        )
        return (loss, outputs) if return_outputs else loss


def main():
    cfg = parse(TrainConfig)
    weak_tokenizer = AutoTokenizer.from_pretrained(cfg.weak_name)

    def weak_processor(examples):
        out = weak_tokenizer(examples["txt"], truncation=True)
        out["labels"] = examples["hard_label"]
        return out

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return dict(
            accuracy=np.mean(np.argmax(predictions, axis=1) == labels),
            auroc=roc_auc_score(labels, predictions[:, 1]),
        )

    splits = load_and_process_dataset(
        cfg.dataset, split_sizes=dict(train=20_000, test=1_000)
    )
    weak_test = splits["test"].map(weak_processor, batched=True)
    weak_train = splits["train"].map(weak_processor, batched=True)

    root = Path("results") / cfg.dataset
    training_args = TrainingArguments(
        str(root / "weak"),
        adam_beta2=0.95,
        # bf16=True,  # bf16 mixed precision training
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        weight_decay=0.1,
    )

    # Gather weak labels
    label_dir = root / "weak/preds"
    if label_dir.exists():
        print(f"Loading weak labels from {label_dir}")
        train_probs = np.load(label_dir / "train.npy")
        test_probs = np.load(label_dir / "test.npy")
    else:
        should_train = True
        weak_path = root / "weak/best-ckpt"
        if not weak_path.exists():
            weak_path = cfg.weak_name
        else:
            print("Loading weak model from:", weak_path)
            should_train = False

        weak_model = AutoModelForSequenceClassification.from_pretrained(
            weak_path, torch_dtype="auto"
        )
        # Make sure the pad token is set
        weak_model.config.pad_token_id = (
            weak_tokenizer.pad_token_id
        ) = weak_tokenizer.eos_token_id

        trainer = Trainer(
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(weak_tokenizer),
            eval_dataset=weak_test,
            model=weak_model,
            tokenizer=weak_tokenizer,
            train_dataset=weak_train,
        )
        if should_train:
            print("Training weak model")
            trainer.train()

            path = trainer.state.best_model_checkpoint
            assert path, "No best checkpoint found"

            src = Path(path)
            dest = src.parent / "best-ckpt"
            src.rename(dest)
            print(f"Best model saved at: {dest}")

        print("Gathering weak labels")
        train_logits = trainer.predict(weak_train).predictions
        test_logits = trainer.predict(weak_test).predictions

        # Convert to probabilities, then keep only the positive probs
        _, train_probs = torch.from_numpy(train_logits).softmax(-1).unbind(-1)
        _, test_probs = torch.from_numpy(test_logits).softmax(-1).unbind(-1)
        train_probs, test_probs = train_probs.numpy(), test_probs.numpy()

        label_dir.mkdir(parents=True, exist_ok=True)
        np.save(label_dir / "train.npy", train_probs)
        np.save(label_dir / "test.npy", test_probs)

    # Train the strong model
    STRONG_NAME = "meta-llama/Meta-Llama-3-8B"
    strong_model = AutoModelForSequenceClassification.from_pretrained(
        STRONG_NAME, torch_dtype="auto"
    )
    strong_tokenizer = AutoTokenizer.from_pretrained(STRONG_NAME)
    strong_model.config.pad_token_id = (
        strong_tokenizer.pad_token_id
    ) = strong_tokenizer.eos_token_id

    # Make sure that we use CrossEntropyLoss
    strong_model.config.problem_type = "single_label_classification"

    def strong_processor(examples):
        out = strong_tokenizer(examples["txt"], truncation=True)
        if "weak_label" in examples:
            out["labels"] = examples["weak_label"]

        return out

    strong_train = splits["train"].add_column("weak_label", train_probs)
    strong_test = splits["test"].add_column("weak_label", test_probs)

    strong_train = strong_train.map(strong_processor, batched=True)
    strong_test = strong_test.map(strong_processor, batched=True)

    train_acts = gather_hiddens(strong_model, strong_train)
    labels = knn_average(train_acts, strong_train["labels"], 200)
    top = torch.abs(labels - 0.5).topk(len(labels) // 2).indices
    strong_train = strong_train.select(top.tolist())

    training_args.output_dir = str(root / "strong")

    lora_cfg = LoraConfig(
        target_modules=[
            "gate_proj",
            "down_proj",
            "up_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )
    trainer = DistillationTrainer(
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(strong_tokenizer),
        eval_dataset=strong_test,
        model=get_peft_model(strong_model, lora_cfg),
        tokenizer=strong_tokenizer,
        train_dataset=strong_train,
    )
    trainer.train()


if __name__ == "__main__":
    main()
