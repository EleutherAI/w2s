from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import (
    AutoPeftModelForSequenceClassification,
    LoraConfig,
    get_peft_model,
)
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


# Works for both Llama and Qwen architectures
LORA_MODULES = [
    "gate_proj",
    "down_proj",
    "up_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


def move_best_ckpt(trainer: Trainer):
    path = trainer.state.best_model_checkpoint
    perf = trainer.state.best_metric
    assert path is not None, "No best checkpoint found"
    assert perf is not None, "No best metric"

    src = Path(path)
    dest = src.parent / "best-ckpt"
    src.rename(dest)
    print(f"Best model (loss {perf:.3f}) saved at: {dest}")


def main():
    cfg = parse(TrainConfig)
    lora_cfg = LoraConfig(target_modules=LORA_MODULES)

    STRONG_NAME = "meta-llama/Meta-Llama-3-8B"
    strong_tokenizer = AutoTokenizer.from_pretrained(STRONG_NAME)
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
    test = splits["test"].select_columns(["hard_label", "txt"])
    train = splits["train"].select_columns(["hard_label", "txt"])
    weak_test = test.map(weak_processor, batched=True)
    weak_train = train.map(weak_processor, batched=True)

    root = Path("results") / cfg.dataset
    training_args = TrainingArguments(
        str(root / "floor"),
        adam_beta2=0.95,
        evaluation_strategy="epoch",
        label_names=["labels"],
        load_best_model_at_end=True,
        logging_steps=50,
        metric_for_best_model="auroc",
        save_strategy="epoch",
        save_total_limit=1,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        warmup_steps=100,
        weight_decay=0.01,
    )

    # Gather weak labels
    label_dir = root / "floor/preds"
    if label_dir.exists():
        print(f"Loading weak labels from {label_dir}")
        train_probs = np.load(label_dir / "train.npy")
        test_probs = np.load(label_dir / "test.npy")
    else:
        should_train = True
        weak_path = root / "floor/best-ckpt"
        if not weak_path.exists():
            weak_model = AutoModelForSequenceClassification.from_pretrained(
                cfg.weak_name, torch_dtype="auto"
            )
            # HuggingFace init for the head is too large
            weak_model.score.weight.data *= 0.01

            weak_model = get_peft_model(weak_model, lora_cfg)
        else:
            print("Loading weak model from:", weak_path)
            should_train = False

            weak_model = AutoPeftModelForSequenceClassification.from_pretrained(
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
            print("\033[32m===== Training weak model =====\033[0m")
            trainer.train()
            move_best_ckpt(trainer)

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

    print("\033[32m===== Training strong ceiling model =====\033[0m")
    strong_model = AutoModelForSequenceClassification.from_pretrained(
        STRONG_NAME, torch_dtype="auto", device_map={"": "cuda"}
    )
    # HuggingFace init for the head is too large
    strong_model.score.weight.data *= 0.01

    strong_model.config.pad_token_id = (
        strong_tokenizer.pad_token_id
    ) = strong_tokenizer.eos_token_id

    def strong_processor(examples):
        return strong_tokenizer(examples["txt"], truncation=True)

    strong_train = train.map(strong_processor, batched=True)
    ceil_test = test.map(strong_processor, batched=True).rename_column(
        "hard_label", "labels"
    )

    training_args.output_dir = str(root / "ceil")
    trainer = Trainer(
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(strong_tokenizer),
        eval_dataset=ceil_test,
        model=get_peft_model(strong_model, lora_cfg),
        tokenizer=strong_tokenizer,
        train_dataset=strong_train.rename_column("hard_label", "labels"),
    )
    trainer.train()
    move_best_ckpt(trainer)

    # Init a fresh model for w2s experiment
    strong_model = AutoModelForSequenceClassification.from_pretrained(
        STRONG_NAME, torch_dtype="auto", device_map={"": "cuda"}
    )
    # HuggingFace init for the head is too large
    strong_model.score.weight.data *= 0.01

    # Make sure that we use CrossEntropyLoss
    strong_model.config.problem_type = "single_label_classification"

    # Weak to strong generalization
    acts_path = root / "strong/acts.pt"
    if acts_path.exists():
        print(f"Loading strong activations from {acts_path}")
        train_acts = torch.load(acts_path, map_location=strong_model.device)
    else:
        print("Gathering strong activations")
        train_acts = gather_hiddens(strong_model, strong_train)
        torch.save(train_acts, acts_path)

    y = torch.tensor(strong_train["labels"], device=train_acts.device)
    labels = knn_average(train_acts, y, 200)
    top = torch.abs(labels - 0.5).topk(len(labels) // 2).indices
    strong_train = strong_train.select(top.tolist())

    training_args.label_names = ["labels"]
    training_args.output_dir = str(root / "w2s")

    w2s_train = strong_train.add_column("labels", train_probs)
    trainer = DistillationTrainer(
        args=training_args,
        data_collator=DataCollatorWithPadding(strong_tokenizer),
        eval_dataset=ceil_test,
        model=get_peft_model(strong_model, lora_cfg),
        tokenizer=strong_tokenizer,
        train_dataset=w2s_train,
    )
    trainer.train()
    move_best_ckpt(trainer)


if __name__ == "__main__":
    main()
