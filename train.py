from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Value
from peft import (
    AutoPeftModelForSequenceClassification,
    LoraConfig,
    get_peft_model,
)
from simple_parsing import Serializable, field, parse
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import gc

import wandb
from w2s.ds_registry import load_and_process_dataset
from w2s.oraclefilter import rand_filter, oracle_filter
from w2s.knn import gather_hiddens, topofilter
from w2s.loss import log_confidence_loss
from w2s.roc_auc import roc_auc


@dataclass
class TrainConfig(Serializable):
    weak_name: str
    """Name of the weak model to use."""

    dataset: str = field(positional=True)
    """Name of the dataset to use."""

    contamination: float = field(default=0.0)
    """What fraction of data points to remove as outliers."""

    minibatch_size: int = 8
    """Size of the minibatches to use during training."""

    outlier_k: int = field(default=0)
    """Number of neighbors to consider when removing outliers."""

    run_name: str = ""
    """Name of the run."""

    oracle_filter: float = field(default=0.0)
    """What fraction of misclassified data points to remove using the ground truth labels as an oracle. Happens after decontamination."""

    rand_filter: float = field(default=0.0)
    """What fraction of datapoints to randomly filter. Happens before decontamination."""

class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        frac = self.state.global_step / self.state.max_steps
        loss = log_confidence_loss(outputs.logits, labels, frac)

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
    print(f"Best model (auroc {perf:.3f}) saved at: {dest}")


def train(cfg: TrainConfig):
    lora_cfg = LoraConfig(target_modules=LORA_MODULES)

    STRONG_NAME = "meta-llama/Meta-Llama-3-8B"
    strong_tokenizer = AutoTokenizer.from_pretrained(STRONG_NAME)
    weak_tokenizer = AutoTokenizer.from_pretrained(cfg.weak_name)

    # Make sure that the pad token is set
    if strong_tokenizer.pad_token_id is None:
        strong_tokenizer.pad_token = strong_tokenizer.eos_token
    if weak_tokenizer.pad_token_id is None:
        weak_tokenizer.pad_token = weak_tokenizer.eos_token

    splits = load_and_process_dataset(
        cfg.dataset, split_sizes=dict(train=20_000, test=1_000)
    )

    def weak_processor(examples):
        out = weak_tokenizer(examples["txt"], truncation=True)
        out["labels"] = examples["hard_label"]
        return out

    def compute_metrics(eval_pred):
        predictions, labels = map(torch.from_numpy, eval_pred)
        return dict(
            accuracy=predictions.argmax(dim=1).eq(labels).float().mean(),
            auroc=roc_auc(labels, predictions[:, 1]),
        )

    cols = ["hard_label", "txt"]
    test = splits["test"].select_columns(cols)
    train = splits["train"].select_columns(cols)

    weak_test = test.map(weak_processor, batched=True).cast_column(
        "labels", Value("int64")
    )
    weak_train = train.map(weak_processor, batched=True).cast_column(
        "labels", Value("int64")
    )

    root = Path("results") / cfg.dataset
    training_args = TrainingArguments(
        str(root / "floor"),
        adam_beta2=0.95,
        gradient_accumulation_steps=8 // cfg.minibatch_size,
        evaluation_strategy="epoch",
        label_names=["labels"],
        load_best_model_at_end=True,
        logging_steps=50,
        metric_for_best_model="auroc",
        per_device_train_batch_size=cfg.minibatch_size,
        per_device_eval_batch_size=cfg.minibatch_size,
        run_name=cfg.dataset + "/floor" + cfg.run_name,
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
        train_probs = torch.load(label_dir / "train.pt")
        test_probs = torch.load(label_dir / "test.pt")
    else:
        should_train = True
        weak_path = root / "floor/best-ckpt"
        if not weak_path.exists():
            weak_model = AutoModelForSequenceClassification.from_pretrained(
                cfg.weak_name, torch_dtype="auto"
            )

            # HuggingFace init for the head is too large
            weak_model.score.weight.data *= 0.01
            weak_model.config.problem_type = "single_label_classification"
            weak_model = get_peft_model(weak_model, lora_cfg)
        else:
            print("Loading weak model from:", weak_path)
            should_train = False

            weak_model = AutoPeftModelForSequenceClassification.from_pretrained(
                weak_path, torch_dtype="auto"
            )

        weak_model.config.pad_token_id = weak_tokenizer.pad_token_id
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
            print("\n\033[32m===== Training weak model =====\033[0m")
            trainer.train()
            wandb.finish()
            move_best_ckpt(trainer)

        print("Gathering weak labels")
        train_logits = trainer.predict(weak_train).predictions
        test_logits = trainer.predict(weak_test).predictions

        # Convert to probabilities, then keep only the positive probs
        _, train_probs = torch.from_numpy(train_logits).softmax(-1).unbind(-1)
        _, test_probs = torch.from_numpy(test_logits).softmax(-1).unbind(-1)

        label_dir.mkdir(parents=True, exist_ok=True)
        torch.save(train_probs, label_dir / "train.pt")
        torch.save(test_probs, label_dir / "test.pt")

    def strong_processor(examples):
        return strong_tokenizer(examples["txt"], truncation=True)

    strong_train = (
        train.map(strong_processor, batched=True)
        .rename_column("hard_label", "labels")
        .cast_column("labels", Value("int64"))
    )
    ceil_test = (
        test.map(strong_processor, batched=True)
        .rename_column("hard_label", "labels")
        .cast_column("labels", Value("int64"))
    )

    strong_ckpt = root / "ceil" / "best-ckpt"
    if strong_ckpt.exists():
        print(f"Strong ceiling model already exists at {strong_ckpt}")
        strong_trained = AutoModelForSequenceClassification.from_pretrained(
            strong_ckpt, torch_dtype="auto"
        )
        strong_trained.config.pad_token_id = strong_tokenizer.pad_token_id

    else:
        print("\n\033[32m===== Training strong ceiling model =====\033[0m")
        strong_model = AutoModelForSequenceClassification.from_pretrained(
            STRONG_NAME, torch_dtype="auto", device_map={"": "cuda"}
        )
        # HuggingFace init for the head is too large
        strong_model.score.weight.data *= 0.01
        strong_model.config.pad_token_id = strong_tokenizer.pad_token_id

        training_args.output_dir = str(root / "ceil")
        training_args.run_name = cfg.dataset + "/ceil" + cfg.run_name

        trainer = Trainer(
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(strong_tokenizer),
            eval_dataset=ceil_test,
            model=get_peft_model(strong_model, lora_cfg),
            tokenizer=strong_tokenizer,
            train_dataset=strong_train,
        )
        trainer.train()
        wandb.finish()
        move_best_ckpt(trainer)
        strong_trained = strong_model

    strong_trained_trainer = Trainer(
        model=strong_trained,
        compute_metrics=compute_metrics,
        args=training_args,
        data_collator=DataCollatorWithPadding(strong_tokenizer),
        tokenizer=strong_tokenizer,
        train_dataset=strong_train,
        eval_dataset=ceil_test
    )

    strong_trained_logits = strong_trained_trainer.predict(strong_train).predictions
    # Keep both negative and positive class probs
    strong_trained_probs_all = torch.from_numpy(strong_trained_logits).softmax(-1)
    del strong_trained, strong_trained_trainer
    gc.collect()

    print("\n\033[32m===== Training w2s model =====\033[0m")
    strong_model = AutoModelForSequenceClassification.from_pretrained(
        STRONG_NAME, torch_dtype="auto", device_map={"": "cuda"}
    )
    # HuggingFace init for the head is too large
    strong_model.score.weight.data *= 0.01
    strong_model.config.pad_token_id = strong_tokenizer.pad_token_id

    # Weak to strong generalization
    acts_path = root / "ceil/acts.pt"
    if acts_path.exists():
        print(f"Loading strong activations from {acts_path}")
        train_acts = torch.load(acts_path, map_location=strong_model.device)
    else:
        print("Gathering strong activations")
        train_acts = gather_hiddens(strong_model, strong_train)
        torch.save(train_acts, acts_path)

    w2s_train = strong_train.remove_columns("labels")
    w2s_train = w2s_train.add_column("labels", train_probs.numpy())

    if cfg.rand_filter > 0.0:
        retain = rand_filter(train_probs, q=1.0 - cfg.rand_filter)
        w2s_train = w2s_train.select(retain.tolist())
    if cfg.contamination > 0.0:
        y = train_probs.to(train_acts.device)
        indices = topofilter(train_acts, y, cfg.contamination, k=cfg.outlier_k)
        w2s_train = w2s_train.select(indices)
    if cfg.oracle_filter > 0.0:
        y = train_probs.to(train_acts.device)
        # labels = torch.tensor(weak_train['labels']).to(train_acts.device)
        top = oracle_filter(strong_trained_probs_all, y, q=1.0 - cfg.oracle_filter)
        w2s_train = w2s_train.select(top.tolist())

    # Check gt metrics every 100 steps during w2s training.
    # We can overfit to the weak labels before a single epoch.
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = 100
    training_args.save_steps = 100

    training_args.label_names = ["labels"]
    training_args.output_dir = str(root / "w2s") + cfg.run_name
    training_args.run_name = cfg.dataset + "/w2s" + cfg.run_name

    trainer = DistillationTrainer(
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(strong_tokenizer),
        eval_dataset=ceil_test,
        model=get_peft_model(strong_model, lora_cfg),
        tokenizer=strong_tokenizer,
        train_dataset=w2s_train,
    )
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    train(parse(TrainConfig))
