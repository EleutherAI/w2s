from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Value, disable_caching
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

import wandb
from w2s.ds_registry import load_and_process_dataset
from w2s.knn import gather_hiddens, topofilter
from w2s.roc_auc import roc_auc

disable_caching()


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

    s2s_iter: int = 0
    """Number of strong-to-strong iterations to perform."""


class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()

        outputs = model(**inputs)

        labels = torch.stack([1.0 - labels, labels], dim=-1)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


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
    # Works for Llama, Mistral, and Qwen architectures
    PEFT_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    peft_cfg = LoraConfig(modules_to_save=["score"], target_modules=PEFT_MODULES)

    # for 2 strong models we do WEAK -> STRONG[0] -> STRONG[1] -> STRONG[0] -> ...
    STRONG_NAMES = ["meta-llama/Meta-Llama-3-8B", "mistralai/Mistral-7B-v0.1"]
    MAIN_STRONG_NAME = STRONG_NAMES[0]
    strong_tokenizers = {k: AutoTokenizer.from_pretrained(k) for k in STRONG_NAMES}
    weak_tokenizer = AutoTokenizer.from_pretrained(cfg.weak_name)

    # Make sure that the pad token is set
    for t in strong_tokenizers.values():
        if t.pad_token_id is None:
            t.pad_token = t.eos_token
    if weak_tokenizer.pad_token_id is None:
        weak_tokenizer.pad_token = weak_tokenizer.eos_token

    splits = load_and_process_dataset(
        cfg.dataset, split_sizes=dict(train=20_000, test=1_000)
    )

    def init_strong_model(name):
        model = AutoModelForSequenceClassification.from_pretrained(
            name, torch_dtype="auto", device_map={"": "cuda"}
        )
        model.config.pad_token_id = strong_tokenizers[name].pad_token_id  # type: ignore
        model.score.weight.data *= 0.01
        return model

    def make_peft(model):
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

        return model

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
    print(f"Train example:\n\n{train[0]['txt']}\n\nLabel: {train[0]['hard_label']}")

    weak_test = test.map(weak_processor, batched=True).cast_column(
        "labels", Value("int64")
    )
    weak_train = train.map(weak_processor, batched=True).cast_column(
        "labels", Value("int64")
    )

    root = Path("results") / cfg.dataset
    train_cfg = dict(
        adam_beta2=0.95,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=8 // cfg.minibatch_size,
        label_names=["labels"],
        learning_rate=5e-4,
        load_best_model_at_end=True,
        logging_steps=50,
        metric_for_best_model="auroc",
        num_train_epochs=3,
        output_dir=str(root / "floor"),
        per_device_eval_batch_size=cfg.minibatch_size,
        per_device_train_batch_size=cfg.minibatch_size,
        run_name=cfg.dataset + "/floor" + cfg.run_name,
        save_strategy="epoch",
        save_total_limit=1,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        warmup_steps=40,
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
        if weak_path.exists():
            weak_model = AutoModelForSequenceClassification.from_pretrained(
                cfg.weak_name, torch_dtype="auto"
            )

            # HuggingFace init for the head is too large
            weak_model.score.weight.data *= 0.01
            weak_model.config.problem_type = "single_label_classification"
            weak_model = make_peft(weak_model)
        else:
            print("Loading weak model from:", weak_path)
            should_train = False

            weak_model = AutoPeftModelForSequenceClassification.from_pretrained(
                weak_path, torch_dtype="auto"
            )

        weak_model.config.pad_token_id = weak_tokenizer.pad_token_id  # type: ignore
        trainer = Trainer(
            args=TrainingArguments(**train_cfg),  # type: ignore
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
        train_probs = torch.from_numpy(train_logits).softmax(-1)[:, 1]
        test_probs = torch.from_numpy(test_logits).softmax(-1)[:, 1]

        label_dir.mkdir(parents=True, exist_ok=True)
        torch.save(train_probs, label_dir / "train.pt")
        torch.save(test_probs, label_dir / "test.pt")

    def strong_processor(examples, tokenizer):
        return tokenizer(examples["txt"], truncation=True)

    strong_trains = {
        name: train.map(
            strong_processor,
            batched=True,
            fn_kwargs={"tokenizer": strong_tokenizers[name]},
        )
        .rename_column("hard_label", "labels")
        .cast_column("labels", Value("int64"))
        for name in STRONG_NAMES
    }
    ceil_tests = {
        name: test.map(
            strong_processor,
            batched=True,
            fn_kwargs={"tokenizer": strong_tokenizers[name]},
        )
        .rename_column("hard_label", "labels")
        .cast_column("labels", Value("int64"))
        for name in STRONG_NAMES
    }

    # Lower learning rate for the larger model
    train_cfg["learning_rate"] = 2e-4

    strong_ckpt = root / "ceil" / "best-ckpt"
    if strong_ckpt.exists():
        print(f"Strong ceiling model already exists at {strong_ckpt}")
    else:
        print("\n\033[32m===== Training strong ceiling model =====\033[0m")
        train_cfg["output_dir"] = str(root / "ceil")
        train_cfg["run_name"] = cfg.dataset + "/ceil" + cfg.run_name

        trainer = Trainer(
            args=TrainingArguments(**train_cfg),  # type: ignore
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(strong_tokenizers[MAIN_STRONG_NAME]),
            eval_dataset=ceil_tests[MAIN_STRONG_NAME],
            model=make_peft(init_strong_model(MAIN_STRONG_NAME)),
            tokenizer=strong_tokenizers[MAIN_STRONG_NAME],
            train_dataset=strong_trains[MAIN_STRONG_NAME],
        )
        trainer.train()
        wandb.finish()
        move_best_ckpt(trainer)

    # Weak to strong generalization
    w2s_train = strong_trains[MAIN_STRONG_NAME].remove_columns("labels")
    w2s_train = w2s_train.add_column("labels", train_probs.numpy())

    train_cfg.update(
        # Check gt metrics every 100 steps during w2s training.
        # We can overfit to the weak labels before a single epoch.
        eval_steps=250,
        evaluation_strategy="steps",
        # Use a 4x larger batch size, performing fewer training steps
        # given the same number of examples to reduce overfitting
        # to the weak model's errors.
        gradient_accumulation_steps=32 // cfg.minibatch_size,
        label_names=["labels"],
        num_train_epochs=3,
        output_dir=str(root / ("w2s" + cfg.run_name)),
        run_name=cfg.dataset + "/w2s" + cfg.run_name,
        save_steps=250,
        save_strategy="steps",
    )

    should_train = True
    w2s_ckpt = root / ("w2s" + cfg.run_name) / "best-ckpt"
    if w2s_ckpt.exists():
        print(f"W2S model already exists at {w2s_ckpt}")

        w2s_model = AutoPeftModelForSequenceClassification.from_pretrained(
            w2s_ckpt, torch_dtype="auto", device_map={"": "cuda"}
        )
        should_train = False
    else:
        print("\n\033[32m===== Training w2s model =====\033[0m")

        strong_model = init_strong_model(MAIN_STRONG_NAME)
        if cfg.contamination > 0.0:
            acts_path = root / "ceil/acts.pt"
            if acts_path.exists():
                print(f"Loading strong activations from {acts_path}")
                train_acts = torch.load(acts_path, map_location=strong_model.device)
            else:
                print("Gathering strong activations")
                train_acts = gather_hiddens(strong_model, w2s_train)
                torch.save(train_acts, acts_path)

            y = train_probs.to(train_acts.device)
            indices = topofilter(train_acts, y, cfg.contamination, k=cfg.outlier_k)
            w2s_train = w2s_train.select(indices)

        w2s_model = make_peft(strong_model)

    w2s_model.config.pad_token_id = strong_tokenizers[MAIN_STRONG_NAME].pad_token_id  # type: ignore # noqa
    trainer = DistillationTrainer(
        args=TrainingArguments(**train_cfg),  # type: ignore
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(strong_tokenizers[MAIN_STRONG_NAME]),
        eval_dataset=ceil_tests[MAIN_STRONG_NAME],
        model=w2s_model,
        tokenizer=strong_tokenizers[MAIN_STRONG_NAME],
        train_dataset=w2s_train,
    )
    if should_train:
        trainer.train()
        wandb.finish()
        move_best_ckpt(trainer)

    # Save memory
    del w2s_model

    preds_path = root / ("w2s" + cfg.run_name) / "preds.pt"

    # Strong to strong generalization
    for i in range(cfg.s2s_iter):
        strong_name = STRONG_NAMES[(i + 1) % len(STRONG_NAMES)]
        print(f"\n\033[32m===== S2S-distillation {i + 1} ({strong_name}) =====\033[0m")

        # Gather strong labels
        if preds_path.exists():
            print(f"Loading strong preds from {preds_path}")
            train_probs = torch.load(preds_path)
        else:
            train_logits = trainer.predict(w2s_train).predictions
            train_probs = torch.from_numpy(train_logits).softmax(-1)[:, 1]

            torch.save(train_probs, preds_path)

        del trainer

        w2s_train = (
            strong_trains[strong_name]
            .remove_columns("labels")
            .add_column("labels", train_probs.numpy())
        )

        name = f"s2s_iter{i + 1}" + cfg.run_name
        train_cfg["output_dir"] = str(root / name)
        train_cfg["run_name"] = cfg.dataset + "/" + name

        trainer = DistillationTrainer(
            args=TrainingArguments(**train_cfg),  # type: ignore
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(strong_tokenizers[strong_name]),
            eval_dataset=ceil_tests[strong_name],
            model=make_peft(init_strong_model(strong_name)),
            tokenizer=strong_tokenizers[strong_name],
            train_dataset=w2s_train,
        )
        trainer.train()
        wandb.finish()

        preds_path = root / name / "preds.pt"


if __name__ == "__main__":
    train(parse(TrainConfig))
