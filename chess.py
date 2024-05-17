from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Sequence, Value
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
)
from simple_parsing import Serializable, field, parse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import wandb
from w2s.ds_registry import load_and_process_dataset


@dataclass
class TrainConfig(Serializable):
    weak_name: str
    """Name of the weak model to use."""

    dataset: str = field(positional=True)
    """Name of the dataset to use."""

    minibatch_size: int = 8
    """Size of the minibatches to use during training."""

    outlier_k: int = field(default=0)
    """Number of neighbors to consider when removing outliers."""

    run_name: str = ""
    """Name of the run."""

    s2s_iter: int = 0
    """Number of strong-to-strong iterations to perform."""


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
    print(f"Best model (accuracy {perf:.3f}) saved at: {dest}")


def lolcat(lol1, lol2):
    # list-of-list concatenation along the second dimension
    assert len(lol1) == len(lol2)
    return [l1 + l2 for l1, l2 in zip(lol1, lol2)]


def lolconst(lol, const):
    return [[const for _ in l_] for l_ in lol]


def synthetic_labels(labels, predictions):
    synthetic_labels = []
    i = 0
    for label in labels:
        label = torch.tensor(label)
        mask = label != -100
        synthetic_labels.append(
            torch.cat([label[~mask], predictions[i : i + mask.sum()]])
        )
        i += mask.sum()
        assert label.shape == synthetic_labels[-1].shape
    return synthetic_labels


def preprocess_logits_for_metrics(logits, labels):
    logits = logits[:, :-1][labels[:, 1:] != -100]
    labels = labels[labels != -100]

    accuracy = logits.argmax(dim=-1).eq(labels).float().mean().item()
    loss = torch.nn.functional.cross_entropy(logits, labels).item()
    return torch.tensor([accuracy, loss])


def compute_metrics(eval_pred):
    processed_logits, labels = map(torch.from_numpy, eval_pred)
    breakpoint()
    # extract metrics packed into single tensor in preprocess_logits_for_metrics
    return dict(
        accuracy=processed_logits[::2].mean().item(),
        loss=processed_logits[1::2].mean().item(),
    )


def preprocess_logits_for_predictions(logits, labels):
    label_logits = logits[:, :-1][labels[:, 1:] != -100]
    return label_logits.argmax(dim=-1)


def train(cfg: TrainConfig):
    lora_cfg = LoraConfig(target_modules=LORA_MODULES)

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
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype="auto", device_map={"": "cuda"}
        )
        model.config.pad_token_id = strong_tokenizers[name].pad_token_id  # type: ignore
        return model

    def weak_processor(examples):
        ctx_out = weak_tokenizer(examples["ctx"], truncation=True)
        trg_out = weak_tokenizer(
            examples["target"], truncation=True, add_special_tokens=False
        )
        return dict(
            input_ids=lolcat(ctx_out["input_ids"], trg_out["input_ids"]),
            attention_mask=lolcat(ctx_out["attention_mask"], trg_out["attention_mask"]),
            labels=lolcat(lolconst(ctx_out["input_ids"], -100), trg_out["input_ids"]),
        )

    cols = ["ctx", "target"]
    test = splits["test"].select_columns(cols)
    train = splits["train"].select_columns(cols)
    print(f"Train example:\n\n{train[0]['ctx']}\n\nLabel: {train[0]['target']}")

    weak_test = test.map(weak_processor, batched=True).cast_column(
        "labels", Sequence(Value("int64"))
    )
    weak_train = train.map(weak_processor, batched=True).cast_column(
        "labels", Sequence(Value("int64"))
    )

    root = Path("results") / cfg.dataset
    train_cfg = dict(
        output_dir=str(root / "floor"),
        num_train_epochs=2,
        adam_beta2=0.95,
        gradient_accumulation_steps=8 // cfg.minibatch_size,
        evaluation_strategy="epoch",
        label_names=["labels"],
        load_best_model_at_end=True,
        logging_steps=50,
        metric_for_best_model="accuracy",
        per_device_train_batch_size=cfg.minibatch_size,
        per_device_eval_batch_size=cfg.minibatch_size,
        run_name=cfg.dataset + "/floor" + cfg.run_name,
        save_strategy="epoch",
        save_total_limit=1,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=3e-5,
    )

    # Gather weak labels
    label_dir = root / "floor/preds"
    if label_dir.exists():
        print(f"Loading weak labels from {label_dir}")
        train_predictions = torch.load(label_dir / "train.pt")
        test_predictions = torch.load(label_dir / "test.pt")
    else:
        should_train = True
        weak_path = root / "floor/best-ckpt"
        if not weak_path.exists():
            weak_model = AutoModelForCausalLM.from_pretrained(
                cfg.weak_name, torch_dtype="auto"
            )
            weak_model = get_peft_model(weak_model, lora_cfg)
        else:
            print("Loading weak model from:", weak_path)
            should_train = False

            weak_model = AutoPeftModelForCausalLM.from_pretrained(
                weak_path, torch_dtype="auto"
            )

        weak_model.config.pad_token_id = weak_tokenizer.pad_token_id
        trainer = Trainer(
            args=TrainingArguments(**train_cfg),  # type: ignore
            compute_metrics=compute_metrics,
            data_collator=DataCollatorForTokenClassification(weak_tokenizer),
            eval_dataset=weak_test,
            model=weak_model,
            tokenizer=weak_tokenizer,
            train_dataset=weak_train,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        if should_train:
            print("\n\033[32m===== Training weak model =====\033[0m")
            trainer.train()
            wandb.finish()
            move_best_ckpt(trainer)

        print("Gathering weak labels")

        trainer.preprocess_logits_for_metrics = preprocess_logits_for_predictions
        trainer.compute_metrics = lambda _: dict()
        train_predictions = torch.from_numpy(trainer.predict(weak_train).predictions)
        test_predictions = torch.from_numpy(trainer.predict(weak_test).predictions)

        label_dir.mkdir(parents=True, exist_ok=True)
        torch.save(train_predictions, label_dir / "train.pt")
        torch.save(test_predictions, label_dir / "test.pt")

    def strong_processor(examples, tokenizer):
        ctx_out = tokenizer(examples["ctx"], truncation=True)
        trg_out = tokenizer(
            examples["target"], truncation=True, add_special_tokens=False
        )
        return dict(
            input_ids=lolcat(ctx_out["input_ids"], trg_out["input_ids"]),
            attention_mask=lolcat(ctx_out["attention_mask"], trg_out["attention_mask"]),
            labels=lolcat(lolconst(ctx_out["input_ids"], -100), trg_out["input_ids"]),
        )

    strong_trains = {
        name: train.map(
            strong_processor,
            batched=True,
            fn_kwargs={"tokenizer": strong_tokenizers[name]},
        ).cast_column("labels", Sequence(Value("int64")))
        for name in STRONG_NAMES
    }
    ceil_tests = {
        name: test.map(
            strong_processor,
            batched=True,
            fn_kwargs={"tokenizer": strong_tokenizers[name]},
        ).cast_column("labels", Sequence(Value("int64")))
        for name in STRONG_NAMES
    }

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
            data_collator=DataCollatorForTokenClassification(
                strong_tokenizers[MAIN_STRONG_NAME]
            ),
            eval_dataset=ceil_tests[MAIN_STRONG_NAME],
            model=get_peft_model(init_strong_model(MAIN_STRONG_NAME), lora_cfg),
            tokenizer=strong_tokenizers[MAIN_STRONG_NAME],
            train_dataset=strong_trains[MAIN_STRONG_NAME],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        trainer.train()
        wandb.finish()
        move_best_ckpt(trainer)

    # Weak to strong generalization
    w2s_train = strong_trains[MAIN_STRONG_NAME].cast_column(
        "labels", Sequence(Value("int64"))
    )

    train_weak_labels = synthetic_labels(w2s_train["labels"], train_predictions)
    w2s_train = w2s_train.remove_columns("labels").add_column(
        "labels", [train_weak_labels[i].numpy() for i in range(len(train_weak_labels))]
    )

    # Check gt metrics every 100 steps during w2s training.
    # We can overfit to the weak labels before a single epoch.
    train_cfg["evaluation_strategy"] = "steps"
    train_cfg["save_strategy"] = "steps"
    train_cfg["eval_steps"] = 100
    train_cfg["save_steps"] = 100
    train_cfg["label_names"] = ["labels"]
    train_cfg["output_dir"] = str(root / ("w2s" + cfg.run_name))
    train_cfg["run_name"] = cfg.dataset + "/w2s" + cfg.run_name

    should_train = True
    w2s_ckpt = root / ("w2s" + cfg.run_name) / "best-ckpt"
    if w2s_ckpt.exists():
        print(f"W2S model already exists at {w2s_ckpt}")

        w2s_model = AutoPeftModelForCausalLM.from_pretrained(
            w2s_ckpt, torch_dtype="auto", device_map={"": "cuda"}
        )
        should_train = False
    else:
        print("\n\033[32m===== Training w2s model =====\033[0m")
        strong_model = init_strong_model(MAIN_STRONG_NAME)
        w2s_model = get_peft_model(strong_model, lora_cfg)

    w2s_model.config.pad_token_id = strong_tokenizers[MAIN_STRONG_NAME].pad_token_id  # type: ignore # noqa
    trainer = Trainer(
        args=TrainingArguments(**train_cfg),  # type: ignore
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForTokenClassification(
            strong_tokenizers[MAIN_STRONG_NAME]
        ),
        eval_dataset=ceil_tests[MAIN_STRONG_NAME],
        model=w2s_model,
        tokenizer=strong_tokenizers[MAIN_STRONG_NAME],
        train_dataset=w2s_train,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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
            train_predictions = torch.load(preds_path)
        else:
            trainer.preprocess_logits_for_metrics = preprocess_logits_for_predictions
            trainer.compute_metrics = lambda _: dict()
            train_predictions = torch.from_numpy(trainer.predict(w2s_train).predictions)

            torch.save(train_predictions, preds_path)

        del trainer
        train_strong_labels = synthetic_labels(w2s_train["labels"], train_predictions)

        w2s_train = (
            strong_trains[strong_name]
            .remove_columns("labels")
            .add_column(
                "labels",
                [
                    train_strong_labels[i].numpy()
                    for i in range(len(train_strong_labels))
                ],
            )
        )

        name = f"s2s_iter{i + 1}" + cfg.run_name
        train_cfg["output_dir"] = str(root / name)
        train_cfg["run_name"] = cfg.dataset + "/" + name

        trainer = Trainer(
            args=TrainingArguments(**train_cfg),  # type: ignore
            compute_metrics=compute_metrics,
            data_collator=DataCollatorForTokenClassification(
                strong_tokenizers[strong_name]
            ),
            eval_dataset=ceil_tests[strong_name],
            model=get_peft_model(init_strong_model(strong_name), lora_cfg),
            tokenizer=strong_tokenizers[strong_name],
            train_dataset=w2s_train,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        trainer.train()
        wandb.finish()

        preds_path = root / name / "preds.pt"


if __name__ == "__main__":
    train(parse(TrainConfig))
