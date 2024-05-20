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
from w2s.loss import log_confidence_loss
from w2s.roc_auc import roc_auc

disable_caching()


@dataclass
class TrainConfig(Serializable):
    dataset: str = field(positional=True)
    """Name of the dataset to use."""

    weak_name: str = "Qwen/Qwen1.5-0.5B"
    """Name of the weak model to use."""

    minibatch_size: int = 8
    """Size of the minibatches to use during training."""

    run_name: str = ""
    """Name of the run."""

    s2s_iter: int = 0
    """Number of strong-to-strong iterations to perform."""

    strong_model: str = "meta-llama/Meta-Llama-3-8B"
    """Name of the main strong model to use."""

    ood_boost: float = 0.0
    """Boost factor for the OOD logits in the S2S training."""

    def to_dict(self):
        return vars(self)


class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()

        outputs = model(**inputs)
        loss = log_confidence_loss(
            outputs.logits, labels, self.state.global_step, aux_coef=0.0
        )

        return (loss, outputs) if return_outputs else loss


# Works for Llama, Mistral, Gemma, and Qwen architectures
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
    if trainer.args.load_best_model_at_end:
        path = trainer.state.best_model_checkpoint
        perf = trainer.state.best_metric
        assert path is not None, "No best checkpoint found"
        assert perf is not None, "No best metric"

        src = Path(path)
        dest = src.parent / "best-ckpt"
        src.rename(dest)
        metric = trainer.args.metric_for_best_model
        print(f"Best model ({metric}={perf:.3f}) saved at: {dest}")


def train(cfg: TrainConfig):
    # train weak on union, predict on union
    # train strong on union
    # train w2s on train1, predict on train1 and train2
    # train w2s on train2, predict on train1 and train2
    # train s2s on train1 OOD, predict on train1 and train2
    # train s2s on train2 OOD, predict on train1 and train2...

    lora_cfg = LoraConfig(target_modules=LORA_MODULES)
    lrs = {
        "Qwen/Qwen1.5-0.5B": 5e-5,
        "meta-llama/Meta-Llama-3-8B": 8e-5,
        "mistralai/Mistral-7B-v0.1": 3e-5,
        "google/gemma-7b": 5e-5,
    }

    # for 2 strong models we do WEAK -> STRONG[0] -> STRONG[1] -> STRONG[0] -> ...
    STRONG_NAMES = [
        cfg.strong_model,
    ]  # "google/gemma-7b"]
    MAIN_STRONG_NAME = STRONG_NAMES[0]
    strong_tokenizers = {k: AutoTokenizer.from_pretrained(k) for k in STRONG_NAMES}
    weak_tokenizer = AutoTokenizer.from_pretrained(cfg.weak_name)

    # Make sure that the pad token is set
    for t in strong_tokenizers.values():
        if t.pad_token_id is None:
            t.pad_token = t.eos_token
    if weak_tokenizer.pad_token_id is None:
        weak_tokenizer.pad_token = weak_tokenizer.eos_token

    def init_strong_model(name):
        model = AutoModelForSequenceClassification.from_pretrained(
            name, torch_dtype="auto", device_map={"": "cuda"}
        )
        model.config.pad_token_id = strong_tokenizers[name].pad_token_id  # type: ignore
        model.score.weight.data *= 0.01
        return model

    def weak_processor(examples):
        out = weak_tokenizer(examples["txt"], truncation=True)
        out["labels"] = examples["hard_label"]
        return out

    def compute_metrics(eval_pred):
        predictions, labels = map(torch.from_numpy, eval_pred)
        hard_labels = (labels > 0.5).long()
        return dict(
            accuracy=predictions.argmax(dim=1).eq(hard_labels).float().mean(),
            auroc=roc_auc(hard_labels, predictions[:, 1]),
        )

    splits = load_and_process_dataset(
        cfg.dataset, split_sizes=dict(train=20_000, val=1_000, test=1_000)
    )
    trains = splits["train"].train_test_split(test_size=0.5, shuffle=False)
    trains = [trains["train"], trains["test"]]

    cols = ["hard_label", "txt"]
    train_union = splits["train"].select_columns(cols)
    trains = [t.select_columns(cols) for t in trains]
    val = splits["val"].select_columns(cols)
    test = splits["test"].select_columns(cols)
    print(
        f"Train example:\n\n{trains[0][0]['txt']}\n\nLabel: {trains[0][0]['hard_label']}"  # noqa
    )

    weak_train_union = train_union.map(weak_processor, batched=True).cast_column(
        "labels", Value("int64")
    )
    weak_trains = [
        t.map(weak_processor, batched=True).cast_column("labels", Value("int64"))
        for t in trains
    ]
    weak_val = val.map(weak_processor, batched=True).cast_column(
        "labels", Value("int64")
    )
    weak_test = test.map(weak_processor, batched=True).cast_column(
        "labels", Value("int64")
    )

    root = Path("s2s-results") / cfg.dataset
    train_cfg = dict(
        output_dir=str(root / "floor"),
        max_steps=1_000,
        adam_beta2=0.95,
        gradient_accumulation_steps=8 // cfg.minibatch_size,
        evaluation_strategy="epoch",
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="val_auroc",
        greater_is_better=True,
        logging_steps=50,
        per_device_train_batch_size=cfg.minibatch_size,
        per_device_eval_batch_size=cfg.minibatch_size,
        run_name=cfg.dataset + "/floor" + cfg.run_name,
        save_strategy="epoch",
        save_total_limit=1,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        warmup_steps=40,
        weight_decay=0.01,
        learning_rate=lrs[cfg.weak_name],
    )

    # Gather weak labels
    label_dir = root / "floor/preds"
    if (label_dir / "train1.pt").exists() and (label_dir / "train2.pt").exists():
        print(f"Loading weak labels from {label_dir}")
        trains_probs = [
            torch.load(label_dir / "train1.pt"),
            torch.load(label_dir / "train2.pt"),
        ]
        val_probs = torch.load(label_dir / "val.pt")
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

        weak_model.config.pad_token_id = weak_tokenizer.pad_token_id  # type: ignore
        trainer = Trainer(
            args=TrainingArguments(**train_cfg),  # type: ignore
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(weak_tokenizer),
            eval_dataset={"val": weak_val, "test": weak_test},
            model=weak_model,
            tokenizer=weak_tokenizer,
            train_dataset=weak_train_union,
        )
        if should_train:
            print("\n\033[32m===== Training weak model =====\033[0m")
            trainer.train()
            wandb.config.update(cfg.to_dict())
            wandb.finish()
            move_best_ckpt(trainer)

        print("Gathering weak labels")
        trains_logits = [trainer.predict(wt).predictions for wt in weak_trains]
        val_logits = trainer.predict(weak_val).predictions
        test_logits = trainer.predict(weak_test).predictions

        # Convert to probabilities, then keep only the positive probs
        trains_probs = [torch.from_numpy(tl).softmax(-1)[:, 1] for tl in trains_logits]
        val_probs = torch.from_numpy(val_logits).softmax(-1)[:, 1]
        test_probs = torch.from_numpy(test_logits).softmax(-1)[:, 1]

        label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(trains_probs)):
            torch.save(trains_probs[i], label_dir / f"train{i + 1}.pt")
        torch.save(val_probs, label_dir / "val.pt")
        torch.save(test_probs, label_dir / "test.pt")

    def strong_processor(examples, tokenizer):
        return tokenizer(examples["txt"], truncation=True)

    strongs_trains = {
        name: [
            t.map(
                strong_processor,
                batched=True,
                fn_kwargs={"tokenizer": strong_tokenizers[name]},
            )
            .rename_column("hard_label", "labels")
            .cast_column("labels", Value("int64"))
            for t in trains
        ]
        for name in STRONG_NAMES
    }
    main_strong_train_union = (
        train_union.map(
            strong_processor,
            batched=True,
            fn_kwargs={"tokenizer": strong_tokenizers[MAIN_STRONG_NAME]},
        )
        .rename_column("hard_label", "labels")
        .cast_column("labels", Value("int64"))
    )
    strongs_val = {
        name: val.map(
            strong_processor,
            batched=True,
            fn_kwargs={"tokenizer": strong_tokenizers[name]},
        )
        .rename_column("hard_label", "labels")
        .cast_column("labels", Value("int64"))
        for name in STRONG_NAMES
    }
    strongs_test = {
        name: test.map(
            strong_processor,
            batched=True,
            fn_kwargs={"tokenizer": strong_tokenizers[name]},
        )
        .rename_column("hard_label", "labels")
        .cast_column("labels", Value("int64"))
        for name in STRONG_NAMES
    }

    strong_ckpt = root / "ceil" / "best-ckpt"
    if strong_ckpt.exists():
        print(f"Strong ceiling model already exists at {strong_ckpt}")
    else:
        print("\n\033[32m===== Training strong ceiling model =====\033[0m")
        train_cfg["output_dir"] = str(root / "ceil")
        train_cfg["run_name"] = cfg.dataset + "/ceil" + cfg.run_name
        train_cfg["learning_rate"] = lrs[MAIN_STRONG_NAME]

        trainer = Trainer(
            args=TrainingArguments(**train_cfg),  # type: ignore
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(strong_tokenizers[MAIN_STRONG_NAME]),
            eval_dataset={
                "val": strongs_val[MAIN_STRONG_NAME],
                "test": strongs_test[MAIN_STRONG_NAME],
            },
            model=get_peft_model(init_strong_model(MAIN_STRONG_NAME), lora_cfg),
            tokenizer=strong_tokenizers[MAIN_STRONG_NAME],
            train_dataset=main_strong_train_union,
        )
        trainer.train()
        wandb.config.update(cfg.to_dict())
        wandb.finish()
        move_best_ckpt(trainer)

    # Weak to strong generalization
    w2s_trains = strongs_trains[MAIN_STRONG_NAME]
    w2s_trains = [
        t.remove_columns("labels").add_column("labels", tp.numpy())
        for t, tp in zip(w2s_trains, trains_probs)
    ]
    w2s_val = (
        strongs_val[MAIN_STRONG_NAME]
        .remove_columns("labels")
        .add_column("labels", val_probs.numpy())
    )

    for i, w2s_train in enumerate(w2s_trains):
        out_path = root / (f"w2s{i + 1}" + cfg.run_name)
        # Check gt metrics every 100 steps during w2s training.
        # We can overfit to the weak labels before a single epoch.
        train_cfg["evaluation_strategy"] = "steps"
        train_cfg["save_strategy"] = "steps"
        train_cfg["eval_steps"] = 100
        train_cfg["save_steps"] = 100
        train_cfg["label_names"] = ["labels"]
        train_cfg["output_dir"] = str(out_path)
        train_cfg["run_name"] = cfg.dataset + f"/w2s{i + 1}" + cfg.run_name
        train_cfg["learning_rate"] = lrs[MAIN_STRONG_NAME]

        should_train = True
        w2s_ckpt = out_path / "best-ckpt"
        if w2s_ckpt.exists():
            print(f"W2S model already exists at {w2s_ckpt}")

            w2s_model = AutoPeftModelForSequenceClassification.from_pretrained(
                w2s_ckpt, torch_dtype="auto", device_map={"": "cuda"}
            )
            should_train = False
        else:
            print(f"\n\033[32m===== Training w2s{i + 1} model =====\033[0m")

            strong_model = init_strong_model(MAIN_STRONG_NAME)
            w2s_model = get_peft_model(strong_model, lora_cfg)

        w2s_model.config.pad_token_id = strong_tokenizers[MAIN_STRONG_NAME].pad_token_id  # type: ignore # noqa
        trainer = DistillationTrainer(
            args=TrainingArguments(**train_cfg),  # type: ignore
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(strong_tokenizers[MAIN_STRONG_NAME]),
            eval_dataset={"val": w2s_val, "test": strongs_test[MAIN_STRONG_NAME]},
            model=w2s_model,
            tokenizer=strong_tokenizers[MAIN_STRONG_NAME],
            train_dataset=w2s_train,
        )
        if should_train:
            trainer.train()
            wandb.config.update(cfg.to_dict())
            wandb.finish()
            move_best_ckpt(trainer)

        # save preds
        preds_path = out_path / "preds"
        trains_logits = [trainer.predict(t).predictions for t in w2s_trains]
        val_logits = trainer.predict(w2s_val).predictions
        test_logits = trainer.predict(strongs_test[MAIN_STRONG_NAME]).predictions
        trains_probs = [torch.from_numpy(tl).softmax(-1)[:, 1] for tl in trains_logits]
        val_probs = torch.from_numpy(val_logits).softmax(-1)[:, 1]
        test_probs = torch.from_numpy(test_logits).softmax(-1)[:, 1]
        preds_path.mkdir(parents=True, exist_ok=True)
        for j in range(len(trains_probs)):
            torch.save(trains_probs[j], preds_path / f"train{j + 1}.pt")
        torch.save(val_probs, preds_path / "val.pt")
        torch.save(test_probs, preds_path / "test.pt")

        # Save memory
        del w2s_model
        del trainer

    prev_preds_dirs = [
        root / (f"w2s{i + 1}" + cfg.run_name) / "preds" for i in range(len(w2s_trains))
    ]

    # Strong to strong generalization
    for i in range(cfg.s2s_iter):
        names = [f"s2s{j + 1}_iter{i + 1}" + cfg.run_name for j in range(len(trains))]
        for j, name in enumerate(names):
            strong_name = STRONG_NAMES[(i + 1) % len(STRONG_NAMES)]
            print(
                f"\n\033[32m===== S2S{j + 1}-distillation iter {i + 1} ({strong_name}) =====\033[0m"  # noqa
            )

            # Gather strong preds on this set from the
            # previous model trained on the other set
            ood_train_preds_path = prev_preds_dirs[1 - j] / f"train{j + 1}.pt"
            print(f"Loading OOD strong train preds from {ood_train_preds_path}")
            ood_train_probs = torch.load(ood_train_preds_path)
            id_train_preds_path = prev_preds_dirs[j] / f"train{j + 1}.pt"
            print(f"Loading ID strong train preds from {id_train_preds_path}")
            id_train_probs = torch.load(id_train_preds_path)

            # train on id_logodds + (ood_logodds - id_logodds) * (1 + ood_boost)
            id_train_logodds = id_train_probs.log() - (1 - id_train_probs).log()
            ood_train_logodds = ood_train_probs.log() - (1 - ood_train_probs).log()
            train_logodds = id_train_logodds + (
                ood_train_logodds - id_train_logodds
            ) * (1 + cfg.ood_boost)
            # we also re-balance the labels, so that we don't fall
            # into the constant prediction eigenvector
            prior = 0.5
            train_logodds -= torch.quantile(train_logodds, 1 - prior)
            train_probs = torch.sigmoid(train_logodds)
            val_probs = torch.load(prev_preds_dirs[1 - j] / "val.pt")

            s2s_train = (
                strongs_trains[strong_name][j]
                .remove_columns("labels")
                .add_column("labels", train_probs.numpy())
            )
            s2s_val = (
                strongs_val[strong_name]
                .remove_columns("labels")
                .add_column("labels", val_probs.numpy())
            )

            train_cfg["output_dir"] = str(root / name)
            train_cfg["run_name"] = cfg.dataset + "/" + name
            train_cfg["learning_rate"] = lrs[strong_name]

            trainer = DistillationTrainer(
                args=TrainingArguments(**train_cfg),  # type: ignore
                compute_metrics=compute_metrics,
                data_collator=DataCollatorWithPadding(strong_tokenizers[strong_name]),
                eval_dataset={"val": s2s_val, "test": strongs_test[strong_name]},
                model=get_peft_model(init_strong_model(strong_name), lora_cfg),
                tokenizer=strong_tokenizers[strong_name],
                train_dataset=s2s_train,
            )
            trainer.train()
            wandb.config.update(cfg.to_dict())
            wandb.finish()
            move_best_ckpt(trainer)

            # save preds
            preds_path = root / name / "preds"
            trains_logits = [
                trainer.predict(t).predictions for t in strongs_trains[strong_name]
            ]
            val_logits = trainer.predict(s2s_val).predictions
            test_logits = trainer.predict(strongs_test[strong_name]).predictions
            trains_probs = [
                torch.from_numpy(tl).softmax(-1)[:, 1] for tl in trains_logits
            ]
            val_probs = torch.from_numpy(val_logits).softmax(-1)[:, 1]
            test_probs = torch.from_numpy(test_logits).softmax(-1)[:, 1]

            preds_path.mkdir(parents=True, exist_ok=True)
            for k in range(len(trains_probs)):
                torch.save(trains_probs[k], preds_path / f"train{k + 1}.pt")
            torch.save(val_probs, preds_path / "val.pt")
            torch.save(test_probs, preds_path / "test.pt")

            # Save memory
            del trainer

        prev_preds_dirs = [root / name / "preds" for name in names]


if __name__ == "__main__":
    train(parse(TrainConfig))
