from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from simple_parsing import parse
from transformers import (
    TrainingArguments,
)

from underspec.ds_registry import load_and_process_dataset
from underspec.model import ModelConfig
from underspec.sft import train
from underspec.sft_config import SFTConfig
from underspec.utils import get_config_foldername


def run_train(cfg: SFTConfig):
    splits = load_and_process_dataset(
        cfg.dataset, cfg.n_train, cfg.n_val, cfg.n_test, cfg.n_predict
    )

    cols = ["hard_label", "txt"]
    splits = splits.select_columns(cols).rename_column("hard_label", "labels")
    print(
        f"Example:\n\n{splits['train'][0]['txt']}\n\nLabel: {splits['train'][0]['labels']}"
    )

    root = Path(cfg.results_folder) / cfg.run_name
    cfg_name = get_config_foldername(vars(cfg))
    train_args: dict = dict(
        num_train_epochs=cfg.n_epochs,
        adam_beta2=0.95,
        gradient_accumulation_steps=cfg.batch_size // cfg.minibatch_size,
        eval_strategy="steps",
        label_names=["labels"],
        load_best_model_at_end=cfg.load_best_model_at_end,
        logging_steps=25,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        per_device_train_batch_size=cfg.minibatch_size,
        per_device_eval_batch_size=cfg.minibatch_size,
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        warmup_steps=cfg.n_warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_schedule,
        eval_steps=cfg.eval_every,
        save_steps=cfg.save_every,
    )

    def get_model_and_run_name(model_name, current_name):
        model_last = model_name.split("/")[-1]
        model_cfg = ModelConfig(name=model_name, enable_lora=not cfg.disable_lora)
        run_name = f"{current_name}-{cfg.run_name}-{cfg.dataset}-{model_last}"
        return model_cfg, run_name

    # train weak floor, get predictions
    print("\n\033[32m===== Training weak model =====\033[0m")
    model_cfg, run_name = get_model_and_run_name(cfg.weak_model_name, "weak")
    train_args["run_name"] = run_name
    train_args["output_dir"] = str(root / cfg_name / "weak")
    train_args["learning_rate"] = cfg.weak_lr
    weak_ds_dict = DatasetDict(
        {
            "train": splits["train"],
            "val": splits["val"],
            "test": splits["test"],
        }
    )
    weak_predict_dict = {"train": splits["train"], "val": splits["val"]}
    train(
        weak_ds_dict,
        model_cfg,
        TrainingArguments(**train_args),
        cfg.to_dict(),
        predict_dict=weak_predict_dict,
    )

    # train strong ceil
    print("\n\033[32m===== Training strong model =====\033[0m")
    model_cfg, run_name = get_model_and_run_name(cfg.strong_model_name, "strong")
    train_args["run_name"] = run_name
    train_args["output_dir"] = str(root / cfg_name / "strong")
    train_args["learning_rate"] = cfg.strong_lr
    strong_ds_dict = DatasetDict(
        {
            "train": splits["train"],
            "val": splits["val"],
            "test": splits["test"],
        }
    )
    train(strong_ds_dict, model_cfg, TrainingArguments(**train_args), cfg.to_dict())

    # load weak predictions
    weak_preds_root = root / cfg_name / "weak" / "predictions"
    weak_train_preds_ds = load_from_disk(str(weak_preds_root / "train"))
    weak_val_preds_ds = load_from_disk(str(weak_preds_root / "val"))

    # train w2s with logconf, get predictions
    print("\n\033[32m===== Training w2s model =====\033[0m")
    model_cfg, run_name = get_model_and_run_name(cfg.strong_model_name, "w2s")
    train_args["run_name"] = run_name
    train_args["output_dir"] = str(root / cfg_name / "w2s")
    train_args["learning_rate"] = cfg.strong_lr
    w2s_ds_dict = DatasetDict(
        {
            "train": (
                splits["train"]
                .remove_columns("labels")
                .add_column("labels", weak_train_preds_ds["soft_pred"])  # type: ignore
            ),
            "val": (
                splits["val"]
                .remove_columns("labels")
                .add_column("labels", weak_val_preds_ds["soft_pred"])
            ),  # type: ignore
            "test": splits["test"],
        }
    )
    # assert (weak_train_preds_ds["id"] == w2s_ds_dict["train"]["id"])
    # assert (weak_val_preds_ds["id"] == w2s_ds_dict["val"]["id"])
    w2s_predict_dict = {"train": splits["train"], "val": splits["val"]}
    train(
        w2s_ds_dict,
        model_cfg,
        TrainingArguments(**train_args),
        cfg.to_dict(),
        predict_dict=w2s_predict_dict,
        logconf_weight=cfg.logconf_weight,
        logconf_warmup_steps=cfg.logconf_warmup_steps,
    )

    # load w2s predictions, and balanced-harden them
    print("\n\033[32m===== Training (s+w)2s model =====\033[0m")
    w2s_preds_root = root / cfg_name / "w2s" / "predictions"
    w2s_train_preds_ds = load_from_disk(str(w2s_preds_root / "train")).with_format(
        type="torch", columns=["soft_pred"]
    )
    w2s_val_preds_ds = load_from_disk(str(w2s_preds_root / "val")).with_format(
        type="torch", columns=["soft_pred"]
    )
    prior = torch.tensor(splits["train"]["labels"]).float().mean()
    thresh = torch.quantile(w2s_train_preds_ds["soft_pred"], 1 - prior)  # type: ignore
    # set the label column of train to be (1 - a) * weak + a * hard_w2s
    sw2s_train_labels = (
        (
            (1 - cfg.strong_weight) * torch.tensor(weak_train_preds_ds["soft_pred"])  # type: ignore
            + cfg.strong_weight * (w2s_train_preds_ds["soft_pred"] > thresh).float()
        )
        .float()
        .tolist()
    )
    sw2s_val_labels = (
        (
            (1 - cfg.strong_weight) * torch.tensor(weak_val_preds_ds["soft_pred"])  # type: ignore
            + cfg.strong_weight * (w2s_val_preds_ds["soft_pred"] > thresh).float()
        )
        .float()
        .tolist()
    )

    # train sw2s
    model_cfg, run_name = get_model_and_run_name(cfg.strong_model_name, "sw2s")
    train_args["run_name"] = run_name
    train_args["output_dir"] = str(root / cfg_name / "sw2s")
    train_args["learning_rate"] = cfg.strong_lr
    sw2s_ds_dict = DatasetDict(
        {
            "train": (
                splits["train"]
                .remove_columns("labels")
                .add_column("labels", sw2s_train_labels)  # type: ignore
            ),
            "val": (
                splits["val"]
                .remove_columns("labels")
                .add_column("labels", sw2s_val_labels)  # type: ignore
            ),
            "test": splits["test"],
        }
    )
    # assert (w2s_train_preds_ds["id"] == sw2s_ds_dict["train"]["id"])
    # assert (w2s_val_preds_ds["id"] == sw2s_ds_dict["val"]["id"])

    train(
        sw2s_ds_dict,
        model_cfg,
        TrainingArguments(**train_args),
        cfg.to_dict(),
        logconf_weight=cfg.logconf_weight,
        logconf_warmup_steps=cfg.logconf_warmup_steps,
    )


if __name__ == "__main__":
    run_train(parse(SFTConfig))
