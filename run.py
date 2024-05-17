from pathlib import Path

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
    model_last = cfg.model_name.split("/")[-1]
    train_args = TrainingArguments(
        output_dir=str(root / cfg_name),
        num_train_epochs=cfg.n_epochs,
        adam_beta2=0.95,
        gradient_accumulation_steps=cfg.batch_size // cfg.minibatch_size,
        evaluation_strategy="steps",
        label_names=["labels"],
        load_best_model_at_end=True,
        logging_steps=25,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=cfg.minibatch_size,
        per_device_eval_batch_size=cfg.minibatch_size,
        run_name=f"{cfg.run_name}-{cfg.dataset}-{model_last}",
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        warmup_steps=cfg.n_warmup_steps,
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.lr_schedule,
        eval_steps=cfg.eval_every,
        save_steps=cfg.save_every,
    )

    model_cfg = ModelConfig(name=cfg.model_name, enable_lora=not cfg.disable_lora)
    train(
        splits,
        model_cfg,
        train_args,
        cfg.loss,
        cfg.store_pre_hiddens,
        cfg.store_post_hiddens,
        cfg.to_dict(),
    )


if __name__ == "__main__":
    run_train(parse(SFTConfig))
