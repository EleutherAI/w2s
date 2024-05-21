import json
from pathlib import Path
from typing import Union

import torch
from datasets import DatasetDict
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import wandb
from underspec.loss import log_confidence_loss
from underspec.model import ModelConfig, init_model_and_tokenizer
from underspec.roc_auc import roc_auc
from underspec.sft_utils import (
    clear_mem,
    get_gpu_mem_used,
    move_best_ckpt,
)


class CustomLossTrainer(Trainer):
    def __init__(
        self, logconf_weight: float, logconf_warmup_steps: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.logconf_weight = logconf_weight
        self.logconf_warmup_steps = logconf_warmup_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()

        outputs = model(**inputs)

        loss = log_confidence_loss(
            outputs.logits,
            labels,
            self.state.global_step,
            aux_coef=self.logconf_weight,
            warmup_steps=self.logconf_warmup_steps,
        )

        return (loss, outputs) if return_outputs else loss


def train(
    ds_dict: DatasetDict,
    model_cfg: ModelConfig,
    train_args: TrainingArguments,
    cfg: dict,
    logconf_weight: float = 0.0,
    logconf_warmup_steps: int = 200,
    predict_dict: Union[DatasetDict, dict, None] = None,
):
    """
    ds_dict: DatasetDict with splits for train, val, test, and (optionally) predict,
        with columns "txt" and "labels"
    model_cfg: ModelConfig with the model name and whether to enable LoRA
    train_args: TrainingArguments with the training hyperparameters
    logconf_weight: the weight for the log confidence loss
    store_pre_hiddens: whether to store the hiddens (all layers,
        final token position, on train set) before training
    store_post_hiddens: whether to store the hiddens after training
    cfg: a dictionary containing all the relevant details for reproducibility.
        This will be updated with your train_args and model_cfg before saving.

    This function trains a model on ds_dict["train"], uses ds_dict["val"] for early stopping,
        and evaluates on ds_dict["test"].
    It also optionally predicts on ds_dict["predict"] and saves the predictions.
    """
    save_dir = Path(train_args.output_dir)
    results_path = save_dir / "results.json"
    if results_path.exists():
        print(
            f"Results already exist at {results_path}. Skipping training and evaluation."
        )
        return

    clear_mem()
    print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use before training")

    model, tokenizer = init_model_and_tokenizer(model_cfg)

    def process(examples):
        out = tokenizer(examples["txt"], truncation=True)
        return out

    ds_dict = ds_dict.map(process, batched=True)

    def compute_metrics(eval_pred):
        predictions, labels = map(torch.from_numpy, eval_pred)
        hard_labels = (labels > 0.5).long()
        return dict(
            accuracy=predictions.argmax(dim=1).eq(hard_labels).float().mean(),
            auroc=roc_auc(hard_labels, predictions[:, 1]),
        )

    trainer = CustomLossTrainer(
        logconf_weight=logconf_weight,
        logconf_warmup_steps=logconf_warmup_steps,
        args=train_args,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        eval_dataset={k: ds_dict[k] for k in ["val", "test"]},
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_dict["train"],
    )

    # train
    trainer.train()

    # evaluate on test dataset
    eval_results = trainer.evaluate(ds_dict["test"])  # type: ignore
    move_best_ckpt(trainer)

    # save results
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    # save config
    with open(save_dir / "config.json", "w") as f:
        cfg["model"] = model_cfg.to_dict()
        cfg["train_args"] = train_args.to_dict()
        cfg["logconf_weight"] = logconf_weight
        json.dump(cfg, f, indent=2)
    wandb.config.update(cfg)

    # save predictions
    if predict_dict is not None:
        for name, predict_ds in predict_dict.items():
            predict_ds = predict_ds.map(process, batched=True)
            print("Gathering predictions for", name)
            pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)
            preds = pred_logits.softmax(-1)[:, 1].cpu().float().numpy()
            pred_ds = predict_ds.add_column("soft_pred", preds)

            pred_ds.save_to_disk(str(save_dir / "predictions" / name))

    wandb.finish()
