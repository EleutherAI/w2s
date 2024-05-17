import json
from pathlib import Path

import torch
from datasets import DatasetDict, Value
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
    gather_hiddens,
    get_gpu_mem_used,
    move_best_ckpt,
)


class CustomLossTrainer(Trainer):
    def __init__(self, loss_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()

        outputs = model(**inputs)
        if self.loss_name == "xent":
            labels = torch.stack([1.0 - labels, labels], dim=-1)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
        elif self.loss_name == "logconf":
            loss = log_confidence_loss(outputs.logits, labels, self.state.global_step)
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

        return (loss, outputs) if return_outputs else loss


def train(
    ds_dict: DatasetDict,
    model_cfg: ModelConfig,
    train_args: TrainingArguments,
    loss: str,
    store_pre_hiddens: bool,
    store_post_hiddens: bool,
    cfg: dict,
):
    """
    ds_dict: DatasetDict with splits for train, val, test, and (optionally) predict,
        with columns "txt" and "labels"
    model_cfg: ModelConfig with the model name and whether to enable LoRA
    train_args: TrainingArguments with the training hyperparameters
    loss: a string indicating the loss function to use
    store_pre_hiddens: whether to store the hiddens (all layers,
        final token position, on train set) before training
    store_post_hiddens: whether to store the hiddens after training
    cfg: a dictionary containing all the relevant details for reproducibility.
        This will be updated with your train_args and model_cfg before saving.

    This function trains a model on ds_dict["train"], uses ds_dict["val"] for early stopping,
        and evaluates on ds_dict["test"].
    It also optionally predicts on ds_dict["predict"] and saves the predictions.
    """
    clear_mem()
    print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use before training")

    save_dir = Path(train_args.output_dir)
    results_path = save_dir / "results.json"
    if results_path.exists():
        print(f"Results already exist at {results_path}. Skipping training.")
        return

    model, tokenizer = init_model_and_tokenizer(model_cfg)

    def process(examples):
        out = tokenizer(examples["txt"], truncation=True)
        return out

    ds_dict = ds_dict.map(process, batched=True).cast_column("labels", Value("int64"))

    def compute_metrics(eval_pred):
        predictions, labels = map(torch.from_numpy, eval_pred)
        return dict(
            accuracy=predictions.argmax(dim=1).eq(labels).float().mean(),
            auroc=roc_auc(labels, predictions[:, 1]),
        )

    trainer = CustomLossTrainer(
        loss_name=loss,
        args=train_args,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        eval_dataset=ds_dict[
            "val"
        ],  # TODO make sure this doesn't use ground truth labels for w2s runs
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_dict["train"],
    )

    # store pre hiddens
    if store_pre_hiddens:
        print("Gathering hiddens")
        hiddens = gather_hiddens(model, ds_dict["train"])
        torch.save(hiddens, save_dir / "pre_hiddens.pt")

    print("\n\033[32m===== Training weak model =====\033[0m")
    # train
    trainer.train()

    # evaluate on test dataset
    eval_results = trainer.evaluate(ds_dict["test"])  # type: ignore
    wandb.finish()
    move_best_ckpt(trainer)

    # save results
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    # save config
    with open(save_dir / "config.json", "w") as f:
        cfg["model"] = model_cfg.to_dict()
        cfg["train_args"] = train_args.to_dict()
        json.dump(cfg, f, indent=2)

    # save predictions
    if "predict" in ds_dict:
        print("Gathering weak labels")
        pred_logits: torch.Tensor = trainer.predict(ds_dict["predict"]).predictions  # type: ignore
        preds = pred_logits.softmax(-1)[:, 1].cpu().float().numpy()
        pred_ds = ds_dict["predict"].add_column("soft_pred", preds)  # type: ignore

        pred_ds.save_to_disk(save_dir / "predictions")

    # save hiddens
    if store_post_hiddens:
        print("Gathering hiddens")
        hiddens = gather_hiddens(model, ds_dict["train"])
        torch.save(hiddens, save_dir / "post_hiddens.pt")
