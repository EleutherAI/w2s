import json
import math
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from datasets import DatasetDict
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import wandb
from w2s.loss import log_confidence_loss
from w2s.model import TransformerPredictor
from w2s.sft_utils import (
    clear_mem,
    compute_acc_and_auroc,
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
            aux_weight = 0
        elif self.loss_name == "logconf":
            aux_weight = 0.5
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

        # print the current learning rate
        loss = log_confidence_loss(
            outputs.logits, labels, self.state.global_step, aux_coef=aux_weight
        )

        return (loss, outputs) if return_outputs else loss


def lm_sft(
    ds_dict: DatasetDict,
    model: TransformerPredictor,
    train_args: TrainingArguments,
    loss: str,
    store_pre_hiddens: bool,
    store_post_hiddens: bool,
    cfg: dict,
    predict_dict: Union[None, Dict, DatasetDict] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """
    ds_dict: DatasetDict with splits for train, val, test,
        with columns "txt" and "labels"
    model: TransformerPredictor model
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
    save_dir = Path(train_args.output_dir)

    clear_mem()
    print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use before training")

    keep_cols = {"labels", "input_ids", "attention_mask"}

    def preprocess(exs):
        out = model.tokenizer(exs["txt"], truncation=True)
        return {k: v for k, v in out.items() if k in keep_cols}

    ds_dict.reset_format()
    ds_dict = ds_dict.map(
        preprocess,
        batched=True,
        remove_columns=list(set(ds_dict["train"].column_names) - keep_cols),
    )

    trainer = CustomLossTrainer(
        loss_name=loss,
        args=train_args,
        compute_metrics=compute_acc_and_auroc,
        data_collator=DataCollatorWithPadding(
            model.tokenizer, max_length=1024
        ),  # NOTE: this could mess up some datasets
        eval_dataset={
            k: ds_dict[k] for k in {"val", "test"}.intersection(ds_dict.keys())
        },
        model=model.transformer,
        tokenizer=model.tokenizer,
        train_dataset=ds_dict["train"],
    )

    results_path = save_dir / "config.json"
    if results_path.exists() and (save_dir / "best-ckpt").exists():
        print(
            f"Results for sft run already exist at {results_path}. "
            "Skipping training and evaluation. Loading the saved model."
        )
        trainer.state.best_model_checkpoint = str(save_dir / "best-ckpt")
        trainer._load_best_model()
        return

    # store pre hiddens
    if store_pre_hiddens:
        print("Gathering hiddens")
        hiddens = gather_hiddens(model.transformer, ds_dict["train"])
        torch.save(hiddens, save_dir / "pre_hiddens.pt")

    # train
    if resume_from_checkpoint is not None:
        # NOTE: this is a hack to get the trainer to load the optimizer state but not the
        # scheduler state or training state by overwriting/deleting them
        with open(f"{resume_from_checkpoint}/trainer_state.json", "r") as f:
            state = json.load(f)
            state["global_step"] = 0
            num_update_steps_per_epoch = len(ds_dict["train"]) // (
                train_args.gradient_accumulation_steps
                * train_args.per_device_train_batch_size
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            state["max_steps"] = math.ceil(
                train_args.num_train_epochs * num_update_steps_per_epoch
            )
            state["num_train_epochs"] = train_args.num_train_epochs
            state["train_batch_size"] = train_args.per_device_train_batch_size

        with open(f"{resume_from_checkpoint}/trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)

        (Path(resume_from_checkpoint) / "scheduler.pt").unlink(missing_ok=True)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # evaluate on test dataset
    if "test" in ds_dict:
        eval_results = trainer.evaluate(ds_dict["test"])  # type: ignore

        # save results
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
    move_best_ckpt(trainer)

    # save config
    with open(save_dir / "config.json", "w") as f:
        cfg["train_args"] = train_args.to_dict()
        json.dump(cfg, f, indent=2)
    wandb.config.update(cfg)

    # save predictions
    if predict_dict is not None:
        for name, predict_ds in predict_dict.items():
            predict_ds = predict_ds.map(preprocess, batched=True)
            print("Gathering predictions for", name)
            pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)
            preds = pred_logits.softmax(-1).tolist()
            pred_ds = predict_ds.add_column("soft_pred", preds)
            pred_ds.save_to_disk(str(save_dir / "predictions" / name))

    # save hiddens
    if store_post_hiddens:
        print("Gathering hiddens")
        hiddens = gather_hiddens(model.transformer, ds_dict["train"])
        torch.save(hiddens, save_dir / "post_hiddens.pt")

    wandb.finish()
