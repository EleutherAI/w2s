import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import wandb
from w2s.loss import log_confidence_loss
from w2s.model import TransformerPredictor
from w2s.sft_utils import (
    assert_type,
    clear_mem,
    compute_acc_and_auroc,
    gather_hiddens,
    get_gpu_mem_used,
    move_best_ckpt,
)


class CustomLossTrainer(Trainer):
    def __init__(
        self,
        loss_name: str,
        *args,
        resume_from_optimizer_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.resume_from_optimizer_checkpoint = resume_from_optimizer_checkpoint

    def compute_loss(self, model, inputs, return_outputs=False):
        if (
            self.state.global_step == 0
            and self.resume_from_optimizer_checkpoint is not None
            and self.optimizer is not None
        ):
            # check if adam exp buffer is empty, and then load the optimizer state if it is
            if not isinstance(self.optimizer, torch.optim.AdamW):
                assert isinstance(self.optimizer.optimizer, torch.optim.AdamW)
            self.optimizer: torch.optim.AdamW
            state = self.optimizer.state[self.optimizer.param_groups[0]["params"][0]]
            if "exp_avg" not in state:
                # update the step, exp_avg, and exp_avg_sq of the optimizer state
                print(
                    "Loading optimizer state from",
                    self.resume_from_optimizer_checkpoint,
                )
                state_dict = torch.load(
                    self.resume_from_optimizer_checkpoint,
                    map_location=self.model.device,
                )["state"]
                trainable_params = (
                    p for p in self.model.parameters() if p.requires_grad
                )
                for state, p in zip(state_dict.values(), trainable_params):  # type: ignore
                    self.optimizer.state[p] = state  # type: ignore
                self.resume_from_optimizer_checkpoint = None

        labels = inputs.pop("labels").float()

        outputs = model(**inputs)
        if self.loss_name == "xent":
            aux_weight = 0
        elif self.loss_name == "logconf":
            aux_weight = 0.5
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

        loss = log_confidence_loss(
            outputs.logits, labels, self.state.global_step, aux_coef=aux_weight
        )

        return (loss, outputs) if return_outputs else loss


def prepare_for_trainer(ds: Union[DatasetDict, Dataset], tokenizer):
    keep_cols = {"labels", "input_ids", "attention_mask"}

    def preprocess(exs):
        out = tokenizer(exs["txt"], truncation=True)
        return {k: v for k, v in out.items() if k in keep_cols}

    ds.reset_format()
    columns_names = (
        ds.column_names
        if isinstance(ds, Dataset)
        else next(iter(ds.values())).column_names
    )
    ds = ds.map(
        preprocess,
        batched=True,
        remove_columns=list(set(columns_names) - keep_cols),
    )
    return ds


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

    ds_dict = assert_type(DatasetDict, prepare_for_trainer(ds_dict, model.tokenizer))

    trainer = CustomLossTrainer(
        loss_name=loss,
        resume_from_optimizer_checkpoint=resume_from_checkpoint,
        args=train_args,
        compute_metrics=compute_acc_and_auroc,
        data_collator=DataCollatorWithPadding(
            model.tokenizer, max_length=1024, padding="max_length"
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
    trainer.train()

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
            predict_ds = prepare_for_trainer(predict_ds, model.tokenizer)
            print("Gathering predictions for", name)
            pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)  # type: ignore
            preds = pred_logits.softmax(-1).tolist()
            pred_ds = predict_ds.add_column("soft_pred", preds)  # type: ignore
            pred_ds.save_to_disk(str(save_dir / "predictions" / name))

    # save hiddens
    if store_post_hiddens:
        print("Gathering hiddens")
        hiddens = gather_hiddens(model.transformer, ds_dict["train"])
        torch.save(hiddens, save_dir / "post_hiddens.pt")

    wandb.finish()
