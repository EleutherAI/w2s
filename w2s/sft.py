import json
from pathlib import Path
from typing import Union

import torch
from datasets import DatasetDict
from torch import nn, optim
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import wandb
from w2s.loss import log_confidence_loss
from w2s.model import ModelConfig, init_model_and_tokenizer
from w2s.roc_auc import roc_auc
from w2s.sft_utils import (
    clear_mem,
    get_gpu_mem_used,
    move_best_ckpt,
)


class Calibrator(nn.Module):
    def __init__(self, n_components: int):
        super().__init__()
        self.n_components = 3
        self.scale = nn.Parameter(torch.ones(n_components, device="cuda"))
        self.bias = nn.Parameter(
            torch.arange(n_components, dtype=torch.float32, device="cuda")
            - n_components // 2
        )
        self.weight_logits = nn.Parameter(torch.zeros(n_components, device="cuda"))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logodds = logits[:, 1] - logits[:, 0]
        weights = torch.softmax(self.weight_logits, dim=0)
        p = (torch.sigmoid(logodds[:, None] * self.scale + self.bias) * weights).sum(
            dim=-1
        )  # [b,]
        return torch.stack([1 - p, p], dim=1)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100):
        """Fit the mixture of sigmoids to data with LBFGS.

        Args:
            logits: Logits of shape [batch, 2].
            labels: Binary labels of shape [batch].
        """

        opt = optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(logits.dtype).eps,
            tolerance_grad=torch.finfo(logits.dtype).eps,
        )

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(self(logits), labels)
            loss.backward()
            return float(loss)

        opt.step(closure)


class CustomLossTrainer(Trainer):
    def __init__(
        self,
        logconf_weight: float,
        logconf_warmup_steps: int,
        balance_batch: bool,
        calibrate_every: int = -1,  # steps
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logconf_weight = logconf_weight
        self.logconf_warmup_steps = logconf_warmup_steps
        self.balance_batch = balance_batch
        self.calibrator = Calibrator(n_components=3)
        self.calibrate_every = calibrate_every
        self.calibration_buffer = []
        self.n_calib = 32

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()

        outputs = model(**inputs)

        self.maybe_fit_calibrator(outputs.logits.detach(), labels)

        loss = log_confidence_loss(
            outputs.logits,
            labels,
            self.state.global_step,
            aux_coef=self.logconf_weight,
            warmup_steps=self.logconf_warmup_steps,
            balance_batch=self.balance_batch,
        )

        return (loss, outputs) if return_outputs else loss

    def maybe_fit_calibrator(self, logits, labels):
        # don't fite during eval
        if not self.is_in_train or not self.model.training or self.state.max_steps < 1:
            return
        if self.calibrate_every < 0:
            return

        # fill up a buffer of confidently labeled examples
        take_per_minibatch = (
            self.n_calib
            // (self.args.gradient_accumulation_steps * self.calibrate_every)
            + 1
        )
        confidences = (labels - 0.5).abs()
        take = torch.topk(confidences, take_per_minibatch).indices
        self.calibration_buffer.extend(zip(logits[take], labels[take]))

        self.calibration_buffer
        if (
            self.state.global_step % self.calibrate_every == 0
            and self.calibration_buffer
        ):
            print(
                f"(step {self.state.global_step}) calibrating "
                f"using {len(self.calibration_buffer)} examples"
            )
            conf_logits, conf_labels = zip(*self.calibration_buffer)
            conf_logits = torch.stack(conf_logits)
            conf_labels = torch.stack(conf_labels)

            self.calibrator.fit(conf_logits, conf_labels)
            self.calibration_buffer = []


def train(
    ds_dict: DatasetDict,
    model_cfg: ModelConfig,
    train_args: TrainingArguments,
    cfg: dict,
    logconf_weight: float = 0.0,
    logconf_warmup_steps: int = 200,
    balance_batch: bool = False,
    predict_dict: Union[DatasetDict, dict, None] = None,
):
    """
    ds_dict: DatasetDict with splits for train, val, test, and (optionally) predict,
        with columns "txt" and "labels"
    model_cfg: ModelConfig with the model name and whether to enable LoRA
    train_args: TrainingArguments with the training hyperparameters
    cfg: a dictionary containing all the relevant details for reproducibility.
        This will be updated with your train_args and model_cfg before saving.
    logconf_weight: the weight for the log confidence loss
    logconf_warmup_steps: the number of steps to linearly increase the logconf_weight
    balance_batch: whether to balance the batch with the log confidence loss

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
        balance_batch=balance_batch,
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
