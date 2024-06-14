import json
from pathlib import Path
from typing import Union, Optional
import os

import torch
from datasets import DatasetDict
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import wandb
from w2s.loss import (
    log_confidence_loss, 
    confidence_window_loss, 
    cross_entropy_loss,
    kl_divergence_loss,
)
from w2s.model import ModelConfig, init_tokenizer, init_model
from w2s.roc_auc import roc_auc
from w2s.sft_utils import (
    clear_mem,
    get_gpu_mem_used,
    move_best_ckpt,
    gather_hiddens,
    spotcheck_init,
    spotcheck,
    lshape,
)
from w2s.loss import LossConfig
from w2s.probe import PROBES


class CustomLossTrainer(Trainer):
    def __init__(
        self,
        loss_name: str,
        loss_cfg: LossConfig,
        transfer: bool,
        buffer_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.loss_cfg = loss_cfg
        self.transfer = transfer
        if loss_name in ["logconf", "entropy"]:
            self.buffer = []
            self.buffer_size = buffer_size


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()

        outputs = model(**inputs)

        if self.loss_name == 'logconf':
            loss = log_confidence_loss(
                outputs.logits,
                labels,
                self.state.global_step,
                aux_coef=(self.loss_cfg.logconf_weight if self.transfer else 0.),
                warmup_steps=self.loss_cfg.logconf_warmup_steps,
                balance_batch=self.loss_cfg.balance_batch,
                harden=True,
                buffer=self.buffer,
                buffer_size=self.buffer_size,
            )
        elif self.loss_name == 'entropy':
            loss = log_confidence_loss(
                outputs.logits,
                labels,
                self.state.global_step,
                aux_coef=(self.loss_cfg.logconf_weight if self.transfer else 0.),
                warmup_steps=self.loss_cfg.logconf_warmup_steps,
                balance_batch=self.loss_cfg.balance_batch,
                harden=False,
                buffer=self.buffer,
                buffer_size=self.buffer_size,
            )
        elif self.loss_name == 'xent':
            loss = cross_entropy_loss(
                outputs.logits,
                labels,
            )
        elif self.loss_name == 'kl':
            loss = kl_divergence_loss(
                outputs.logits,
                labels,
            )
        elif self.loss_name == 'window':
            loss = confidence_window_loss(
                outputs.logits,
                labels,
                radius=(self.loss_cfg.radius if self.transfer else 0.51),
            )
        else:
            raise ValueError(f"Unknown loss function: {self.loss_name}")

        return (loss, outputs) if return_outputs else loss


def train(
    ds_dict: DatasetDict,
    model_cfg: ModelConfig,
    train_args: TrainingArguments,
    cfg: dict,
    transfer: bool,
    predict_dict: Union[DatasetDict, dict, None] = None,
    save_activations: bool = False,
    use_probe: bool = False,
    acts_dir: Optional[Path] = None,
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

    os.makedirs(save_dir, exist_ok=True)

    clear_mem()

    print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use before toker init")
    tokenizer = init_tokenizer(model_cfg)
    model = None

    def process(examples):
        out = tokenizer(examples["txt"], truncation=True)
        return out

    ds_dict = ds_dict.map(process, batched=True)

    def compute_metrics_torch(predictions, labels):
        hard_labels = (labels > 0.5).long()
        return dict(
            accuracy=predictions.argmax(dim=1).eq(hard_labels).float().mean(),
            auroc=roc_auc(hard_labels, predictions[:, 1]),
        )

    def detorch_metrics(metrics):
        return {k: v.detach().cpu().item() for k, v in metrics.items()}

    def compute_metrics(eval_pred):
        predictions, labels = map(torch.from_numpy, eval_pred)
        return compute_metrics_torch(predictions, labels)

    probe_required = transfer and (cfg['probe_relabel'] or cfg['probe_filter'])

    if save_activations or probe_required:
        if acts_dir.exists() and all((acts_dir / f"{name}.pt").exists() for name in ds_dict.keys()):
            print("Activations already exist at", acts_dir)
        else:
            print("Saving activations to", acts_dir)
            if model is None:
                print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use before training")
                model = init_model(tokenizer, model_cfg)
                print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use after model init")
            acts_dir.mkdir(parents=True, exist_ok=True)
            for name, ds in ds_dict.items():
                acts = gather_hiddens(model, ds)
                torch.save(acts, acts_dir / f"{name}.pt")

    if probe_required:
        print("Training probe")
        print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use before first act load")
        all_acts = torch.load(acts_dir / f"train.pt", map_location="cuda")
        probe_layer = cfg["probe_layer"] 
        if probe_layer is None:
            probe_layer = all_acts.shape[1] // 2
        print(f"Using probe layer {probe_layer}")
        acts = all_acts[:, probe_layer]
        print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use after first act load")
        probe = PROBES[cfg["probe_name"]](cfg["probe"])
        probe.fit(acts, torch.tensor(ds_dict["train"]["labels"], device="cuda"))
        print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use after probe training")
        for name, ds in ds_dict.items():
            all_acts = torch.load(acts_dir / f"{name}.pt", map_location="cuda")
            acts = all_acts[:, probe_layer]
            preds = probe.predict(acts)
            # preds --> (1-preds, preds)
            preds = torch.stack([1 - preds, preds], dim=-1)
            print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use after probe prediction on {name}")
            # print(f"preds shape: {preds.shape}")
            # print(f"labels shape: {torch.tensor(ds['labels']).shape}")
            agree_metrics = compute_metrics_torch(preds, torch.tensor(ds["labels"], device="cuda"))
            gt_metrics = compute_metrics_torch(preds, torch.tensor(ds["gt_labels"], device="cuda"))
            with open(save_dir / f"{name}_probe_metrics.json", "w", ) as f:
                json.dump({
                    "agree": detorch_metrics(agree_metrics), 
                    "gt": detorch_metrics(gt_metrics),
                }, f, indent=2)
            if name in ["train", "val"]:
                if cfg['probe_filter']:
                    good_indices = probe.filter(acts, torch.tensor(ds["labels"], device="cuda"), cfg['contamination'])
                    sizes = {
                        "before": len(ds),
                        "after": len(good_indices),
                        "removed": len(ds) - len(good_indices),
                        "contamination": int(cfg['contamination'] * len(ds)),
                    }
                    with open(save_dir / f"{name}_filter_sizes.json", "w") as f:
                        json.dump(sizes, f, indent=2)
                    ds = ds.select(good_indices)
                    ds_dict[name] = ds
                if cfg['probe_relabel']:
                    # print(lshape(ds["labels"]))
                    ds = ds.remove_columns("labels").add_column("labels", preds[:, 1].detach().cpu().numpy())
                    ds_dict[name] = ds

    if results_path.exists():
        print(
            f"Results already exist at {results_path}. Skipping training and evaluation."
        )
        return
    
    print(f"No results found at {results_path}. Training model.")

    if model is None:
        print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use before training")
        model = init_model(tokenizer, model_cfg)
        print(f"{get_gpu_mem_used() * 100:.2f}% of all GPU memory in use after model init")

    if transfer and cfg["loss_name"] == "window" and cfg["loss"].radius == "midweak":
        confs = torch.abs(torch.tensor(ds_dict["train"]["labels"]) - 0.5)
        cfg["loss"].radius = confs.median().item()
        print(f"Setting radius to {cfg['loss'].radius:.2f} based on median confidence in train set")

    trainer = CustomLossTrainer(
        loss_name=cfg["loss_name"],
        loss_cfg=cfg["loss"],
        buffer_size=cfg["batch_size"],
        transfer=transfer,
        args=train_args,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        eval_dataset={k: ds_dict[k] for k in ["val", "test"]},
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_dict["train"],
    )

    # spotcheck for untrained params
    # spots = spotcheck_init(model)

    # train
    trainer.train()

    # spotcheck
    # spotcheck(model, spots)

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
        cfg["transfer"] = transfer
        cfg["loss"] = cfg["loss"].to_dict()
        cfg["probe"] = cfg["probe"].to_dict()
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
