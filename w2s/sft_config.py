from dataclasses import dataclass
from typing import Literal, Optional, Union

from simple_parsing import Serializable, field, subgroups


@dataclass
class LossConfig(Serializable):
    def to_dict(self):
        irrelevant_fields = []
        return {k: v for k, v in vars(self).items() if k not in irrelevant_fields}

@dataclass
class LogConfidenceLossConfig(LossConfig):
    logconf_weight: float = 0.5
    logconf_warmup_steps: int = 200
    balance_batch: bool = False

@dataclass
class ConfidenceWindowLossConfig(LossConfig):
    radius: float = 0.15

LOSS_CONFIGS = {
    "logconf": LogConfidenceLossConfig, 
    "window": ConfidenceWindowLossConfig,
}

@dataclass
class SFTConfig(Serializable):  # TODO: what is this for??
    # name of the model to train
    weak_model_name: str
    strong_model_name: str
    # name of the dataset to use
    dataset: str
    n_epochs: float = 2
    n_train: int = 20_000
    n_val: int = 500
    n_test: int = 1_000
    # when "train", it uses the training set to generate predictions
    # otherwise it uses n_predict held out examples
    n_predict: Union[Literal["train"], int] = 0
    minibatch_size: int = 8
    # examples per update
    batch_size: int = 32
    results_folder: str = "./results"
    run_name: str = "default"
    disable_lora: bool = False
    lr_schedule: str = "cosine"
    n_warmup_steps: int = 40  # 2 / (1 - 0.95) = 40
    eval_every: int = 100  # steps
    save_every: int = 100  # steps
    save_total_limit: Optional[int] = None
    loss: LossConfig = subgroups(LOSS_CONFIGS, default="logconf")
    weight_decay: float = 0.1
    weak_lr: float = 5e-4
    strong_lr: float = 8e-5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "val_auroc"

    greater_is_better: bool = field(init=False)
    loss_name: str = field(init=False)

    def __post_init__(self):
        if "loss" in self.metric_for_best_model:
            self.greater_is_better = False
        elif (
            "auroc" in self.metric_for_best_model
            or "accuracy" in self.metric_for_best_model
        ):
            self.greater_is_better = True
        else:
            raise ValueError(f"Unknown metric {self.metric_for_best_model}")

        self.loss_name = {LOSS_CONFIGS[k]:k for k in LOSS_CONFIGS}[type(self.loss)]
        
    def to_dict(self):
        irrelevant_fields = ["results_folder", "run_name", "minibatch_size"]
        return {k: v for k, v in vars(self).items() if k not in irrelevant_fields}
