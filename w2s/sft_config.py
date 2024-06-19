from dataclasses import dataclass
from typing import Optional, Union

from simple_parsing import Serializable, field, subgroups
from w2s.loss import LOSS_CONFIGS, LossConfig
from w2s.probe import PROBE_CONFIGS, ProbeConfig
from w2s.sft_utils import literal



@dataclass
class SFTConfig(Serializable):
    # name of the model to train
    weak_model_name: str = "Qwen/Qwen1.5-0.5B"
    strong_model_name: str = "meta-llama/Meta-Llama-3-8B"
    # name of the dataset to use
    dataset: str = "boolq"
    n_epochs: float = 3
    n_train: int = 10_000
    n_val: int = 1_000
    n_test: int = 5_000
    # when "train", it uses the training set to generate predictions
    # otherwise it uses n_predict held out examples
    n_predict: Union[literal("train"), int] = 0
    # examples per minibatch (small to fit in memory for long sequences)
    minibatch_size: int = 1
    # examples per update (gradient accumulated across minibatches)
    batch_size: int = 32
    results_folder: str = "./results"
    run_name: str = "default"
    shared_folder: str = "shared"
    disable_lora: bool = False
    lr_schedule: str = "cosine"
    n_warmup_steps: int = 40  # 2 / (1 - 0.95) = 40
    eval_every: int = 25  # steps
    save_every: int = 25  # steps
    save_total_limit: Optional[int] = 1
    weight_decay: float = 0.1
    weak_lr: float = 5e-4
    strong_lr: float = 8e-5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "val_auroc"

    loss: LossConfig = subgroups(LOSS_CONFIGS, default="logconf")
    probe: ProbeConfig = subgroups(PROBE_CONFIGS, default="knn")

    probe_layer: Optional[int] = None
    probe_relabel: bool = False
    probe_filter: bool = False
    contamination: float = 0.1

    greater_is_better: bool = field(init=False)
    loss_name: str = field(init=False)
    probe_name: str = field(init=False)

    s2s_iters: int = 0
    save_strong_acts: bool = False


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
        self.probe_name = {PROBE_CONFIGS[k]:k for k in PROBE_CONFIGS}[type(self.probe)]
        
    def to_dict(self):
        irrelevant_fields = ["results_folder", "run_name", "minibatch_size"]
        return {k: v for k, v in vars(self).items() if k not in irrelevant_fields}
