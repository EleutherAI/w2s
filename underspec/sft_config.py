from dataclasses import dataclass
from typing import Literal, Optional, Union

from simple_parsing import Serializable


@dataclass
class SFTConfig(Serializable):  # TODO: what is this for??
    # name of the model to train
    model_name: str
    # name of the dataset to use
    dataset: str
    n_epochs: int = 2
    n_train: int = 20_000
    n_val: int = 500
    n_test: int = 1_000
    # this is typically used to generate weak labels
    generate_predictions: bool = False
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
    loss: str = "xent"
    # if true, it stores hiddens at all layers in "hiddens.pt" (also ids in "ids.pt")
    store_pre_hiddens: bool = False
    store_post_hiddens: bool = False
    weight_decay: float = 0.1
    lr: float = 3e-5

    def to_dict(self):
        irrelevant_fields = ["results_folder", "run_name", "minibatch_size"]
        return {k: v for k, v in vars(self).items() if k not in irrelevant_fields}
