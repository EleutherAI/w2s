import random
from pathlib import Path
from typing import Optional, Union

import fire
import numpy as np
import torch
from datasets import Dataset, load_from_disk, DatasetDict
from transformers import TrainingArguments

from w2s.model import ModelConfig, TransformerPredictor
from w2s.reporter import SftStage
from w2s.reporter_experiment import ExperimentConfig, train_and_eval_reporter
from w2s.sft_config import set_default_args
from w2s.sft import lm_sft
from w2s.utils import assert_type, ds_with_labels, split_args_by_prefix

# for a given weak dataset and model (and oracle vs weak label column), 
# we just do regular sft on this data, and measure val KL loss on 200 examples every 10 steps
def train_reporter_on_transformer(
    weak_ds_path: Union[str, Path],
    n_train: int,
    n_test: int,
    use_weak_label: bool = False,
    # model config
    strong_model_name: str = "meta-llama/Meta-Llama-3-8B",
    disable_lora: bool = False,
    quantize: bool = False,
    # ExperimentConfig
    run_name: str = "salience",
    seed: int = 42,
    **train_args,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_args = set_default_args(
        train_args, model_name=strong_model_name, run_name=run_name,
    )

    weak_ds_path = Path(weak_ds_path)
    train_args["output_dir"] = str(weak_ds_path / run_name)
    train_args["run_name"] = f"{weak_ds_path.name}_{run_name}_{'weak' if use_weak_label else 'oracle'}"

    # load datasets
    weak_train = assert_type(Dataset, load_from_disk(str(weak_ds_path / "weak_train"))).shuffle(seed).select(range(n_train))
    weak_test = assert_type(Dataset, load_from_disk(str(weak_ds_path / "weak_test"))).shuffle(seed).select(range(n_test))

    label_col = "soft_pred" if use_weak_label else "soft_label"
    ds_dict = DatasetDict(
        train=ds_with_labels(weak_train, label_col),
        test=ds_with_labels(weak_test, label_col),
    )

    mcfg = ModelConfig(
        strong_model_name,
        not disable_lora,
        TransformerPredictor,
        quantize=quantize,
    )
    model = mcfg.initialize_model()

    lm_sft(
        ds_dict=ds_dict,
        model=model.transformer,
        tokenizer=model.tokenizer,
        train_args=TrainingArguments(**train_args),
        loss="kl",
        store_pre_hiddens=False,
        store_post_hiddens=False,
        cfg=mcfg.to_dict(),
    )

if __name__ == "__main__":
    fire.Fire(train_reporter_on_transformer)
