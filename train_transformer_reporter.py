from pathlib import Path
from typing import Optional

import fire
import torch
from datasets import Dataset, load_from_disk

from w2s.model import ModelConfig, TransformerPredictor
from w2s.reporter_experiment import ExperimentConfig, train_and_eval_reporter
from w2s.sft_config import set_default_args
from w2s.utils import assert_type


def train_reporter_on_transformer(
    weak_ds_path: str,
    oracle_ds_path: str,
    test_ds_path: str,
    n_train: int,
    max_num_oracle: int,
    n_test: int,
    # model config
    strong_model_name: str = "meta-llama/Meta-Llama-3-8B",
    disable_lora: bool = False,
    # ExperimentConfig
    reporter_method: str = "SftReporter",
    results_folder: Optional[str] = None,
    run_name: str = "default",
    input_col: str = "txt",
    **reporter_args,
):
    reporter_args = set_default_args(
        reporter_args, model_name=strong_model_name, run_name=run_name
    )

    # default to parent / "results"
    if results_folder is None:
        results_folder = str(Path(__file__).parent / "results")
    reporter_args["output_dir"] = str(Path(results_folder) / run_name)

    # load datasets
    weak_ds = assert_type(Dataset, load_from_disk(weak_ds_path))
    weak_ds = weak_ds.remove_columns(["soft_label", "hard_label"])
    oracle_ds = assert_type(Dataset, load_from_disk(oracle_ds_path)).select(
        range(max_num_oracle)
    )
    test_ds = assert_type(Dataset, load_from_disk(test_ds_path)).select(range(n_test))

    if reporter_method == "ActiveSftReporter":
        print("Selecting examples with highest entropy for training.")
        # select the weak examples with *highest* entropy (easy examples)
        probs = torch.as_tensor(weak_ds["soft_pred"])
        entropies = -(probs * torch.log(probs)).sum(dim=-1)
        weak_ds = weak_ds.select((-entropies).argsort()[:n_train])
    else:
        weak_ds = weak_ds.select(range(n_train))

    dataset_cfg_dict = {
        "weak_ds_path": str(weak_ds_path),
        "oracle_ds_path": str(oracle_ds_path),
        "test_ds_path": str(test_ds_path),
        "n_train": n_train,
        "max_num_oracle": max_num_oracle,
        "n_test": n_test,
    }

    # load a new, randomly initialized model for the strong model
    strong_model = TransformerPredictor(
        ModelConfig(strong_model_name, not disable_lora)
    )

    exp_cfg = ExperimentConfig(
        reporter_method=reporter_method,
        max_num_oracle=max_num_oracle,
        results_folder=results_folder,
        run_name=run_name,
        input_col=input_col,
    )
    train_and_eval_reporter(
        weak_ds,
        oracle_ds,
        test_ds,
        strong_model,
        exp_cfg,
        dataset_cfg_dict=dataset_cfg_dict,
        **reporter_args,
    )


if __name__ == "__main__":
    fire.Fire(train_reporter_on_transformer)
