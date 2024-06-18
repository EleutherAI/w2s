import random
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch
from datasets import Dataset, load_from_disk

from w2s.model import ModelConfig, TransformerPredictor
from w2s.reporter import SftStageConfig
from w2s.reporter_experiment import ExperimentConfig, train_and_eval_reporter
from w2s.sft_config import set_default_args
from w2s.utils import assert_type, split_args_by_prefix


def train_reporter_on_transformer(
    weak_ds_path: str,
    oracle_ds_path: str,
    test_ds_path: str,
    weak_pool_size: int,
    oracle_pool_size: int,
    n_test: int,
    # model config
    strong_model_name: str = "meta-llama/Meta-Llama-3-8B",
    disable_lora: bool = False,
    # ExperimentConfig
    reporter_stages=1,
    results_folder: Optional[str] = None,
    run_name: str = "default",
    input_col: str = "txt",
    seed: int = 42,
    num_heads: int = 1,
    **reporter_args,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    reporter_args = set_default_args(
        reporter_args, model_name=strong_model_name, run_name=run_name
    )

    # default to parent / "results"
    if results_folder is None:
        results_folder = str(Path(__file__).parent / "results")
    reporter_args["output_dir"] = str(Path(results_folder) / run_name)

    stage_args = split_args_by_prefix(
        reporter_args, [f"stage{i}_" for i in range(reporter_stages)]
    )
    for stage in stage_args:
        stage_args[stage]["output_dir"] = str(
            Path(reporter_args["output_dir"]) / stage[:-1]
        )
        stage_args[stage]["run_name"] = f"{run_name}-{stage[:-1]}"
    stages = [
        SftStageConfig(**stage_args[f"stage{i}_"]) for i in range(reporter_stages)
    ]

    # load datasets
    weak_ds = assert_type(Dataset, load_from_disk(weak_ds_path))
    weak_ds = weak_ds.remove_columns(["soft_label", "hard_label"])
    oracle_ds = (
        assert_type(Dataset, load_from_disk(oracle_ds_path))
        .shuffle()
        .select(range(oracle_pool_size))
    )
    test_ds = assert_type(Dataset, load_from_disk(test_ds_path)).select(range(n_test))

    dataset_cfg_dict = {
        "weak_ds_path": str(weak_ds_path),
        "oracle_ds_path": str(oracle_ds_path),
        "test_ds_path": str(test_ds_path),
        "n_test": n_test,
        "weak_pool_size": weak_pool_size,
        "oracle_pool_size": len(oracle_ds),
    }
    weak_ds = weak_ds.shuffle().select(range(weak_pool_size))

    assert num_heads == 1
    mcfg = ModelConfig(
        strong_model_name,
        not disable_lora,
        TransformerPredictor,
        num_heads=num_heads,
    )
    exp_cfg = ExperimentConfig(
        stages=stages,
        results_folder=results_folder,
        run_name=run_name,
        input_col=input_col,
    )
    train_and_eval_reporter(
        weak_ds,
        oracle_ds,
        test_ds,
        mcfg,
        exp_cfg,
        dataset_cfg_dict=dataset_cfg_dict,
    )


if __name__ == "__main__":
    fire.Fire(train_reporter_on_transformer)
