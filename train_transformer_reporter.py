from typing import List, Optional

import fire
from datasets import Dataset, load_from_disk

from underspec.model import ModelConfig, TransformerPredictor
from underspec.train_reporter import ExperimentConfig, train_reporter
from underspec.utils import assert_type


def train_reporter_on_transformer(
    weak_ds_path: str,
    test_ds_path: str,
    # model config
    model_name: str,
    disable_lora: bool = False,
    lora_modules: Optional[List[str]] = None,
    # ExperimentConfig
    reporter_method: str = "w2s",
    max_num_queries: int = 256,
    results_folder: str = "./results",
    run_name: str = "default",
    input_col: str = "txt",
    **reporter_args,
):
    # weak_ds has "id", input_col, and "soft_pred"
    weak_ds = assert_type(Dataset, load_from_disk(weak_ds_path))
    # remove gt columns from weak_ds to ensure they aren't used
    weak_ds = weak_ds.remove_columns(["soft_label", "hard_label"])

    # test_ds has "id", input_col, and "soft_label"
    test_ds = assert_type(Dataset, load_from_disk(test_ds_path))

    dataset_cfg_dict = {"weak_ds_path": weak_ds_path, "test_ds_path": test_ds_path}

    # load strong model
    model_cfg = ModelConfig(
        name=model_name, enable_lora=not disable_lora, lora_modules=lora_modules
    )
    strong_model = TransformerPredictor(model_cfg)

    exp_cfg = ExperimentConfig(
        reporter_method=reporter_method,
        max_num_queries=max_num_queries,
        results_folder=results_folder,
        run_name=run_name,
        input_col=input_col,
    )
    train_reporter(
        weak_ds,
        test_ds,
        strong_model,
        exp_cfg,
        dataset_cfg_dict=dataset_cfg_dict,
        **reporter_args,
    )


if __name__ == "__main__":
    fire.Fire(train_reporter_on_transformer)
