from pathlib import Path
from typing import Optional

import fire
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import TrainingArguments

from w2s.ds_registry import load_and_process_dataset
from w2s.model import MODEL_REGISTRY, ModelConfig, TransformerPredictor
from w2s.reporter_experiment import ExperimentConfig, train_and_eval_reporter
from w2s.sft import lm_sft
from w2s.utils import assert_type, split_args_by_prefix


def train_reporter_on_transformer(
    ds_name: str,
    n_train: int,
    n_test: int,
    # model config
    strong_model_name: str = "mistralai/Mistral-7B-v0.1",
    weak_model_name: str = "Qwen/Qwen1.5-0.5B",
    disable_lora: bool = False,
    # ExperimentConfig
    reporter_method: str = "SftReporter",
    max_num_oracle: int = 256,
    results_folder: Optional[str] = None,
    run_name: str = "default",
    input_col: str = "txt",
    **reporter_args,
):
    # set defaults
    reporter_args["num_train_epochs"] = reporter_args.get("num_train_epochs", 3)
    reporter_args["per_device_train_batch_size"] = reporter_args.get(
        "per_device_train_batch_size", 32
    )
    reporter_args["per_device_eval_batch_size"] = reporter_args.get(
        "per_device_eval_batch_size", 32
    )
    reporter_args["gradient_accumulation_steps"] = reporter_args.get(
        "gradient_accumulation_steps", 1
    )
    reporter_args["warmup_steps"] = reporter_args.get("warmup_steps", 40)
    reporter_args["lr_scheduler_type"] = reporter_args.get(
        "lr_scheduler_type", "cosine"
    )
    reporter_args["weight_decay"] = reporter_args.get("weight_decay", 0.01)
    reporter_args["eval_strategy"] = reporter_args.get("eval_strategy", "steps")
    reporter_args["eval_steps"] = reporter_args.get("eval_steps", 50)
    reporter_args["save_strategy"] = reporter_args.get("save_strategy", "steps")
    reporter_args["save_steps"] = reporter_args.get("save_steps", 50)
    reporter_args["logging_steps"] = reporter_args.get("logging_steps", 25)
    reporter_args["load_best_model_at_end"] = reporter_args.get(
        "load_best_model_at_end", True
    )
    reporter_args["metric_for_best_model"] = reporter_args.get(
        "metric_for_best_model", "val_loss"
    )
    reporter_args["greater_is_better"] = reporter_args.get("greater_is_better", False)
    reporter_args["save_total_limit"] = reporter_args.get("save_total_limit", 1)
    reporter_args["adam_beta2"] = reporter_args.get("adam_beta2", 0.95)
    reporter_args["tf32"] = reporter_args.get("tf32", True)
    reporter_args["label_names"] = reporter_args.get("label_names", ["labels"])
    reporter_args["run_name"] = run_name

    # default to parent / "results"
    if results_folder is None:
        results_folder = str(Path(__file__).parent / "results")
    reporter_args["output_dir"] = str(Path(results_folder) / run_name)
    save_path = Path(results_folder) / run_name

    # use the "weak" training arguments to train floor and ceiling
    train_args = split_args_by_prefix(reporter_args, ("w2s_", "oracle_"))["w2s_"]

    # load dataset
    source_ds = load_and_process_dataset(ds_name, n_train, 0, n_test, 0)

    # train weak floor, save predictions on train and test
    print("\n\033[32m===== Training weak model =====\033[0m")
    wmc = ModelConfig(weak_model_name, not disable_lora)
    weak_model = TransformerPredictor(wmc)
    weak_args = train_args.copy()
    weak_args["run_name"] = f"weak_{weak_args.get('run_name', 'default')}"
    weak_args["output_dir"] = str(save_path / "weak")
    weak_args["learning_rate"] = MODEL_REGISTRY[weak_model_name]["lr"]
    ds_dict = DatasetDict(
        train=source_ds["train"].add_column("labels", torch.as_tensor(source_ds["train"]["soft_label"])[:, 1].tolist()),  # type: ignore  # noqa
        test=source_ds["test"].add_column("labels", torch.as_tensor(source_ds["test"]["soft_label"])[:, 1].tolist()),  # type: ignore  # noqa
    )
    lm_sft(
        ds_dict=ds_dict,
        model=weak_model,
        train_args=TrainingArguments(**weak_args),
        loss="xent",
        store_pre_hiddens=False,
        store_post_hiddens=False,
        cfg=wmc.to_dict(),
        predict_dict=ds_dict,
    )

    # train strong ceiling
    print("\n\033[32m===== Training strong model =====\033[0m")
    smc = ModelConfig(strong_model_name, not disable_lora)
    strong_model = TransformerPredictor(smc)
    strong_args = train_args.copy()
    strong_args["run_name"] = f"strong_{strong_args.get('run_name', 'default')}"
    strong_args["output_dir"] = str(save_path / "strong")
    strong_args["learning_rate"] = MODEL_REGISTRY[strong_model_name]["lr"]
    lm_sft(
        ds_dict=ds_dict,
        model=strong_model,
        train_args=TrainingArguments(**strong_args),
        loss="xent",
        store_pre_hiddens=False,
        store_post_hiddens=False,
        cfg=smc.to_dict(),
        predict_dict=None,
    )

    # use weak predictions (train) as both weak_ds and oracle_ds
    # use weak predictions (test) as test_ds
    weak_ds_dir = save_path / "weak" / "predictions"
    # weak_ds has "id", input_col, and "soft_pred"
    oracle_ds = assert_type(Dataset, load_from_disk(str(weak_ds_dir / "train")))
    # remove gt columns so they can't be used
    weak_ds = oracle_ds.remove_columns(["soft_label", "hard_label"])
    test_ds = assert_type(Dataset, load_from_disk(str(weak_ds_dir / "test")))

    dataset_cfg_dict = {
        "weak_ds_path": str(weak_ds_dir),
        "ds_name": ds_name,
        "n_train": n_train,
        "n_val": 0,  # not used
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
