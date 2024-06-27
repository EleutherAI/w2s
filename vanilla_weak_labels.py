from pathlib import Path
from typing import Union

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from fire import Fire
from transformers import TrainingArguments

from w2s.ds_registry import load_and_process_dataset
from w2s.model import MODEL_REGISTRY, ModelConfig, TransformerPredictor
from w2s.sft import lm_sft
from w2s.sft_config import set_default_args
from w2s.utils import ds_with_labels


def add_idx_col(ds: Union[DatasetDict, Dataset]) -> Union[DatasetDict, Dataset]:
    if isinstance(ds, DatasetDict):
        for split in ds:
            ds[split] = add_idx_col(ds[split])
        return ds
    else:
        ds = ds.add_column("idx", range(len(ds)))  # type: ignore
        return ds


def main(
    ds_name: str,
    model_name: str = "Qwen/Qwen1.5-0.5B",
    n_train: int = 8_000,
    n_val: int = 500,
    n_test: int = 5_000,
    n_predict: int = 50_000,
    results_folder=None,
    disable_lora: bool = False,
    **train_args,
):
    train_args["num_train_epochs"] = train_args.get("num_train_epochs", 3)
    train_args = set_default_args(train_args, model_name=model_name)

    model_last = model_name.split("/")[-1]
    if results_folder is None:
        results_folder = str(Path(__file__).parent / f"results/{ds_name}_{model_last}")

    # load dataset
    source_ds = load_and_process_dataset(ds_name, n_train, n_val, n_test, n_predict)

    # train weak floor, save predictions on train and test
    print(f"\n\033[32m===== Training {model_name} =====\033[0m")
    mc = ModelConfig(model_name, not disable_lora, TransformerPredictor)
    model = mc.initialize_model()
    train_args["output_dir"] = results_folder
    train_args["learning_rate"] = MODEL_REGISTRY[model_name]["lr"]
    ds_dict = DatasetDict(
        train=ds_with_labels(source_ds["train"]),
        val=ds_with_labels(source_ds["val"]),
    )
    predict_ds_dict = DatasetDict(
        train=concatenate_datasets(
            [
                ds_dict["train"],
                ds_dict["val"],
                ds_with_labels(source_ds["predict"]),
            ]
        ),
        test=ds_with_labels(source_ds["test"]),
    )
    exp_cfg = mc.to_dict()
    exp_cfg.update(
        {
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_predict": n_predict,
        }
    )
    lm_sft(
        ds_dict=ds_dict,
        model=model.transformer,
        tokenizer=model.tokenizer,
        train_args=TrainingArguments(**train_args),
        loss="xent",
        store_pre_hiddens=False,
        store_post_hiddens=False,
        cfg=exp_cfg,
        predict_dict=predict_ds_dict,
    )

    # read the predictions and replace the txt column
    predict_dir = Path(results_folder) / "predictions"
    train_ds = load_from_disk(str(predict_dir / "train"))
    test_ds = load_from_disk(str(predict_dir / "test"))

    # save to disk
    train_ds.save_to_disk(str(Path(results_folder) / "weak_train"))
    test_ds.save_to_disk(str(Path(results_folder) / "weak_test"))


if __name__ == "__main__":
    Fire(main)
