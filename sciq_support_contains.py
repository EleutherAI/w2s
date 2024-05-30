from pathlib import Path
from typing import Union

from datasets import Dataset, DatasetDict
from fire import Fire

from w2s.ds_registry import load_and_process_dataset


def add_idx_col(ds: Union[DatasetDict, Dataset]) -> Union[DatasetDict, Dataset]:
    if isinstance(ds, DatasetDict):
        for split in ds:
            ds[split] = add_idx_col(ds[split])
        return ds
    else:
        ds = ds.add_column("idx", range(len(ds)))  # type: ignore
        return ds


def main(n_train: int = 50_000, n_test: int = 5_000, results_folder=None):
    if results_folder is None:
        results_folder = str(Path(__file__).parent / "results/sciq_support_contains")

    # load dataset
    source_ds = load_and_process_dataset("sciq_support_contains", n_train, 0, n_test, 0)

    def does_support_contain(ex):
        label = int(ex["anwer"].lower() in ex["support"].lower())
        return {"soft_pred": [1 - label, label]}

    train_ds = source_ds["train"].map(does_support_contain)
    test_ds = source_ds["test"].map(does_support_contain)

    # save to disk
    train_ds.save_to_disk(str(Path(results_folder) / "weak_train"))
    test_ds.save_to_disk(str(Path(results_folder) / "weak_test"))


if __name__ == "__main__":
    Fire(main)
