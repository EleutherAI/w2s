from pathlib import Path
from typing import Union, Literal

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


def main(
        prompt: Literal["weak_amplified", "both_amplified", "neither_amplified", "gt_amplified"],
        n_train: int = 50_000, n_test: int = 5_000, results_folder=None):
    if results_folder is None:
        results_folder = str(Path(__file__).parent / f"results/sciq_support_contains_{prompt}")

    # load dataset
    source_ds = load_and_process_dataset("sciq_support_contains", n_train, 0, n_test, 0)

    def does_support_contain_and_reformat(ex):
        last_quote = ex["txt"].rfind('"')
        second_to_last_quote = ex["txt"].rfind('"', 0, last_quote - 1)
        ans = ex["txt"][second_to_last_quote + 1 : last_quote]
        label = int(ans.lower() in ex["support"].lower())
        txt = {
            "weak_amplified": ex["txt"], # "{question}\n\n>>>{support}\n\nDoes the quoted text contain "{ans}"?"
            "both_amplified": f">>>{ex['support']}\n\n{ex['question']}\n\nIs {ans} the correct answer? Does the quoted text contain \"{ans}\"?",
            "neither_amplified": f"{ex['question']}\n\n>>>{ex['support']}\n\n{ans}",
            "gt_amplified": f">>>{ex['support']}\n\n{ex['question']}\n\nIs {ans} the correct answer?",
        }[prompt]
        return {"soft_pred": [1 - label, label], "txt": txt}

    train_ds = source_ds["train"].map(does_support_contain_and_reformat)
    test_ds = source_ds["test"].map(does_support_contain_and_reformat)

    # save to disk
    train_ds.save_to_disk(str(Path(results_folder) / "weak_train"))
    test_ds.save_to_disk(str(Path(results_folder) / "weak_test"))


if __name__ == "__main__":
    Fire(main)
