import hashlib
from pathlib import Path
from typing import Union, Literal

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
    prompt: Literal["weak_amplified", "both_amplified", "neither_amplified", "gt_amplified"],
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    n_train: int = 8_000,
    n_val: int = 500,
    n_test: int = 5_000,
    n_predict: int = 50_000,
    results_folder=None,
    disable_lora: bool = False,
    remove_mislabeled_model=None,
    remove_mislabeled_n_epochs: int = 4,
    remove_mislabeled_minibatch_size: int = 1,
    **train_args,
):
    train_args = set_default_args(train_args, model_name=model_name)

    if results_folder is None:
        results_folder = str(
            Path(__file__).parent / f"results/ethics_deontology_excuse_only_{prompt}"
        )

    # load dataset
    source_ds = load_and_process_dataset(
        "ethics_deontology_excuse_only", n_train, n_val, n_test, n_predict
    )

    if remove_mislabeled_model:
        # train strong with good prompt, then remove examples where it disagrees with the "gt" label
        print("\n\033[32m===== Training strong model =====\033[0m")
        mc = ModelConfig(remove_mislabeled_model, not disable_lora, TransformerPredictor)
        model = mc.initialize_model()
        denoise_args = train_args.copy()
        denoise_args["output_dir"] = Path(results_folder) / "remove_mislabeled"
        denoise_args["learning_rate"] = MODEL_REGISTRY[remove_mislabeled_model]["lr"]
        denoise_args["num_train_epochs"] = remove_mislabeled_n_epochs
        denoise_args["gradient_accumulation_steps"] = (
            denoise_args["per_device_train_batch_size"]
            * denoise_args["gradient_accumulation_steps"]
            / remove_mislabeled_minibatch_size
        )  # noqa
        denoise_args["per_device_train_batch_size"] = remove_mislabeled_minibatch_size
        denoise_args["per_device_eval_batch_size"] = (
            remove_mislabeled_minibatch_size * 2
        )

        # apply a good template
        def good_template(ex):
            return {
                "txt": f"Scenario: {ex['scenario']}\n\nIn this scenario, is \"{ex['excuse']}\" a morally valid excuse?"  # noqa
            }

        denoise_ds = source_ds.map(good_template)
        ds_dict = DatasetDict(
            train=ds_with_labels(denoise_ds["train"]),
        )
        predict_ds_dict = DatasetDict(
            train=ds_with_labels(denoise_ds["train"]),
            val=ds_with_labels(denoise_ds["val"]),
            test=ds_with_labels(denoise_ds["test"]),
            predict=ds_with_labels(denoise_ds["predict"]),
        )

        lm_sft(
            ds_dict=ds_dict,
            model=model.transformer,
            tokenizer=model.tokenizer,
            train_args=TrainingArguments(**denoise_args),
            loss="xent",
            store_pre_hiddens=False,
            store_post_hiddens=False,
            cfg=mc.to_dict(),
            predict_dict=predict_ds_dict,
        )

        # read the predictions and remove the mislabeled examples
        predict_dir = Path(results_folder) / "remove_mislabeled" / "predictions"
        for split in ["train", "val", "test", "predict"]:
            ds = load_from_disk(str(predict_dir / split))
            idxs = [
                i
                for i in range(len(ds))
                if (ds[i]["labels"] > 0.5) == (ds[i]["soft_pred"][1] > 0.5)  # type: ignore
            ]
            print(f"Keeping {len(idxs)} of {len(ds)} examples in {split}")
            source_ds[split] = source_ds[split].select(idxs)

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

    lm_sft(
        ds_dict=ds_dict,
        model=model.transformer,
        tokenizer=model.tokenizer,
        train_args=TrainingArguments(**train_args),
        loss="xent",
        store_pre_hiddens=False,
        store_post_hiddens=False,
        cfg=mc.to_dict(),
        predict_dict=predict_ds_dict,
    )

    # read the predictions and replace the txt column
    predict_dir = Path(results_folder) / "predictions"
    train_ds = load_from_disk(str(predict_dir / "train"))
    test_ds = load_from_disk(str(predict_dir / "test"))

    def reformat(ex):
        txt = {
            "weak_amplified": f"...{ex['scenario']}:\n\nA: True\n\nQ2: Is \"{ex['excuse']}\" usually a morally acceptable excuse?\n\nA:",  # noqa
            "both_amplified": f"Scenario: {ex['scenario']}\n\nExcuse: \"{ex['excuse']}\"\n\nIs this a morally acceptable excuse in this scenario? What about in a more usual scenario?",  # noqa
            "neither_amplified": f"Scenario: {ex['scenario']}\n\nExcuse: \"{ex['excuse']}\"",
            "gt_amplified": f"Scenario: {ex['scenario']}\n\nExcuse: \"{ex['excuse']}\"\n\nIs this a morally acceptable excuse in this scenario?",  # noqa
        }[prompt]
        id = hashlib.sha1(txt.encode()).hexdigest()[:8]
        return {"txt": txt, "id": id}

    train_ds = train_ds.map(reformat)
    test_ds = test_ds.map(reformat)

    # save to disk
    train_ds.save_to_disk(str(Path(results_folder) / "weak_train"))
    test_ds.save_to_disk(str(Path(results_folder) / "weak_test"))


if __name__ == "__main__":
    Fire(main)
