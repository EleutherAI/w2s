from typing import Optional

from datasets import DatasetDict
from transformers import TrainingArguments

from w2s.ds_registry import load_and_process_dataset
from w2s.model import MODEL_REGISTRY, ModelConfig, TransformerPredictor
from w2s.sft import lm_sft
from w2s.utils import ds_with_labels


def main(
    ds_name,
    model_name,
    n_train,
    n_val,
    n_test,
    save_predictions: bool = False,
    results_folder: Optional[str] = None,
    disable_lora: bool = False,
    **train_args,
):
    # load dataset
    source_ds = load_and_process_dataset(ds_name, n_train, n_val, n_test, 0)

    # train weak floor, save predictions on train and test
    print("\n\033[32m===== Training {model_name} =====\033[0m")
    mc = ModelConfig(model_name, not disable_lora)
    model = TransformerPredictor(mc)
    train_args["output_dir"] = results_folder
    train_args["learning_rate"] = MODEL_REGISTRY[model_name]["lr"]
    ds_dict = DatasetDict(
        train=ds_with_labels(source_ds["train"]),
        test=ds_with_labels(source_ds["test"]),
    )

    lm_sft(
        ds_dict=ds_dict,
        model=model,
        train_args=TrainingArguments(**train_args),
        loss="xent",
        store_pre_hiddens=False,
        store_post_hiddens=False,
        cfg=mc.to_dict(),
        predict_dict=ds_dict if save_predictions else None,
    )
