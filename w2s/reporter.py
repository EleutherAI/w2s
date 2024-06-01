import random
from abc import ABC
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from w2s.model import Predictor, TransformerPredictor
from w2s.sft import lm_sft, prepare_for_trainer
from w2s.utils import assert_type, ds_with_labels, split_args_by_prefix


class Oracle:
    _df: pd.DataFrame
    ids_labeled: set

    def __init__(self, gt_dataset: Dataset, input_col: str = "txt") -> None:
        assert (
            "id" in gt_dataset.column_names and "soft_label" in gt_dataset.column_names
        )
        assert "gt_soft_label" not in gt_dataset.column_names

        self._df = gt_dataset.to_pandas()  # type: ignore
        self._df.set_index("id", inplace=True)
        self.input_col = input_col

        self.ids_labeled = set()

    def query_id(self, id: str) -> float:
        self.ids_labeled.add(id)
        return assert_type(float, self._df.loc[id]["soft_label"])

    def query_ids(self, ids: list) -> pd.DataFrame:
        self.ids_labeled.update(ids)
        return self._df.loc[ids]

    def get_inputs(self) -> pd.DataFrame:
        # remove soft_label from inputs
        return self._df.drop(columns=["soft_label"], inplace=False)

    def reset(self):
        self.ids_labeled = set()


class Reporter(ABC):
    """
    This reporter is underspecified in the sense that it has many different possible
    generalization behaviors, represented by the discrete list of classifiers in
    self.classifiers_. The disambiguate step then uses extra data to choose between them.
    One could imagine disambiguating between a continuum of candidate classifiers that have
    similar behavior, but this is not supported here, and would need to be approximated by
    having a large number of candidate classifiers.

    Takes a weakly-labeled dataset and a strong pretrained model,
    and a desired number of candidate classifiers, and then fits candidate classifiers
    and disambiguates between them.
    """

    weak_ds: Dataset
    oracle: Oracle
    test_ds: Dataset
    strong_model: Predictor

    def __init__(
        self,
        weak_ds: Dataset,
        oracle: Oracle,
        test_ds: Dataset,
        strong_model: Predictor,
        input_col: str = "txt",
        save_dir: str = "./results",
    ):
        """
        weak_ds: a dataset with columns ["id", input_col, "soft_pred"]
        oracle: an Oracle object
        strong_model: a
        input_col: the column in weak_ds that contains the input

        """
        assert input_col in weak_ds.column_names
        assert "soft_pred" in weak_ds.column_names
        assert "id" in weak_ds.column_names
        self.weak_ds = weak_ds

        self.oracle = oracle
        self.test_ds = test_ds
        self.strong_model = strong_model
        self.input_col = input_col
        self.save_dir = save_dir

    def fit(self, max_queries: int) -> "Reporter":
        """
        max_queries: the maximum number of queries to the oracle
        """
        ...

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions
        """
        ...

    def to_dict(self) -> dict[str, str | int | float]:
        """A summary of the method that approximately uniquely identifies it.
        It should include a name and all the important hyperparameters."""
        ...


class SftReporter(Reporter):
    """
    A weak-to-strong reporter that uses the same weakly-labeled
    dataset to train all candidate classifiers.
    """

    strong_model: TransformerPredictor  # override type

    def __init__(
        self,
        weak_ds: Dataset,
        oracle: Oracle,
        test_ds: Dataset,
        strong_model: TransformerPredictor,
        input_col: str = "txt",
        save_dir: str = "./results",
        **kwargs,
    ):
        super().__init__(weak_ds, oracle, test_ds, strong_model, input_col)
        self.test_ds = ds_with_labels(test_ds)
        split_args = split_args_by_prefix(kwargs, ("w2s_", "oracle_"))
        for split in split_args:
            split_args[split][
                "run_name"
            ] = f"{split}{split_args[split].get('run_name', 'default')}"
            split_args[split]["output_dir"] = str(Path(save_dir) / split[:-1])

        self.weak_train_args = split_args["w2s_"]
        self.oracle_train_args = split_args["oracle_"]

        assert input_col == "txt", "Only LM SFT is supported"

    def fit(self, max_queries: int) -> "SftReporter":
        weak_ds_dict = DatasetDict(
            train=ds_with_labels(self.weak_ds, labels_column="soft_pred"),
            test=self.test_ds,
        )
        lm_sft(
            ds_dict=weak_ds_dict,
            model=self.strong_model,
            train_args=TrainingArguments(**self.weak_train_args),
            loss="xent",
            store_pre_hiddens=False,
            store_post_hiddens=False,
            cfg=self.to_dict(),
            predict_dict=None,
        )

        if max_queries > 0:
            # then train on oracle (by default just random subset of oracle)
            random_oracle_ids = random.sample(
                self.oracle.get_inputs().index.values.tolist(), max_queries
            )
            oracle_ds = Dataset.from_pandas(self.oracle.query_ids(random_oracle_ids))

            print(
                f"\n\033[32m===== {len(oracle_ds)} oracle queries finetuning =====\033[0m"
            )
            oracle_ds = ds_with_labels(oracle_ds)
            # add num_queries to the run name and output dir
            # the previous results will be cached, but these will not
            ota = self.oracle_train_args.copy()
            ota["run_name"] += f"_{max_queries}queries"
            ota["output_dir"] = f"{ota['output_dir']}_{max_queries}queries"
            max_use_test = 200
            ota["eval_strategy"] = "no" if max_queries < max_use_test else "steps"
            ota["num_train_epochs"] = {
                1: 32,
                2: 32,
                4: 32,
                8: 32,
                16: 32,
                32: 32,
                64: 32,
                128: 16,
                256: 8,
                512: 4,
                1024: 2,
                2048: 1,
                4096: 1,
            }[max_queries]
            if max_queries < max_use_test:
                oracle_dict = DatasetDict(train=oracle_ds)
            else:
                oracle_dict = DatasetDict(
                    train=oracle_ds, test=self.test_ds  # type: ignore
                )
            lm_sft(
                ds_dict=oracle_dict,
                model=self.strong_model,
                train_args=TrainingArguments(**ota),
                loss="xent",
                store_pre_hiddens=False,
                store_post_hiddens=False,
                cfg=self.to_dict(),
                predict_dict=None,
                resume_from_checkpoint=f"{self.weak_train_args['output_dir']}/best-ckpt/optimizer.pt",
            )

        return self

    def to_dict(self) -> dict:
        return {
            "method": "SftReporter",
            "weak_train_args": self.weak_train_args,
            "oracle_train_args": self.oracle_train_args,
            "model": self.strong_model.to_dict(),
        }

    def __call__(self, inputs: list) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions
        """
        trainer = Trainer(
            args=TrainingArguments(**self.weak_train_args),
            data_collator=DataCollatorWithPadding(
                self.strong_model.tokenizer, max_length=1024, padding="max_length"
            ),  # NOTE: this could mess up some datasets
            model=self.strong_model.transformer,
            tokenizer=self.strong_model.tokenizer,
        )
        predict_ds = prepare_for_trainer(
            Dataset.from_dict({self.input_col: inputs}), self.strong_model.tokenizer
        )
        pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)  # type: ignore
        return pred_logits.diff(dim=-1).squeeze()
        # self.strong_model.transformer.eval()
        # idxs = torch.arange(len(inputs))
        # batches = torch.split(idxs, batch_size)
        # logodds = []
        # for batch in tqdm(batches, desc="Calling reporter", disable=len(batches) < 5):
        #     logodds.append(self.strong_model(inputs[batch[0] : batch[-1] + 1]))
        # return torch.cat(logodds)


class DivDisReporter(Reporter):
    ...


class DivDisFinetuningReporter(Reporter):
    ...


class DivDisProbingReporter(Reporter):
    # optionally finetunes on trusted examples first
    ...


REPORTER_REGISTRY: dict[str, type[Reporter]] = {
    c.__name__: c
    for c in locals().values()
    if isinstance(c, type) and issubclass(c, Reporter) and c is not Reporter
}
