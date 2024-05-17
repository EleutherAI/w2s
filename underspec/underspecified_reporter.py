from abc import ABC

import pandas as pd
import torch
from datasets import Dataset

from underspec.model import Predictor


class Oracle:
    _df: pd.DataFrame
    ids_labeled: set

    def __init__(self, gt_dataset: Dataset) -> None:
        assert (
            "id" in gt_dataset.column_names and "soft_label" in gt_dataset.column_names
        )
        assert "gt_soft_label" not in gt_dataset.column_names

        self._df = gt_dataset.to_pandas()  # type: ignore
        self._df.set_index("id", inplace=True)

        self.ids_labeled = set()

    def query(self, id: str):
        self.ids_labeled.add(id)
        return self._df.loc[id]["soft_label"]

    def reset(self):
        self.ids_labeled = set()


class UnderspecifiedReporter(ABC):
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
    strong_model: Predictor

    def __init__(
        self,
        weak_ds: Dataset,
        oracle: Oracle,
        strong_model: Predictor,
        input_col: str = "input_ids",
    ):
        """
        weak_ds: a dataset with columns ["id", input_col, "soft_pred"]
        oracle: an Oracle object
        strong_model: a
        input_col: the column in weak_ds that contains the input

        """
        weak_ds.set_format(type="torch", columns=[input_col])
        assert input_col in weak_ds.column_names
        assert "soft_pred" in weak_ds.column_names
        assert "id" in weak_ds.column_names
        self.weak_ds = weak_ds

        self.oracle = oracle

        self.strong_model = strong_model

    def fit(self, max_queries: int) -> "UnderspecifiedReporter":
        """
        max_queries: the maximum number of queries to the oracle
        """
        ...

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions
        """
        ...

    def get_cfg_summary(self) -> dict[str, str | int | float]:
        """A summary of the method that approximately uniquely identifies it.
        It should include a name and all the important hyperparameters."""
        ...


class W2SReporter(UnderspecifiedReporter):
    """
    A weak-to-strong reporter that uses the same weakly-labeled
    dataset to train all candidate classifiers.
    """

    def __init__(self, weak_dataset, strong_model, num_classifiers: int, **kwargs):
        # self.train_config = TrainConfig(**kwargs)
        ...

    def fit(self, max_queries: int) -> "W2SReporter":
        # typically you'll not want to do any test examples,
        # and instead just use val for early stopping
        # train_simple.main(self.train_config)
        # TODO: load model
        raise NotImplementedError()

        # e.g. train on some unconfident data, and use
        return self

    def get_cfg_summary(self) -> dict[str, str | int | float]:
        ...


class DivDisReporter(UnderspecifiedReporter):
    ...


class DivDisFinetuningReporter(UnderspecifiedReporter):
    ...


class DivDisProbingReporter(UnderspecifiedReporter):
    # optionally finetunes on trusted examples first
    ...


REPORTER_REGISTRY: dict[str, type[UnderspecifiedReporter]] = {
    c.__name__: c
    for c in locals().values()
    if isinstance(c, type)
    and issubclass(c, UnderspecifiedReporter)
    and c is not UnderspecifiedReporter
}
