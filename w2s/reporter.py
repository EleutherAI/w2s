import random
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from torch import nn, optim
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from w2s.metrics import roc_auc
from w2s.model import Predictor, TransformerPredictor
from w2s.sft import lm_sft, prepare_for_trainer
from w2s.utils import (
    assert_type,
    ds_with_labels,
    split_args_by_prefix,
    uncertainty_sample,
)


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
        if "labels" in self._df.columns:
            self._df.drop(columns=["labels"], inplace=True)
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
        return self._df.drop(columns=["soft_label", "hard_label"], inplace=False)

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
            model=self.strong_model.transformer,
            tokenizer=self.strong_model.tokenizer,
            train_args=TrainingArguments(**self.weak_train_args),
            loss="xent",
            store_pre_hiddens=False,
            store_post_hiddens=False,
            cfg=self.to_dict(),
            predict_dict=None,
        )

        if max_queries > 0:
            # then train on oracle
            oracle_ds = self.get_oracle_ds(max_queries)

            print(
                f"\n\033[32m===== {len(oracle_ds)} oracle queries finetuning =====\033[0m"
            )
            oracle_ds = ds_with_labels(oracle_ds)
            # add num_queries to the run name and output dir
            # the previous results will be cached, but these will not
            ota = self.oracle_train_args.copy()
            ota["run_name"] += f"_{max_queries}queries"
            ota["output_dir"] = f"{ota['output_dir']}_{max_queries}queries"
            min_use_test = 200
            ota["eval_strategy"] = "no" if max_queries < min_use_test else "steps"
            mult = 4
            ota["num_train_epochs"] = {
                1: 32 * mult,
                2: 32 * mult,
                4: 32 * mult,
                8: 32 * mult,
                16: 32 * mult,
                32: 32 * mult,
                64: 32 * mult,
                128: 16 * mult,
                256: 8 * mult,
                512: 4 * mult,
                1024: 2 * mult,
                2048: 1 * mult,
                4096: max(int(0.5 * mult), 1),
                8192: max(int(0.25 * mult), 1),
                16384: max(int(0.125 * mult), 1),
            }[max_queries]
            if max_queries < min_use_test:
                oracle_dict = DatasetDict(train=oracle_ds)
            else:
                oracle_dict = DatasetDict(
                    train=oracle_ds, test=self.test_ds  # type: ignore
                )
            lm_sft(
                ds_dict=oracle_dict,
                model=self.strong_model.transformer,
                tokenizer=self.strong_model.tokenizer,
                train_args=TrainingArguments(**ota),
                loss="xent",
                store_pre_hiddens=False,
                store_post_hiddens=False,
                cfg=self.to_dict(),
                predict_dict=None,
                resume_from_checkpoint=f"{self.weak_train_args['output_dir']}/best-ckpt/optimizer.pt",
            )

        return self

    def get_oracle_ds(self, max_queries: int) -> Dataset:
        random_oracle_ids = random.sample(
            self.oracle.get_inputs().index.values.tolist(), max_queries
        )
        return Dataset.from_pandas(self.oracle.query_ids(random_oracle_ids))

    def to_dict(self) -> dict:
        return {
            "method": self.__class__.__name__,
            "weak_train_args": self.weak_train_args,
            "oracle_train_args": self.oracle_train_args,
            "model": self.strong_model.to_dict(),
        }

    def __call__(self, inputs: list) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions
        """
        predict_ds = prepare_for_trainer(
            Dataset.from_dict({self.input_col: inputs}), self.strong_model.tokenizer
        )
        # turn off wandb logging in trainer
        targs = self.weak_train_args.copy()
        targs["report_to"] = "none"
        targs["output_dir"] = "tmp"
        targs["run_name"] = "tmp"
        trainer = Trainer(
            args=TrainingArguments(**targs),
            data_collator=DataCollatorWithPadding(
                self.strong_model.tokenizer, max_length=1024, padding="max_length"
            ),  # NOTE: this could mess up some datasets
            model=self.strong_model.transformer,
            tokenizer=self.strong_model.tokenizer,
        )
        pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)  # type: ignore # noqa
        return pred_logits.diff(dim=-1).squeeze()


class ActiveSftReporter(SftReporter):
    strong_model: TransformerPredictor  # override type

    def get_oracle_ds(self, max_queries: int) -> Dataset:
        # get reporter predictions on all data, and pick the `max_queries` most uncertain
        all_oracle_inputs = Dataset.from_pandas(self.oracle.get_inputs())

        print("Selecting examples with highest entropy for training.")

        pred_logodds = self(all_oracle_inputs["txt"])
        probs = torch.nn.functional.sigmoid(pred_logodds)
        probs = torch.stack([1 - probs, probs], dim=-1)

        uncertain_idxs = uncertainty_sample(
            probs, max_queries, "sample", most_confident=False
        )
        oracle_ids = np.array(all_oracle_inputs["id"])[uncertain_idxs].tolist()
        np.random.shuffle(oracle_ids)
        return Dataset.from_pandas(self.oracle.query_ids(oracle_ids))


class DivDisSftReporter(Reporter):
    """
    Diversify and Disambiguate finetuning: https://arxiv.org/abs/2202.03418

    Diversification:
    - Train multiple heads with different random initializations
    - Use trusted (weak) data with xent loss and diversify on unlabeled target (oracle) data
    Disambiguation:
    - Use oracle examples to select a head
    - Calibrate that head with Platt scaling
    """

    best_head: int
    bias: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(0.0))
    scale: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(1.0))

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
        self.weak_train_args = kwargs
        self.weak_train_args[
            "run_name"
        ] = f"div_{self.weak_train_args.get('run_name', 'default')}"
        self.weak_train_args["output_dir"] = str(Path(save_dir) / "div")

        assert input_col == "txt", "Only LM SFT is supported"

    def fit(self, max_queries: int) -> "DivDisSftReporter":
        # ### Diversification ###
        # we use label -1 for target data, and pass a custom loss function that deals
        # with -1 examples separately
        weak_ds = ds_with_labels(self.weak_ds, labels_column="soft_pred")
        train_target_ds = (
            Dataset.from_pandas(self.oracle.get_inputs())
            .shuffle()
            .select(range(len(weak_ds)))  # NOTE: this is a hyperparameter
        )
        train_target_ds = train_target_ds.add_column(
            "labels", [-1.0] * len(train_target_ds)
        ).cast(weak_ds.features)
        weak_ds = concatenate_datasets([weak_ds, train_target_ds])
        weak_ds_dict = DatasetDict(train=weak_ds, test=self.test_ds)

        self.div_trainer = lm_sft(
            ds_dict=weak_ds_dict,
            model=self.strong_model.transformer,
            tokenizer=self.strong_model.tokenizer,
            train_args=TrainingArguments(**self.weak_train_args),
            loss="divdis",
            store_pre_hiddens=False,
            store_post_hiddens=False,
            cfg=self.to_dict(),
            predict_dict=None,
        )

        # then disambiguate
        if max_queries > 0:
            oracle_ds = ds_with_labels(self.get_oracle_ds(max_queries))
            self._disambiguate(oracle_ds)
            self._platt_scale(oracle_ds)
        else:
            self.best_head = 0

        return self

    def get_oracle_ds(self, max_queries: int) -> Dataset:
        # Select examples according to the amount of disagreement between heads
        # Lee et al. use total distance between head predictions (hardened, I believe)
        # but we would prefer to also care about the confidence of disagreements
        # so we use the total cross entropy between every pair of heads

        all_oracle_inputs = Dataset.from_pandas(self.oracle.get_inputs())

        print(
            "Selecting examples with average cross entropy between pairs of heads for training."
        )

        pred_logodds = self._call_all_heads(all_oracle_inputs["txt"])
        logprobs = torch.nn.functional.logsigmoid(pred_logodds)  # [b, h]
        log1mprobs = torch.nn.functional.logsigmoid(-pred_logodds)
        probs = logprobs.exp()
        # xent = -p * log(q) - (1-p) * log(1-q) for each pair p, q
        xents = -torch.einsum("bh,bg->bhg", probs, logprobs) - torch.einsum(
            "bh,bg->bhg", 1 - probs, log1mprobs
        )  # [b, h, h]
        avg_xents = xents.mean(dim=-1).mean(dim=-1)  # [b]

        uncertain_idxs = torch.multinomial(avg_xents, max_queries, replacement=False)

        oracle_ids = np.array(all_oracle_inputs["id"])[uncertain_idxs].tolist()
        return Dataset.from_pandas(self.oracle.query_ids(oracle_ids)).shuffle()

    def _disambiguate(self, oracle_ds: Dataset) -> int:
        # get predictions from all heads
        pred_logits = self._call_all_heads(oracle_ds[self.input_col])

        # pick the head with the highest auroc on the oracle data
        labels = (torch.as_tensor(oracle_ds["labels"]) > 0.5).long()
        labels = labels.unsqueeze(-1).expand(-1, pred_logits.shape[-1])
        aurocs = roc_auc(labels, pred_logits)
        self.best_head = aurocs.argmax().item()

    def _platt_scale(self, oracle_ds: Dataset, max_iter: int = 100) -> None:
        """Fit the scale and bias terms to data with LBFGS.

        Args:
            oracle_ds: Dataset with columns ["txt", "labels"]
            max_iter: Maximum number of iterations for LBFGS.
        """
        opt = optim.LBFGS(
            [self.bias, self.scale],
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
            tolerance_change=torch.finfo(torch.bfloat16).eps,
            tolerance_grad=torch.finfo(torch.bfloat16).eps,
        )

        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy(
                torch.sigmoid(self(oracle_ds[self.input_col])),
                torch.as_tensor(oracle_ds["labels"]).float(),
            )

            loss.backward()
            return float(loss)

        opt.step(closure)

    def _call_all_heads(self, inputs: list) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions for all heads
        """
        predict_ds = prepare_for_trainer(
            Dataset.from_dict({self.input_col: inputs}), self.strong_model.tokenizer
        )
        # turn off wandb logging in trainer
        targs = self.weak_train_args.copy()
        targs["report_to"] = "none"
        targs["output_dir"] = "tmp"
        targs["run_name"] = "tmp"
        trainer = Trainer(
            args=TrainingArguments(**targs),
            data_collator=DataCollatorWithPadding(
                self.strong_model.tokenizer, max_length=1024, padding="max_length"
            ),  # NOTE: this could mess up some datasets
            model=self.strong_model.transformer,
            tokenizer=self.strong_model.tokenizer,
        )
        pred_logits = torch.from_numpy(trainer.predict(predict_ds).predictions)  # type: ignore # noqa
        return pred_logits.diff(dim=-1).squeeze()

    def __call__(self, inputs: list) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions from its best head
        inputs: a list of strings
        """
        assert hasattr(self, "best_head"), "Must fit before calling"
        lo = self._call_all_heads(inputs)[..., self.best_head]
        return self.scale.to(lo.dtype).to(lo.device) * lo + self.bias.to(lo.dtype).to(
            lo.device
        )

    def to_dict(self) -> dict:
        return {
            "method": self.__class__.__name__,
            "weak_train_args": self.weak_train_args,
            "model": self.strong_model.to_dict(),
        }


class DivDisProbingReporter(Reporter):
    # optionally finetunes on trusted examples first
    ...


REPORTER_REGISTRY: dict[str, type[Reporter]] = {
    c.__name__: c
    for c in locals().values()
    if isinstance(c, type) and issubclass(c, Reporter) and c is not Reporter
}
