from __future__ import annotations

import random
from abc import ABC
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from peft.tuners.lora.layer import LoraLayer
from torch import nn, optim
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from w2s.metrics import roc_auc
from w2s.model import Predictor, TransformerPredictor
from w2s.sft import lm_sft, prepare_for_trainer
from w2s.sft_utils import get_gpu_mem_used
from w2s.utils import (
    assert_type,
    ds_with_labels,
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
        self._df.set_index("id", inplace=True, drop=False)
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

    def __call__(self, inputs) -> torch.Tensor:
        """
        Returns the logodds of the classifier's predictions
        """
        ...

    def to_dict(self) -> dict[str, str | int | float]:
        """A summary of the method that approximately uniquely identifies it.
        It should include a name and all the important hyperparameters."""
        ...


class SftStage:
    modules_with_grad: Literal["all", "head", "body"] = "all"
    reinit_head: bool = False
    train_args: dict
    type: Literal["weak", "oracle"] = "weak"
    size: int = 1000
    sampling: Literal[
        "random", "most_confident_label", "least_confident_pred"
    ] = "random"
    n_test: int = 0
    weak_ids_used: set = set()

    def __init__(
        self,
        type: Literal["weak", "oracle"] = "weak",
        size: int = 1000,
        sampling: Literal[
            "random", "most_confident_label", "least_confident_pred"
        ] = "random",
        n_test: int = 0,
        modules_with_grad: Literal["all", "head", "body"] = "all",
        reinit_head: bool = False,
        sample_temp: float = 0.25,
        reuse_optimizer_checkpoint: bool = False,
        **kwargs,
    ):
        self.type = type
        self.size = size
        self.sampling = sampling
        self.n_test = n_test
        self.modules_with_grad = modules_with_grad
        self.reinit_head = reinit_head
        self.sample_temp = sample_temp
        self.reuse_optimizer_checkpoint = bool(reuse_optimizer_checkpoint)
        self.train_args = kwargs

    def get_dataset(
        self, oracle: Oracle, weak_ds: Dataset, test_ds: Dataset, reporter: ModularSftReporter
    ) -> DatasetDict:
        inputs = oracle.get_inputs() if self.type == "oracle" else weak_ds
        label_col = "soft_pred" if self.type == "weak" else "soft_label"

        if self.sampling == "random":
            idxs = random.sample(range(len(inputs)), self.size)
        elif self.sampling == "least_confident_pred":
            print("Selecting examples with highest reporter entropy for training.")
            pred_logodds = reporter(inputs["txt"])  # type: ignore
            probs = torch.nn.functional.sigmoid(pred_logodds)
            probs = torch.stack([1 - probs, probs], dim=-1)

            idxs = uncertainty_sample(
                probs, self.size, self.sample_temp, most_confident=False
            )
        elif self.sampling == "most_confident_label":
            print("Selecting examples with lowest label entropy for training.")
            probs = torch.softmax(
                torch.tensor(inputs[label_col], dtype=torch.float32), dim=-1
            )
            idxs = uncertainty_sample(
                probs, self.size, self.sample_temp, most_confident=True
            )
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

        if self.type == "oracle":
            ids = [inputs["id"].iloc[idx] for idx in idxs] if len(inputs) > 0 else []  # type: ignore
            train_ds = Dataset.from_pandas(oracle.query_ids(ids), preserve_index=False)
        else:
            train_ds = weak_ds.select(idxs)
            self.weak_ids_used.update(train_ds["id"])

        ds_dict = {"train": ds_with_labels(train_ds, labels_column=label_col)}
        if self.n_test > 0:
            ds_dict["test"] = ds_with_labels(
                test_ds.shuffle().select(range(self.n_test)), labels_column="soft_label"
            )
        return DatasetDict(**ds_dict)

    def run(
        self, reporter: ModularSftReporter, optimizer_checkpoint: Optional[str] = None
    ) -> str:
        assert isinstance(reporter.strong_model, TransformerPredictor)
        if reporter.strong_model.cfg.enable_lora:
            # TODO: support models without `score` attribute
            lora_params = [
                (*list(m.lora_A.parameters()), *list(m.lora_B.parameters()))
                for m in reporter.strong_model.transformer.modules()
                if isinstance(m, LoraLayer)
            ]
            lora_params = [p for params in lora_params for p in params]
            if self.modules_with_grad == "all":
                for p in lora_params:
                    p.requires_grad_()
                reporter.strong_model.transformer.score.requires_grad_()
            elif self.modules_with_grad == "head":
                reporter.strong_model.transformer.requires_grad_(False)
                reporter.strong_model.transformer.score.requires_grad_()
            elif self.modules_with_grad == "body":
                for p in lora_params:
                    if p.dtype not in {
                        torch.bfloat16,
                        torch.float16,
                        torch.float32,
                        torch.float64,
                    }:
                        print(f"Skipping parameter {p} with dtype {p.dtype}")
                    p.requires_grad_()
                reporter.strong_model.transformer.score.requires_grad_(False)
            else:
                raise ValueError(f"Unknown modules_with_grad: {self.modules_with_grad}")
        else:
            raise ValueError("Only Lora models are supported")

        if self.reinit_head:
            score_data = reporter.strong_model.transformer.score.weight.data
            score_data.normal_(0, 0.01 / score_data.shape[-1] ** 0.5)

        save_dir = Path(self.train_args["output_dir"])
        results_path = save_dir / "config.json"
        # we temporarily change the sampling method to avoid doing inference for cached training run data selection
        if results_path.exists() and (save_dir / "best-ckpt").exists():
            actual_sampling, self.sampling = self.sampling, "random"
        ds_dict = self.get_dataset(
            reporter.oracle, reporter.weak_ds, reporter.test_ds, reporter
        )
        if results_path.exists() and (save_dir / "best-ckpt").exists():
            self.sampling = actual_sampling
        train_args = self.train_args.copy()

        print(f"{get_gpu_mem_used()} before training")

        lm_sft(
            ds_dict=ds_dict,
            model=reporter.strong_model.transformer,
            tokenizer=reporter.strong_model.tokenizer,
            train_args=TrainingArguments(**train_args),
            loss="xent",
            store_pre_hiddens=False,
            store_post_hiddens=False,
            cfg=reporter.to_dict(),
            predict_dict=None,
            resume_from_checkpoint=optimizer_checkpoint
            if self.reuse_optimizer_checkpoint
            else None,
        )

        return f"{train_args['output_dir']}/best-ckpt/optimizer.pt"

    def to_dict(self) -> dict:
        return vars(self)


class ModularSftReporter(Reporter):
    strong_model: TransformerPredictor  # override type

    def __init__(
        self,
        weak_ds: Dataset,
        oracle: Oracle,
        test_ds: Dataset,  # for logging
        stages: list[SftStage],
        strong_model: TransformerPredictor,
        input_col: str = "txt",
    ):
        super().__init__(weak_ds, oracle, test_ds, strong_model, input_col)
        self.stages = stages
        self.test_ds = ds_with_labels(test_ds)

        assert input_col == "txt", "Only LM SFT is supported"

    def fit(self) -> ModularSftReporter:
        optimizer_checkpoint = None
        for i, stage_config in enumerate(self.stages):
            print(f"\n\033[32m [Stage {i}] \033[0m")
            optimizer_checkpoint = stage_config.run(self, optimizer_checkpoint)

        return self

    def to_dict(self) -> dict:
        return {
            "method": self.__class__.__name__,
            "stages": [s.to_dict() for s in self.stages],
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
        targs = self.stages[0].train_args.copy()
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
    bias = torch.nn.Parameter(torch.tensor(0.0))
    scale = torch.nn.Parameter(torch.tensor(1.0))

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
            Dataset.from_pandas(self.oracle.get_inputs(), preserve_index=False)
            .shuffle()
            .select(range(len(weak_ds)))  # NOTE: this is a hyperparameter
        )
        train_target_ds = train_target_ds.add_column(
            "labels", [-1.0] * len(train_target_ds)
        ).cast(weak_ds.features)  # type: ignore
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

        all_oracle_inputs = Dataset.from_pandas(
            self.oracle.get_inputs(), preserve_index=False
        )

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

        oracle_ids = [all_oracle_inputs["id"][idx] for idx in uncertain_idxs] if len(all_oracle_inputs) > 0 else []
        return Dataset.from_pandas(
            self.oracle.query_ids(oracle_ids), preserve_index=False
        ).shuffle()

    def _disambiguate(self, oracle_ds: Dataset) -> int:
        # get predictions from all heads
        pred_logits = self._call_all_heads(oracle_ds[self.input_col])

        # pick the head with the highest auroc on the oracle data
        labels = (torch.as_tensor(oracle_ds["labels"]) > 0.5).long()
        labels = labels.unsqueeze(-1).expand(-1, pred_logits.shape[-1])
        aurocs = roc_auc(labels, pred_logits)
        self.best_head = int(aurocs.argmax())
        return self.best_head

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
