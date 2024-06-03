import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from w2s.metrics import acc_ci, roc_auc_ci
from w2s.model import Predictor
from w2s.reporter import REPORTER_REGISTRY, Oracle


@dataclass
class ExperimentConfig:
    reporter_method: str
    max_num_oracle: int = 256
    results_folder: str = "./results"
    run_name: str = "default"
    input_col: str = "txt"


def train_and_eval_reporter(
    # this dataset is cheap to query but may not have perfect labels
    weak_ds: Dataset,
    # this dataset is expensive to query but has perfect labels
    oracle_ds: Dataset,
    # this dataset is untrusted
    test_ds: Dataset,
    strong_model: Predictor,
    cfg: ExperimentConfig,
    dataset_cfg_dict: dict,
    **reporter_args,
):
    predictor_cls = strong_model.__class__
    reporter_cls = REPORTER_REGISTRY[cfg.reporter_method]

    # fit reporter, with various numbers of queries allowed to the oracle
    results = []
    base = 4
    for num_queries in [0] + [
        base**i for i in range(1, int(np.log(cfg.max_num_oracle) / np.log(base)) + 1)
    ]:
        curr_results_path = (
            Path(cfg.results_folder) / cfg.run_name / f"results_{num_queries}.json"
        )
        if curr_results_path.exists():
            print(
                f"Results for {num_queries} queries already exist at {curr_results_path}."
            )
            with open(curr_results_path, "r") as f:
                results.append(json.load(f))
            continue
        print(
            f"\n\033[32m===== Training reporter with {num_queries} oracle queries =====\033[0m"
        )
        # load a new predictor each time, since the weights
        # are often changed by the reporter
        strong_model = predictor_cls(strong_model.cfg)

        # load reporter
        reporter = reporter_cls(
            weak_ds=weak_ds,
            oracle=Oracle(oracle_ds),
            test_ds=test_ds,
            strong_model=strong_model,
            input_col=cfg.input_col,
            save_dir=str(Path(cfg.results_folder) / cfg.run_name),
            **reporter_args,
        )

        reporter.fit(num_queries)
        with torch.no_grad():
            cal_logodds = reporter(test_ds[reporter.input_col])  # type: ignore

        cal_logodds = cal_logodds.cpu().float().numpy()
        gt_labels = np.array(test_ds["soft_label"])[:, 1]
        if not ((gt_labels == 0) | (gt_labels == 1)).all():
            warnings.warn(
                "Ground truth labels are not binary, so we're thresholding them."
            )
        auc_result = roc_auc_ci(gt_labels > 0.5, cal_logodds)
        acc_result = acc_ci((cal_logodds > 0), (gt_labels > 0.5))

        if "soft_pred" in test_ds.column_names:
            weak_test_labels = np.array(test_ds["soft_pred"])[:, 1]
            weak_auc_result = roc_auc_ci(weak_test_labels > 0.5, cal_logodds)
            weak_acc_result = acc_ci((cal_logodds > 0), (weak_test_labels > 0.5))

            weak_results = {
                "auroc_against_weak": float(weak_auc_result.estimate),
                "auroc_against_weak_lo": float(weak_auc_result.lower),
                "auroc_against_weak_hi": float(weak_auc_result.upper),
                "acc_against_weak": float(weak_acc_result.estimate),
                "acc_against_weak_lo": float(weak_acc_result.lower),
                "acc_against_weak_hi": float(weak_acc_result.upper),
                "weak_soft_labels": weak_test_labels.tolist(),
            }
        else:
            weak_results = {}

        result = {
            "auroc": float(auc_result.estimate),
            "auroc_lo": float(auc_result.lower),
            "auroc_hi": float(auc_result.upper),
            "acc": float(acc_result.estimate),
            "acc_lo": float(acc_result.lower),
            "acc_hi": float(acc_result.upper),
            **weak_results,
            "num_weak": len(weak_ds),
            "num_oracle": len(
                reporter.oracle.ids_labeled
            ),  # could be diff from num_queries
            "oracle_ids": list(reporter.oracle.ids_labeled),
            "ids": test_ds["id"],
            "calibrated_logodds": cal_logodds.tolist(),
            "gt_soft_labels": gt_labels.tolist(),
        }
        with open(curr_results_path, "w") as f:
            json.dump(result, f, indent=2)
        print(
            {
                k: v
                for k, v in result.items()
                if k
                not in [
                    "calibrated_logodds",
                    "gt_soft_labels",
                    "weak_soft_labels",
                    "ids",
                    "oracle_ids",
                ]
            }
        )
        results.append(result)

    # save configuration
    config: dict = {
        "dataset": dataset_cfg_dict,
        "model": strong_model.to_dict(),
        "max_num_oracle": cfg.max_num_oracle,
        "reporter": reporter.to_dict(),
    }
    save_path = Path(cfg.results_folder) / cfg.run_name

    os.makedirs(save_path, exist_ok=True)
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # save results
    df = pd.DataFrame(results)
    df.to_json(save_path / "results.json", orient="records", lines=True)
