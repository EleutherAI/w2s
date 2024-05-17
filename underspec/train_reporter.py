import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import roc_auc_score

from underspec.model import Predictor
from underspec.underspecified_reporter import REPORTER_REGISTRY, Oracle
from underspec.utils import get_config_foldername


@dataclass
class ExperimentConfig:
    reporter_method: str
    max_num_queries: int = 256
    results_folder: str = "./results"
    run_name: str = "default"
    input_col: str = "txt"


def train_reporter(
    # this dataset is "trusted" but may not have perfect labels
    weak_ds: Dataset,
    # this dataset is untrusted
    test_ds: Dataset,
    strong_model: Predictor,
    cfg: ExperimentConfig,
    dataset_cfg_dict: dict,
    **reporter_args,
):
    # load reporter
    reporter_cls = REPORTER_REGISTRY[cfg.reporter_method]
    reporter = reporter_cls(
        weak_ds=weak_ds,
        oracle=Oracle(
            test_ds
        ),  # note that in this setting we allow some access to the test set
        strong_model=strong_model,
        input_col=cfg.input_col,
        **reporter_args,
    )

    # fit reporter, with various numbers of queries allowed to the oracle
    results = []
    for num_queries in [0] + [
        2**i for i in range(int(np.log2(cfg.max_num_queries)) + 1)
    ]:
        reporter.fit(num_queries)
        cal_logodds = reporter(weak_ds[reporter.input_col])  # type: ignore
        cal_logodds = cal_logodds.detach().cpu().float().numpy()
        gt_labels = np.array(test_ds["soft_label"])
        if not ((gt_labels == 0) | (gt_labels == 1)).all():
            warnings.warn(
                "Ground truth labels are not binary, so we're thresholding them."
            )

        auc = roc_auc_score(gt_labels > 0.5, cal_logodds)
        acc = ((cal_logodds > 0) == (gt_labels > 0.5)).mean()
        results.append(
            {
                "num_queries": len(
                    reporter.oracle.ids_labeled
                ),  # this could be different from num_queries
                "ids_queried": reporter.oracle.ids_labeled,
                "ids": test_ds["id"],
                "calibrated_logodds": cal_logodds,
                "gt_soft_labels": gt_labels,
                "auroc": auc,
                "accuracy": acc,
            }
        )
        reporter.oracle.reset()

    # save configuration
    config: dict = {
        "dataset": dataset_cfg_dict,
        "model": strong_model.get_cfg_summary(),
        "max_num_queries": cfg.max_num_queries,
        "reporter": reporter.get_cfg_summary(),
    }
    save_path = Path(cfg.results_folder) / cfg.run_name / get_config_foldername(config)
    os.makedirs(save_path, exist_ok=True)
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # save results
    df = pd.DataFrame(results)
    df.to_json(save_path / "results.json", orient="records", lines=True)
