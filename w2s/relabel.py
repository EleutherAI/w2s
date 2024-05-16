
import os, sys
import torch
import json
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from itertools import product
from collections import defaultdict

from simple_parsing import Serializable, field, parse, list_field
from tqdm import tqdm

from .ds_registry import load_and_process_dataset, _REGISTRY
from .knn import topo_relabel, zeta_relabel
from .roc_auc import roc_auc
from .probe import logreg_relabel


RESULT_DIR = "/mnt/ssd-1/{user}/w2s/results/"

@dataclass
class RelabelSweepConfig(Serializable):
    method: str
    resusers: list[str] = list_field('nora', 'alexm', 'adam')
    outuser: str = field(default='adam')
    outfile: str = field(default='relabel.json')
    save_labels: bool = field(default=False)

@dataclass
class RelabelResult:
    dataset: str
    method: str
    params: dict[str, list]
    metrics: dict[str, list]
    baseline_metrics: dict[str, float]

METHODS = {}

@dataclass
class TopoParams(RelabelSweepConfig):
    kcc: list[int] = list_field(20)
    kzeta: list[int] = list_field(20)

METHODS['topo'] = (topo_relabel, TopoParams)

@dataclass
class ZetaParams(RelabelSweepConfig):
    kzeta: list[int] = list_field(20)

METHODS['zeta'] = (zeta_relabel, ZetaParams)

def zeta_disagree_relabel(acts, wk, kzeta):
    zeta_labels = zeta_relabel(acts, wk, kzeta)
    hard_wk = wk > 0.5
    hard_zeta = zeta_labels > 0.5
    # replace wk labels with zeta labels where they disagree
    new_labels = torch.where(hard_wk == hard_zeta, wk, zeta_labels)
    return new_labels

METHODS['zeta_disagree'] = (zeta_disagree_relabel, ZetaParams)

# sanity check
METHODS['zeta_null'] = (lambda acts, wk, kzeta: wk, ZetaParams)

@dataclass
class LogRegParams(RelabelSweepConfig):
    l2p: list[float] = list_field(0.001)

METHODS['logreg'] = (logreg_relabel, LogRegParams)


@dataclass
class ZetaWorstParams(RelabelSweepConfig):
    kzeta: list[int] = list_field(20)
    contamination: list[float] = list_field(0.1)

def zeta_worst_relabel(acts, wk, kzeta, contamination):
    zeta_labels = zeta_relabel(acts, wk, kzeta)
    dists = torch.abs(wk - zeta_labels)
    n = round(len(wk) * contamination)
    filtered = dists.topk(n).indices
    new_labels = wk.clone()
    new_labels[filtered] = zeta_labels[filtered]
    return new_labels

METHODS['zeta_worst'] = (zeta_worst_relabel, ZetaWorstParams)

def get_datasets(resusers, require_acts=True, ds_names=None):
    datasets = {}

    for resuser in resusers:
        res_dir = Path(RESULT_DIR.format(user=resuser))
        for dataset in os.listdir(res_dir):
            # skip if not directory
            if not os.path.isdir(res_dir / dataset):
                continue
            # skip if not in list of datasets to process
            if ds_names is not None and dataset not in ds_names:
                continue

            if dataset in datasets:
                print(f"Dataset {dataset} already loaded, skipping -- check for duplicates")
            elif dataset not in _REGISTRY:
                print(f"Dataset {dataset} not in registry, skipping")
            elif require_acts and not (Path(res_dir) / dataset / "ceil/acts.pt").exists():
                print(f"Dataset {dataset} is missing activations, skipping")
            elif not (Path(res_dir) / dataset / "floor/preds/train.pt").exists():
                print(f"Dataset {dataset} is missing weak train labels, skipping")
            elif not (Path(res_dir) / dataset / "floor/preds/test.pt").exists():
                print(f"Dataset {dataset} is missing weak test labels, skipping")
            else:
                datasets[dataset] = Path(res_dir) / dataset

    return datasets

def compute_metrics(new_labels, gt):
    metrics = {}
    metrics['auc'] = roc_auc(gt, new_labels).cpu().item()
    metrics['acc'] = ((new_labels > 0.5) == gt).float().mean().cpu().item()
    return metrics