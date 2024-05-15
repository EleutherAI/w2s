
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


RESULT_DIR = "/mnt/ssd-1/{user}/w2s/results/"

@dataclass
class RelabelSweepConfig(Serializable):
    method: str
    resusers: list[str] = list_field('nora', 'alexm', 'adam')
    outuser: str = field(default='adam')
    outfile: str = field(default='relabel.json')

@dataclass
class RelabelResult:
    dataset: str
    method: str
    params: dict[str, list]
    metrics: dict[str, list]
    baseline_metrics: dict[str, float]

_METHODS = {}

@dataclass
class TopoParams(RelabelSweepConfig):
    kcc: list[int] = list_field(20)
    kzeta: list[int] = list_field(20)

_METHODS['topo'] = (topo_relabel, TopoParams)

@dataclass
class ZetaParams(RelabelSweepConfig):
    kzeta: list[int] = list_field(20)

_METHODS['zeta'] = (zeta_relabel, ZetaParams)

def zeta_disagree_relabel(acts, wk, kzeta):
    zeta_labels = zeta_relabel(acts, wk, kzeta)
    hard_wk = wk > 0.5
    hard_zeta = zeta_labels > 0.5
    # replace wk labels with zeta labels where they disagree
    new_labels = torch.where(hard_wk == hard_zeta, wk, zeta_labels)
    return new_labels

_METHODS['zeta_disagree'] = (zeta_disagree_relabel, ZetaParams)

# sanity check
_METHODS['zeta_null'] = (lambda acts, wk, kzeta: wk, ZetaParams)

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

_METHODS['zeta_worst'] = (zeta_worst_relabel, ZetaWorstParams)

def get_datasets(resusers):
    datasets  = {}

    for resuser in resusers:
        res_dir = Path(RESULT_DIR.format(user=resuser))
        for dataset in os.listdir(res_dir):
            # skip if not directory
            if not os.path.isdir(res_dir / dataset):
                continue

            if dataset in datasets:
                print(f"Dataset {dataset} already loaded, skipping -- check for duplicates")
            elif dataset not in _REGISTRY:
                print(f"Dataset {dataset} not in registry, skipping")
            elif not (Path(res_dir) / dataset / "ceil/acts.pt").exists():
                print(f"Dataset {dataset} is missing activations, skipping")
            elif not (Path(res_dir) / dataset / "floor/preds/train.pt").exists():
                print(f"Dataset {dataset} is missing weak train labels, skipping")
            elif not (Path(res_dir) / dataset / "floor/preds/test.pt").exists():
                print(f"Dataset {dataset} is missing weak test labels, skipping")
            else:
                datasets[dataset] = Path(res_dir) / dataset

    return datasets

def make_grid(param_lists):
    keys = list(param_lists.keys())
    values = list(param_lists.values())
    for i, v in enumerate(values):
        if not isinstance(v, list):
            values[i] = [v]
    return [{k: v for k, v in zip(keys, vals)} for vals in product(*values)]

def compute_metrics(new_labels, gt):
    metrics = {}
    metrics['auc'] = roc_auc(gt, new_labels).cpu().item()
    metrics['acc'] = ((new_labels > 0.5) == gt).float().mean().cpu().item()
    return metrics

def relabel_sweep(cfg, dataset, root):
    method = cfg.method
    param_lists = cfg.to_dict()
    # remove RelabelSweepConfig fields
    for field in fields(RelabelSweepConfig):
        del param_lists[field.name]

    print(f"Scanning {dataset}")
    print(f"Loading dataset {dataset}")
    splits = load_and_process_dataset(dataset, split_sizes=dict(train=20_000, test=1_000))
    train, test = splits["train"], splits["test"]

    train_probs = torch.load(root / "floor/preds/train.pt")
    test_probs = torch.load(root / "floor/preds/test.pt")
    acts = torch.load(root / "ceil/acts.pt")
    gt = torch.tensor(train['hard_label'])

    acts = acts.float().cuda()
    wk = train_probs.float().to(acts.device)
    gt = gt.float().to(acts.device)

    grid = make_grid(param_lists)

    method_fn, _ = _METHODS[method]

    result_params = defaultdict(list)
    result_metrics = defaultdict(list)

    print(f"Running relabel sweep for {method} on {dataset}")
    for params in tqdm(grid):
        new_labels = method_fn(acts, wk, **params)
        #breakpoint()
        metrics = compute_metrics(new_labels, gt)
        for k, v in params.items():
            result_params[k].append(v)
        for k, v in metrics.items():
            result_metrics[k].append(v)

    baseline_metrics = compute_metrics(wk, gt)
    #breakpoint()
    result_params = dict(result_params)
    result_metrics = dict(result_metrics)
    return asdict(RelabelResult(
        dataset=dataset,
        method=method,
        params=result_params,
        metrics=result_metrics,
        baseline_metrics=baseline_metrics,
    ))

def main(method):
    _, method_param_cls = _METHODS[method]
    cfg = parse(method_param_cls)  # e.g. TopoParams


    out_dir = RESULT_DIR.format(user=cfg.outuser)
    out_file = Path(out_dir) / cfg.outfile
    
    # get existing results
    if out_file.exists():
        with open(out_file, 'r') as f:
            results = json.load(f)
    else:
        results = []

    datasets = get_datasets(cfg.resusers)

    print(f"Found {len(datasets)} datasets:")
    print(list(datasets.keys()))
    print("at paths:")
    for path in datasets.values():
        print(path)

    for dataset, root in datasets.items():
        result = relabel_sweep(cfg, dataset, root)
        results.append(result)
        with open(out_file, 'w') as f:
            json.dump(results, f)

if __name__ == "__main__":
    assert sys.argv[1] == '--method'
    method = sys.argv[2]
    print(f"Running relabel sweep for method {method}")
    main(method)