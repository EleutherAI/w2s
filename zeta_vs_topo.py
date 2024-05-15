
import os
import torch
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from simple_parsing import Serializable, field, parse, list_field

from w2s.ds_registry import load_and_process_dataset, _REGISTRY
from w2s.knn import gather_hiddens, topofilter
from w2s.roc_auc import roc_auc


@dataclass
class RunConfig(Serializable):
    kcc: list[int] = list_field(10, 20, 50)
    kzeta: int = field(default=20)
    resuser: str = field(default='nora')
    outuser: str = field(default='adam')


def main(cfg: RunConfig):
    RES_DIR = f"/mnt/ssd-1/{cfg.resuser}/w2s/results/"
    OUT_DIR = f"/mnt/ssd-1/{cfg.outuser}/w2s/results/"

    datasets = os.listdir(RES_DIR)

    for ds_name in datasets:
        # check if ds/ceil/acts.pt exists
        if not (Path(RES_DIR) / ds_name / "ceil/acts.pt").exists():
            datasets.remove(ds_name)
            print(f"Dataset {ds_name} is missing activations, skipping")

    print(f"Found {len(datasets)} datasets:")
    print(datasets)

    for ds_name in datasets:
        scan_topo(ds_name, cfg)


def scan_topo(ds_name, cfg):
    RES_DIR = f"/mnt/ssd-1/{cfg.resuser}/w2s/results/"
    OUT_DIR = f"/mnt/ssd-1/{cfg.outuser}/w2s/results/"

    print(f"Scanning {ds_name}")
    print(f"Loading dataset {ds_name}")
    splits = load_and_process_dataset(ds_name, split_sizes=dict(train=20_000, test=1_000))
    train, test = splits["train"], splits["test"]
    
    root = Path(RES_DIR) / ds_name
    label_dir = root / "floor/preds"
    
    print(f"Loading weak labels from {label_dir}")
    train_probs = torch.load(label_dir / "train.pt")
    test_probs = torch.load(label_dir / "test.pt")
    print(f"Loading activations from {root / 'ceil/acts.pt'}")
    acts = torch.load(root / "ceil/acts.pt").to(torch.float)

    
    contamination = 0.2
    y = train_probs.to(acts.device)

    gt = torch.tensor(train['hard_label']).to(acts.device).to(torch.float)
    wk = y.to(acts.device).to(torch.float)

    print('==================== ' + ds_name + ' ====================')
    
    print(f"ROC AUC, Acc unfiltered:\t {roc_auc(gt, wk)} \t {((wk > 0.5) == gt).to(torch.float).mean()}")

    aucs, accs, fracs = [], [], []
    good_ks = []

    print(f"Scanning k_CC: {cfg.kcc}")
    print(f"With k_Zeta: {cfg.kzeta}")

    for k_cc in cfg.kcc:
        indices = topofilter(acts, y, contamination, k=cfg.kzeta, kcc=k_cc)

        auc = roc_auc(gt[indices], wk[indices])
        acc = ((wk[indices] > 0.5) == gt[indices]).to(torch.float).mean()

        frac = indices.shape[-1] / y.shape[-1]
        # if indices.shape[-1] < (.99 - contamination) * y.shape[-1]:
        #     print(f"ROC AUC, Acc filtered k_CC={k_cc}:\t {auc} \t {acc} \t {indices.shape[-1] / y.shape[-1]} \t SKIPPED.", flush=True)
        # else:
        aucs.append(auc.cpu().item())
        accs.append(acc.cpu().item())
        fracs.append(frac)
        good_ks.append(k_cc)

        print(f"ROC AUC, Acc, frac filtered k_CC={k_cc}:\t {auc} \t {acc} \t {frac}", flush=True)

    # make and save plot

    plt.plot(good_ks, aucs, label='AUC')
    plt.plot(good_ks, accs, label='Acc')
    plt.xlabel('k_CC')
    # plt.xscale('log')
    plt.ylabel('Performance')
    plt.title(f'ds = {ds_name}, k_Zeta = {cfg.kzeta}')
    plt.legend()
    plt.savefig(OUT_DIR + f"topo_filter_perf_{ds_name}.png")
    plt.clf()

    plt.plot(good_ks, fracs, label='frac kept')
    plt.xlabel('k_CC')
    # plt.xscale('log')
    plt.ylabel('Fraction of points kept')
    plt.title(f'ds = {ds_name}, k_Zeta = {cfg.kzeta}')
    plt.savefig(OUT_DIR + f"topo_filter_frac_{ds_name}.png")
    plt.clf()
    
    return


if __name__ == "__main__":
    main(parse(RunConfig))