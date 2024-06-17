import sys

from w2s.ds_registry import VALID_DATASETS
from w2s.run import run_train
from w2s.sft_config import SFTConfig
import w2s.loss
import w2s.probe

datasets = VALID_DATASETS.copy()
datasets.remove('amazon_polarity_gt')
datasets.remove('amazon_polarity_weak')

assert len(datasets) == 25

mode_names = [
    "xent",
    "strong2strong",
    "relabel_knn",
    "relabel_logreg",
    "filter_knn",
    "filter_logreg",
    "filter_topo",
    "window",
    "entropy",
]

def get_config(ds: int, mode: int):
    kwargs = {
        "weak_model_name": "Qwen/Qwen1.5-0.5B",
        "strong_model_name": "meta-llama/Meta-Llama-3-8B",
        "minibatch_size": 1,
        "eval_every": 25,
        "save_every": 25,
        "shared_folder": "shared",
        "dataset": datasets[ds],
        "run_name": f"{datasets[ds]}_{mode_names[mode]}",
    }
    if mode == 0:
        kwargs["loss"] = w2s.loss.CrossEntropyLossConfig()
    elif mode == 1:
        kwargs["s2s_iters"] = 2
    elif mode == 2:
        kwargs["probe"] = w2s.probe.KnnProbeConfig()
        kwargs["probe_relabel"] = True
    elif mode == 3:
        kwargs["probe"] = w2s.probe.LogisticProbeConfig()
        kwargs["probe_relabel"] = True
    elif mode == 4:
        kwargs["probe"] = w2s.probe.KnnProbeConfig()
        kwargs["probe_filter"] = True
    elif mode == 5:
        kwargs["probe"] = w2s.probe.LogisticProbeConfig()
        kwargs["probe_filter"] = True
    elif mode == 6:
        kwargs["probe"] = w2s.probe.TopoProbeConfig()
        kwargs["probe_filter"] = True
    elif mode == 7:
        kwargs["loss"] = w2s.loss.ConfidenceWindowLossConfig(radius="midweak")
    elif mode == 8:
        kwargs["loss"] = w2s.loss.LogEntropyLossConfig()

    return SFTConfig(**kwargs)


def run_experiment(expt: int):
    assert expt < len(datasets) * len(mode_names)
    
    print("Running experiment", expt)

    ds = expt % len(datasets)
    mode = expt // len(datasets)

    print("Dataset:", datasets[ds])
    print("Mode:", mode_names[mode])

    cfg = get_config(ds, mode)

    run_train(cfg)


if __name__ == "__main__":
    expt = int(sys.argv[1])

    run_experiment(expt)