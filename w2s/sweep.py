from argparse import ArgumentParser

from w2s.ds_registry import _REGISTRY
from w2s.train import TrainConfig, train


def main():
    parser = ArgumentParser()
    parser.add_argument("rank", type=int)
    args = parser.parse_args()

    DATASETS = sorted(_REGISTRY.keys())
    train(TrainConfig("Qwen/Qwen1.5-0.5B", DATASETS[args.rank]))


if __name__ == "__main__":
    main()
