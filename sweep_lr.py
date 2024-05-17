from argparse import ArgumentParser

from train import TrainConfig, train


def main():
    lrs = [2e-5, 5e-5, 8e-5, 2e-4]

    parser = ArgumentParser()
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()

    dataset = "cosmos_qa"
    strong_model = "mistralai/Mistral-7B-v0.1"
    strong_model_last = strong_model.split("/")[-1]
    train(
        TrainConfig(
            "Qwen/Qwen1.5-0.5B",
            dataset,
            lr=lrs[args.rank],
            run_name=f"_{strong_model_last}_lr_{lrs[args.rank]:.1e}",
            strong_model=strong_model,
            minibatch_size=2,
        )
    )


if __name__ == "__main__":
    main()
