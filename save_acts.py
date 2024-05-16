import torch

from dataclasses import dataclass
from simple_parsing import Serializable, field, list_field, parse
from datasets import Value
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from w2s.knn import gather_hiddens
from w2s.relabel import get_datasets
from w2s.ds_registry import load_and_process_dataset

@dataclass
class SaveActsConfig(Serializable):
    resusers: list[str] = list_field('nora', 'alexm', 'adam')
    #outuser: str = field(default="adam")
    dataset: str = field(default="all")
    test_only: bool = field(default=True)


def load_model_and_tokenizer():
    STRONG_NAME = "meta-llama/Meta-Llama-3-8B"
    strong_tokenizer = AutoTokenizer.from_pretrained(STRONG_NAME)

    # Make sure that the pad token is set
    if strong_tokenizer.pad_token_id is None:
        strong_tokenizer.pad_token = strong_tokenizer.eos_token

    strong_model = AutoModelForSequenceClassification.from_pretrained(
        STRONG_NAME, torch_dtype="auto", device_map={"": "cuda"}
    )
    # HuggingFace init for the head is too large
    strong_model.score.weight.data *= 0.01
    strong_model.config.pad_token_id = strong_tokenizer.pad_token_id

    return strong_model, strong_tokenizer


def load_dataset(dataset: str, strong_tokenizer):
    splits = load_and_process_dataset(
        dataset, split_sizes=dict(train=20_000, test=1_000)
    )
    cols = ["hard_label", "txt"]
    test = splits["test"].select_columns(cols)
    train = splits["train"].select_columns(cols)

    def strong_processor(examples):
        return strong_tokenizer(examples["txt"], truncation=True)

    strong_train = (
        train.map(strong_processor, batched=True)
        .rename_column("hard_label", "labels")
        .cast_column("labels", Value("int64"))
    )
    strong_test = (
        test.map(strong_processor, batched=True)
        .rename_column("hard_label", "labels")
        .cast_column("labels", Value("int64"))
    )

    return strong_train, strong_test


def main(cfg: SaveActsConfig):
    if cfg.dataset == "all":
        print("Saving activations for all datasets")
        datasets = get_datasets(cfg.resusers, require_acts=False)
        print(f"Found {len(datasets)} datasets:")
        print(list(datasets.keys()))
        print("at paths:")
        for path in datasets.values():
            print(path)
    else:
        datasets = get_datasets(cfg.resusers, require_acts=False, ds_names=[cfg.dataset])

    strong_model, strong_tokenizer = load_model_and_tokenizer()

    for dataset, root in datasets.items():
        print()
        print(f"Processing dataset {dataset}")
        strong_train, strong_test = load_dataset(dataset, strong_tokenizer)

        splits = {
            "train": strong_train, 
            "test": strong_test
        }
        acts_paths = {
            'train': root / "ceil/acts.pt",
            'test': root / "ceil/acts_test.pt"
        }
        for split, acts_path in acts_paths.items():
            if acts_path.exists():
                print(f"Found strong {split} activations at {acts_path}")
            elif cfg.test_only and split == "train":
                print(f"Skipping training activations for {dataset}")
            else:
                print(f"Gathering strong {split} activations")
                acts = gather_hiddens(strong_model, splits[split])
                torch.save(acts, acts_path)
                print(f"Saved activations to {acts_path}")


if __name__ == "__main__":
    main(parse(SaveActsConfig))