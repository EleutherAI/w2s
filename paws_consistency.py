from pathlib import Path
from typing import Union

import torch
from datasets import Dataset, DatasetDict
from fire import Fire
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from w2s.ds_registry import encode_choice, load_and_process_dataset


def add_idx_col(ds: Union[DatasetDict, Dataset]) -> Union[DatasetDict, Dataset]:
    if isinstance(ds, DatasetDict):
        for split in ds:
            ds[split] = add_idx_col(ds[split])
        return ds
    else:
        ds = ds.add_column("idx", range(len(ds)))  # type: ignore
        return ds


few_shot_prompt = """
(1) In Paris , in October 1560 , he secretly met the English ambassador , Nicolas Throckmorton , asking him for a passport to return to England through Scotland .
(2) In October 1560 , he secretly met with the English ambassador , Nicolas Throckmorton , in Paris , and asked him for a passport to return to Scotland through England .
(1) and (2) are inconsistent.

(1) The NBA season of 1975 -- 76 was the 30th season of the National Basketball Association .
(2) The 1975 -- 76 season of the National Basketball Association was the 30th season of the NBA .
(1) and (2) are consistent.

(1) There are also specific discussions , public profile debates and project discussions .
(2) There are also public discussions , profile specific discussions , and project discussions .
(1) and (2) are consistent.

(1) When comparable rates of flow can be maintained , the results are high .
(2) The results are high when comparable flow rates can be maintained .
(1) and (2) are consistent.

(1) For their performances in the game , quarterback Jameis Winston and defensive back P. J. Williams were named the game 's most valuable players .
(2) Quarterback P. J. Williams and Defensive Back Jameis Winston were named the most valuable players of the game for their performances in the game .
(1) and (2) are inconsistent.

(1) {sentence1}
(2) {sentence2}
(1) and (2) are
""".strip()


def main(
    n_train: int = 50_000,
    n_test: int = 5_000,
    annotator_model: str = "meta-llama/Meta-Llama-3-8B",
    batch_size=8,
):
    # load dataset
    source_ds = load_and_process_dataset("paws", int(n_train), 0, int(n_test), 0)

    tokenizer = AutoTokenizer.from_pretrained(annotator_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        annotator_model, device_map={"": "cuda"}, torch_dtype="auto"
    )
    model.eval()

    # prompt model to determine if the two paraphrases are consistent
    def fs_prompt_format(ex):
        txt = few_shot_prompt.format(
            sentence1=ex["sentence1"], sentence2=ex["sentence2"]
        )
        return {"txt": txt}

    ds_dict = source_ds.map(fs_prompt_format)
    ds_dict = {
        "train": ds_dict["train"],
        "test": ds_dict["test"],
    }

    # get logodds for consistency, and normalize them to have mean 0
    with torch.inference_mode():
        target_toks = [
            encode_choice("inconsistent", tokenizer),
            encode_choice("consistent", tokenizer),
        ]
        for split, ds in ds_dict.items():
            results = []
            for i in tqdm(range(0, len(ds), batch_size), desc=f"Labeling {split}"):
                batch = ds[i : i + batch_size]
                encodings = tokenizer(
                    batch["txt"], padding=True, truncation=True, return_tensors="pt"
                ).to(model.device)
                outputs = model(**encodings)
                logodds = (
                    outputs.logits[:, -1, target_toks[1]]
                    - outputs.logits[:, -1, target_toks[0]]
                )
                results.append(logodds.cpu())
            logodds = torch.cat(results)
            logodds -= logodds.mean()
            soft_preds = torch.sigmoid(logodds)
            soft_preds = torch.stack([1 - soft_preds, soft_preds], dim=-1).tolist()

            for prompt in [
                "weak_amplified",
                "both_amplified",
                "neither_amplified",
                "gt_amplified",
            ]:

                def reformat(ex):
                    txt = {
                        "weak_amplified": f"Is \"{ex['sentence2']}\" consistent with \"{ex['sentence1']}\"?",
                        "both_amplified": f"(1) {ex['sentence2']}\n(2) {ex['sentence1']}\nSome say (1) and (2) are consistent. Some say they are paraphrases. What do you think?",
                        "neither_amplified": f"(1) {ex['sentence1']}\n(2) {ex['sentence2']}",
                        "gt_amplified": f"\"{ex['sentence2']}\" is a paraphrase of \"{ex['sentence1']}\".",
                    }[prompt]
                    return {"txt": txt}

                # add soft labels to dataset
                results_folder = str(
                    Path(__file__).parent / f"results/paws_consistency_{prompt}"
                )
                save_ds = ds.map(reformat).add_column("soft_pred", soft_preds)  # type: ignore

                # save to disk
                save_ds.save_to_disk(str(Path(results_folder) / f"weak_{split}"))


if __name__ == "__main__":
    Fire(main)
