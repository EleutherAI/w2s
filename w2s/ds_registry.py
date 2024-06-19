import functools
import hashlib
from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Literal, Union

from datasets import Dataset as HfDataset
from datasets import DatasetDict as HfDatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset as hf_load_dataset


@dataclass
class DatasetConfig:
    # split -> unshuffled dataset of items
    loader: Callable[[str], HfDataset]
    # formats items to have keys 'txt' and 'hard_label', takes a random.Random rng
    # (or for generative tasks, 'ctx' and 'target', and no 'hard_label' key)
    # deprecated OAI legacy:
    # optionally also adds the key 'choices', a pair of strings, indicating to use the
    # lm head
    formatter: Callable[[Any], Any]
    # "classify" or "generate"
    task: str = "classify"


# mapping from dataset name to load function and format function
_REGISTRY: dict[str, DatasetConfig] = {}


def register_dataset(name: str, config: DatasetConfig):
    _REGISTRY[name] = config


def balance(ds: HfDataset, seed: int):
    """Undersample balance to 50/50"""

    label_counts = Counter(ds["hard_label"])
    assert len(label_counts) == 2, f"Dataset must be binary {label_counts}"

    # undersample the majority class
    majority_label = max(label_counts, key=lambda k: label_counts[k])
    minority_label = 1 - majority_label
    minority_count = label_counts[minority_label]
    minority_ds = ds.filter(lambda ex: ex["hard_label"] == minority_label)
    majority_ds = (
        ds.filter(lambda ex: ex["hard_label"] == majority_label)
        .shuffle(seed=seed)
        .select(range(minority_count))
    )
    return concatenate_datasets([minority_ds, majority_ds]).shuffle(seed=seed)


def load_and_process_train_test(
    ds_name: str,
    split_sizes: dict[str, int],
    seed: int = 0,
    take_test_from_train: bool = False,
):
    n_test = split_sizes.get("test", 0)
    if take_test_from_train:
        # in this case we gather excess documents from the train set, and
        # at the end redistribute them to the test set
        split_sizes["train"] += n_test
        del split_sizes["test"]

    if ds_name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {ds_name}, please register")
    cfg = _REGISTRY[ds_name]
    results = {}
    for split, n_docs in split_sizes.items():
        ds = cfg.loader(split).shuffle(seed=seed)
        ds = ds.map(functools.partial(cfg.formatter, rng=Random(seed)))  # type: ignore

        if cfg.task == "generate":
            ds = ds.filter(lambda ex: ex["ctx"] != "")  # remove empty texts
            ds = ds.filter(lambda ex: ex["target"] != "")
        else:
            ds = ds.filter(lambda ex: ex["txt"] != "")  # remove empty texts
            ds = balance(ds, seed)  # balance to 50/50

        try:
            ds = ds.select(range(n_docs))
        except IndexError:
            print(f"{ds_name} has < {n_docs} docs after balancing, using all {len(ds)}")

        if cfg.task == "generate":
            ds = ds.map(
                lambda ex: {
                    "id": hashlib.sha1(ex["ctx"].encode()).hexdigest()[:8],
                }
            )
        else:
            ds = ds.map(
                lambda ex: {
                    "id": hashlib.sha1(ex["txt"].encode()).hexdigest()[:8],
                    "soft_label": [
                        1 - float(ex["hard_label"]),
                        float(ex["hard_label"]),
                    ],
                }
            )
        results[split] = ds

    if take_test_from_train:
        # take the first n_test examples from the training set as the test set
        results["test"] = results["train"].select(range(n_test))
        results["train"] = results["train"].select(range(n_test, len(results["train"])))
    return results 


def load_and_process_dataset(
    ds_name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    n_predict: Union[Literal["train"], int],
    take_test_from_train: bool = False,
    seed=0,
) -> HfDatasetDict:
    """
    Returns a dict with keys 'train', 'val', 'test', and optionally 'predict', and dataset values
    Examples in 'test' split can never appear in 'train', 'val', or 'predict' on any run.
    """
    split_sizes = dict(train=n_train + n_val, test=n_test)
    if n_predict != "train":
        assert n_predict >= 0
        split_sizes["train"] += n_predict

    results = load_and_process_train_test(
        ds_name, split_sizes, seed, take_test_from_train
    )

    splits = dict(
        val=results["train"].select(range(n_val)),
        train=results["train"].select(range(n_val, len(results["train"]))),
        test=results["test"],
    )
    if n_predict == "train":
        # simply use the training set for predictions
        splits["predict"] = splits["train"]
    elif n_predict > 0:
        # take the requested *fraction* of examples from the training set
        subsplits = splits["train"].train_test_split(
            test_size=n_predict / (n_train + n_predict)
        )
        splits["train"], splits["predict"] = subsplits["train"], subsplits["test"]

    return HfDatasetDict(splits)


warned_about_choices = set()


def encode_choice(text, tokenizer):
    global warned_about_choices

    c_ids = tokenizer.encode(text, add_special_tokens=False)

    # some tokenizers split off the leading whitespace character
    if tokenizer.decode(c_ids[0]).strip() == "":
        c_ids = c_ids[1:]
        assert c_ids == tokenizer.encode(text.lstrip(), add_special_tokens=False)

    c_ids = tuple(c_ids)
    if len(c_ids) != 1 and c_ids not in warned_about_choices:
        assert c_ids[0] not in [
            c[0] for c in warned_about_choices
        ], "Choice shares first token with another choice"
        warned_about_choices.add(c_ids)
        print(
            f'Warning: Only the first token of multitoken choice "{text}" will be used'
        )
    return c_ids[0]


def hf_loader(*hf_name, split_names=None, n_test=None):
    """
    If `split_names` is provided, it maps from the requested
    split name to the actual name in the hugginface dataset.
    If `n_test` is provided, it will concatenate all splits together
    and then take a deterministic test set of size `n_test` from it.
    """

    # this thunk avoids loading datasets at import time
    def thunk(split):
        nonlocal split_names
        if n_test is not None:
            assert split_names is None
            ds = hf_load_dataset(*hf_name)
            if isinstance(ds, HfDatasetDict):
                ds = concatenate_datasets(ds.values())  # type: ignore
            assert isinstance(ds, HfDataset)
            # the seed is fixed so that all runs use the same test pool
            splits = ds.train_test_split(test_size=n_test, seed=0)

            return splits[split]

        if split_names is None:
            split_names = dict()

        return hf_load_dataset(*hf_name, split=split_names.get(split, split))

    return thunk


##########
# ACTUAL DATASETS
##########


def format_anli(ex, rng):
    txt = (
        f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n\nDoes the premise"
        " entail the hypothesis?"
    )
    return dict(txt=txt, hard_label=ex["label"] == 0)


register_dataset(
    "anli-r2",
    DatasetConfig(
        loader=hf_loader(
            "facebook/anli", split_names=dict(train="train_r2", test="test_r2")
        ),  # type: ignore
        formatter=format_anli,  # type: ignore
    ),
)


def format_cola(ex, rng):
    return dict(txt=ex["sentence"], hard_label=ex["label"])


register_dataset(
    "cola",
    DatasetConfig(
        loader=hf_loader(
            "nyu-mll/glue", "cola", split_names=dict(test="validation")
        ),  # type: ignore
        formatter=format_cola,  # type: ignore
    ),
)


def format_dream(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["answer"]
    else:
        distractors = ex["choice"].copy()
        distractors.remove(ex["answer"])

        ans = rng.choice(distractors)

    joined = "\n".join(ex["dialogue"])
    txt = f"{joined}\n\nQ: {ex['question']} A: {ans}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "dream",
    DatasetConfig(
        loader=hf_loader("dream"),  # type: ignore
        formatter=format_dream,  # type: ignore
    ),
)


def format_ethics_deontology(ex, rng):
    "yes" if ex["label"] else "no"
    txt = (
        "Q: Is the following a morally valid"
        f" excuse?\n\n{ex['scenario']}\n\n{ex['excuse']}\n\nA:"
    )
    return dict(txt=txt, hard_label=ex["label"])


register_dataset(
    "ethics-deontology",
    DatasetConfig(
        loader=hf_loader("hendrycks/ethics", "deontology"),  # type: ignore
        formatter=format_ethics_deontology,  # type: ignore
    ),
)


def format_ethics_justice(ex, rng):
    txt = f"Q: Does this statement exemplify justice?\n\n{ex['scenario']}\n\nA:"
    return dict(txt=txt, hard_label=ex["label"])


register_dataset(
    "ethics-justice",
    DatasetConfig(
        loader=hf_loader("hendrycks/ethics", "justice"),  # type: ignore
        formatter=format_ethics_justice,  # type: ignore
    ),
)


def format_ethics_virtue(ex, rng):
    txt = (
        "Q: Does this behavior match the adjective that"
        f" follows?\n\n{ex['scenario']}\n\nA:"
    )
    return dict(txt=txt, hard_label=ex["label"])


register_dataset(
    "ethics-virtue",
    DatasetConfig(
        loader=hf_loader("hendrycks/ethics", "virtue"),  # type: ignore
        formatter=format_ethics_virtue,  # type: ignore
    ),
)


def format_ethics_utilitarianism(ex, rng):
    hard_label = int(rng.random() < 0.5)

    choices = [ex["baseline"], ex["less_pleasant"]]
    rng.shuffle(choices)

    correct = choices.index(ex["baseline"])
    response = correct if hard_label else 1 - correct

    txt = f"Which is more pleasant?\n1) {choices[0]}\n2) {choices[1]} A: {response + 1}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "ethics-utilitarianism",
    DatasetConfig(
        loader=hf_loader("hendrycks/ethics", "utilitarianism"),  # type: ignore
        formatter=format_ethics_utilitarianism,  # type: ignore
    ),
)


def format_mc_taco(ex, rng):
    template = "{sentence}\n\nGiven the above, {question} Is the answer {answer}?"
    return dict(txt=template.format(**ex), hard_label=ex["label"])


register_dataset(
    "mc_taco",
    DatasetConfig(  # we switch train and test bc test is bigger
        loader=hf_loader(  # type: ignore
            "mc_taco", split_names=dict(train="test", test="validation")
        ),
        formatter=format_mc_taco,  # type: ignore
    ),
)


def format_hellaswag(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["endings"][int(ex["label"])]
    else:
        ans = rng.choice(
            [e for i, e in enumerate(ex["endings"]) if i != int(ex["label"])]
        )

    endings = "\n".join(ex["endings"])
    txt = (
        f'Context:\n{ex["ctx"]}\n\nContinuations:\n\n{endings}\n\nQ: Is "{ans}" the'
        " best continuation?"
    )
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "hellaswag",
    DatasetConfig(
        loader=hf_loader(
            "Rowan/hellaswag", split_names=dict(test="validation")
        ),  # type: ignore
        formatter=format_hellaswag,  # type: ignore
    ),
)


def format_multirc(ex, rng):
    template = 'Passage:\n\n{paragraph}\n\nQ: "{question}" Is the answer "{answer}"?'

    txt = template.format(**ex)
    return dict(txt=txt, hard_label=ex["label"])


register_dataset(
    "multirc",
    DatasetConfig(
        loader=hf_loader(
            "super_glue", "multirc", split_names=dict(test="validation")
        ),  # type: ignore
        formatter=format_multirc,  # type: ignore
    ),
)


def format_openbookqa(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["answerKey"]
    else:
        letters = ex["choices"]["label"]

        distractors = ex["choices"]["text"].copy()
        del distractors[letters.index(ex["answerKey"])]
        ans = rng.choice(distractors)

    choices = [
        f"{a}) {t}" for a, t in zip(ex["choices"]["label"], ex["choices"]["text"])
    ]
    joined = "\n".join(choices)
    txt = f"Q: {ex['question_stem']}\n\nChoices:\n{joined}\n\nAnswer: {ans}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "openbookqa",
    DatasetConfig(
        loader=hf_loader("allenai/openbookqa"),  # type: ignore
        formatter=format_openbookqa,  # type: ignore
    ),
)


def format_paws(ex, rng):
    template = (
        "Sent 1: {sentence1}\nSent 2: {sentence2}\n\nQ: Are these sentences"
        " semantically equivalent?"
    )
    return dict(txt=template.format(**ex), hard_label=ex["label"])


register_dataset(
    "paws",
    DatasetConfig(
        loader=hf_loader("paws", "labeled_final"),  # type: ignore
        formatter=format_paws,  # type: ignore
    ),
)


def format_piqa(ex, rng):
    hard_label = int(rng.random() < 0.5)

    if hard_label:
        ans = ex["sol2"] if ex["label"] else ex["sol1"]
    else:
        ans = ex["sol1"] if ex["label"] else ex["sol2"]

    txt = f"Q: {ex['goal']} A: {ans}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "piqa",
    DatasetConfig(
        loader=hf_loader("piqa", split_names=dict(test="validation")),  # type: ignore
        formatter=format_piqa,  # type: ignore
    ),
)


def format_quail(ex, rng):
    template = 'Passage:\n\n{context}\n\nQ: "{question}" Is the answer "{answer}"?'
    hard_label = int(rng.random() < 0.5)

    correct_id = ex["correct_answer_id"]
    if hard_label:
        ans = ex["answers"][correct_id]
    else:
        ans = rng.choice([a for i, a in enumerate(ex["answers"]) if i != correct_id])

    txt = template.format(**ex, answer=ans)
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "quail",
    DatasetConfig(
        loader=hf_loader("quail", split_names=dict(test="validation")),  # type: ignore
        formatter=format_quail,  # type: ignore
    ),
)


def format_quartz(ex, rng):
    template = 'Passage:\n{para}\n\nQ: "{question}" Is the answer "{answer}"?'
    hard_label = int(rng.random() < 0.5)

    correct_id = ex["choices"]["label"].index(ex["answerKey"])
    ans = ex["choices"]["text"][correct_id if hard_label else 1 - correct_id]

    txt = template.format(**ex, answer=ans)
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "quartz",
    DatasetConfig(
        loader=hf_loader("allenai/quartz"),  # type: ignore
        formatter=format_quartz,  # type: ignore
    ),
)


def format_social_i_qa(ex, rng):
    template = (
        "Context:\n{context}\n\nQuestion:"
        ' "{question}"\n\nChoices:\n{answerA}\n{answerB}\n{answerC}\n\nIs the answer'
        ' "{answer}"?'
    )
    hard_label = int(rng.random() < 0.5)

    answers = [ex["answerA"], ex["answerB"], ex["answerC"]]
    correct_id = int(ex["label"]) - 1
    if hard_label:
        ans = answers[correct_id]
    else:
        ans = rng.choice([a for i, a in enumerate(answers) if i != correct_id])

    txt = template.format(**ex, answer=ans)
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "social_i_qa",
    DatasetConfig(
        loader=hf_loader(
            "social_i_qa", split_names=dict(test="validation")
        ),  # type: ignore
        formatter=format_social_i_qa,  # type: ignore
    ),
)


def format_sst2(ex, rng):
    return dict(txt=ex["sentence"], hard_label=ex["label"])


register_dataset(
    "sst2",
    DatasetConfig(
        loader=hf_loader(
            "stanfordnlp/sst2", split_names=dict(test="validation")
        ),  # type: ignore
        formatter=format_sst2,  # type: ignore
    ),
)


def format_wic(ex, rng):
    template = (
        'Sentence 1:\n{sentence1}\n\nSentence 2:\n{sentence2}\n\nQ: Does "{word}" have'
        " the same meaning in the above sentences?"
    )
    return dict(txt=template.format(**ex), hard_label=ex["label"])


register_dataset(
    "wic",
    DatasetConfig(
        loader=hf_loader(
            "super_glue", "wic", split_names=dict(test="validation")
        ),  # type: ignore
        formatter=format_wic,  # type: ignore
    ),
)


def format_twitter_sentiment(ex, rng):
    return dict(txt=ex["text"], hard_label=ex["label"])


register_dataset(
    "twitter-sentiment",
    DatasetConfig(
        loader=hf_loader("EleutherAI/twitter-sentiment"),  # type: ignore
        formatter=format_twitter_sentiment,  # type: ignore
    ),
)


SCIQ_N_TEST = 3000


def format_sciq(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])

    txt = f"Q: {ex['question']} A: {ans}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "sciq",
    DatasetConfig(
        loader=hf_loader("sciq", n_test=SCIQ_N_TEST),  # type: ignore
        formatter=format_sciq,  # type: ignore
    ),
)


def format_sciq_with_support(ex, rng):
    # from https://github.com/EleutherAI/elk-generalization
    template = (
        'Name: Bob\n\nPassage 1:\n{support}\n\nQ1: "{question}" Is the answer'
        ' "{answer}"?'
    )
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    txt = template.format(support=ex["support"], question=ex["question"], answer=ans)
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "sciq_with_support",
    DatasetConfig(
        loader=hf_loader("sciq", n_test=SCIQ_N_TEST),  # type: ignore
        formatter=format_sciq_with_support,  # type: ignore
    ),
)


def format_anthropic_hh(ex, rng):
    hard_label = int(rng.random() < 0.5)
    txt = ex["chosen"] if hard_label else ex["rejected"]
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "anthropic_hh",
    DatasetConfig(
        loader=hf_loader("Anthropic/hh-rlhf"),  # type: ignore
        formatter=format_anthropic_hh,  # type: ignore
    ),
)


def format_cosmosqa(ex, rng):
    true_answer = ex["answer" + str(ex["label"])]
    if "None of the above choices ." in true_answer:
        hard_label = 0
    else:
        assert "None of the above choices" not in true_answer, true_answer
        hard_label = int(rng.random() < 0.5)
    if hard_label:
        answer = true_answer
    else:
        candidate_answers = [ex["answer" + str(i)] for i in range(4)]
        answer = rng.choice([x for x in candidate_answers if x != true_answer])
    txt = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "cosmos_qa",
    DatasetConfig(
        loader=hf_loader(
            "cosmos_qa", split_names=dict(test="validation")
        ),  # type: ignore
        formatter=format_cosmosqa,  # type: ignore
    ),
)


def format_boolq(ex, rng):
    hard_label = int(ex["answer"])
    txt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "boolq",
    DatasetConfig(
        loader=hf_loader("boolq", split_names=dict(test="validation")),  # type: ignore
        formatter=format_boolq,  # type: ignore
    ),
)


def format_amazon_polarity(ex, rng):
    return dict(txt=f"{ex['title']} {ex['content']}", hard_label=ex["label"])


register_dataset(
    "amazon_polarity",
    DatasetConfig(
        loader=hf_loader("amazon_polarity"),  # type: ignore
        formatter=format_amazon_polarity,  # type: ignore
    ),
)


def format_underspecified_amazon_polarity(ex, rng, use_gt=True):
    txt = f"{ex['title']} {ex['content']}\n\nDoes this review say the product was \"good\" or \"great\"?"  # noqa
    words = {"good", "great"}
    label = (
        ex["label"]
        if use_gt
        else any(w in ex["content"].lower() or w in ex["title"].lower() for w in words)
    )
    return dict(txt=txt, hard_label=label)


# register_dataset(
#     "amazon_polarity_gt",
#     DatasetConfig(
#         loader=hf_loader("amazon_polarity"),  # type: ignore
#         formatter=functools.partial(format_underspecified_amazon_polarity, use_gt=True),  # type: ignore  # noqa
#     ),
# )

# register_dataset(
#     "amazon_polarity_weak",
#     DatasetConfig(
#         loader=hf_loader("amazon_polarity"),  # type: ignore
#         formatter=functools.partial(format_underspecified_amazon_polarity, use_gt=False),  # type: ignore  # noqa
#     ),
# )


VALID_DATASETS: list[str] = list(_REGISTRY.keys())


"""
from datasets import disable_caching
disable_caching()

from w2s.datasets import load_and_process_dataset, VALID_DATASETS
import numpy as np

ds_name = "boolq"
print(VALID_DATASETS)

ds = load_and_process_dataset(ds_name, split_sizes=dict(train=500, test=10))
train = list(ds['train'])
test = list(ds['test'])
print(test[0])
print(np.mean([x['hard_label'] for x in train]))
"""
