from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from w2s.utils import assert_type

# Works for Llama, Mistral, and Qwen architectures
DEFAULT_LORA_MODULES = [
    "gate_proj",
    "down_proj",
    "up_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


@dataclass
class PredictorConfig(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        ...


@dataclass
class ModelConfig(PredictorConfig):
    name: str
    enable_lora: bool
    lora_modules: Optional[List[str]] = None

    def to_dict(self):
        return vars(self)


class AutoCastingScore(torch.nn.Module):
    def __init__(
        self, score: torch.nn.Linear, output_dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        # make a leaf tensor with the same data as score
        self.weight = torch.nn.Parameter(score.weight.to(torch.float32).data)
        self.output_dtype = output_dtype

    def forward(self, hiddens):
        return torch.nn.functional.linear(
            hiddens.to(self.weight.dtype), self.weight, None
        ).to(self.output_dtype)


def init_tokenizer(cfg: ModelConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def init_model(tokenizer, cfg: ModelConfig):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.name, torch_dtype="auto", device_map={"": "cuda"},
        # force_download=True,
    )

    if cfg.lora_modules is None and cfg.enable_lora:
        cfg.lora_modules = MODEL_REGISTRY.get(cfg.name, {}).get(
            "lora_modules", DEFAULT_LORA_MODULES
        )

    model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore
    model.score.weight.data *= 0.01
    model.config.problem_type = "single_label_classification"

    if cfg.enable_lora:
        lora_cfg = LoraConfig(
            target_modules=cfg.lora_modules, task_type=TaskType.SEQ_CLS
        )

        # NOTE: adding task_type causes dtype errors, but is necessary for proper module saving
        # and for making the lm head trainable, so we need to wrap it in an AutoCastingScore
        for attr in ["score", "classifier"]:
            if hasattr(model, attr):
                setattr(
                    model,
                    attr,
                    AutoCastingScore(getattr(model, attr), output_dtype=model.dtype),
                )
                break
        else:
            raise ValueError("Could not find classifier head in model.")
        model = get_peft_model(model, lora_cfg)

    # put all the trainable (e.g. LoRA) parameters in float32
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    return model


def init_model_and_tokenizer(cfg: ModelConfig):
    tokenizer = init_tokenizer(cfg)
    model = init_model(tokenizer, cfg)

    return model, tokenizer


# TODO: make a legitimate model registry
# for now we just have a map from model name to learning rate and lora modules
MODEL_REGISTRY = {
    "meta-llama/Meta-Llama-3-8B": {
        "lr": 8e-5,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "mistralai/Mistral-7B-v0.1": {
        "lr": 8e-5,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "gemma/gemma-7b": {
        "lr": 8e-5,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
    "Qwen/Qwen1.5-0.5B": {
        "lr": 5e-4,
        "lora_modules": DEFAULT_LORA_MODULES,
    },
}