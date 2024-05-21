import warnings
from dataclasses import dataclass
from typing import List, Optional

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaForSequenceClassification,
    MistralForSequenceClassification,
    Qwen2ForSequenceClassification,
)

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
DEFAULT_ARCHS = [
    LlamaForSequenceClassification,
    MistralForSequenceClassification,
    Qwen2ForSequenceClassification,
]


@dataclass
class ModelConfig:
    name: str
    enable_lora: bool
    lora_modules: Optional[List[str]] = None

    def to_dict(self):
        return vars(self)


def init_model_and_tokenizer(cfg: ModelConfig):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.name, torch_dtype="auto", device_map={"": "cuda"}
    )

    if cfg.lora_modules is None and cfg.enable_lora:
        cfg.lora_modules = DEFAULT_LORA_MODULES
        if not any(isinstance(model, arch) for arch in DEFAULT_ARCHS):
            warnings.warn(
                "Using default LORA modules for an architecture that is not Llama, Mistral, or Qwen"
            )

    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore
    model.score.weight.data *= 0.01
    model.config.problem_type = "single_label_classification"

    if cfg.enable_lora:
        lora_cfg = LoraConfig(target_modules=cfg.lora_modules)
        model = get_peft_model(model, lora_cfg)

    return model, tokenizer
