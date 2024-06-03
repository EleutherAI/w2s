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


def init_model_and_tokenizer(cfg: ModelConfig):
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.name, torch_dtype="auto", device_map={"": "cuda"}
    )

    if cfg.lora_modules is None and cfg.enable_lora:
        cfg.lora_modules = MODEL_REGISTRY.get(cfg.name, {}).get(
            "lora_modules", DEFAULT_LORA_MODULES
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore
    model.score.weight.data *= 0.01
    model.config.problem_type = "single_label_classification"

    if cfg.enable_lora:
        lora_cfg = LoraConfig(
            target_modules=cfg.lora_modules, task_type=TaskType.SEQ_CLS  # type: ignore
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

    return model, tokenizer


class Predictor(torch.nn.Module, ABC):
    """
    The strong "predictor", using the terminology of the original ELK report
    https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/edit#heading=h.kkaua0hwmp1d
    this is the model we would like to elicit latent knowledge from using the reporter
    """

    cfg: PredictorConfig

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg

    def __call__(
        self, inputs, output_hidden_states=False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor]]]:
        """
        This takes in a batch of inputs and returns the logodds of the model's predictions.
        If output_hidden_states is True, it also returns the hidden states of the model (second)
        Each of the `num_layers` hiddens are tensors of shape [n, hidden_size]
        """
        ...

    def to_dict(self) -> dict[str, str | int | float]:
        """A summary of the method that approximately uniquely identifies it.
        It should include a name and all the important hyperparameters."""
        ...


class TransformerPredictor(Predictor):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.transformer, self.tokenizer = init_model_and_tokenizer(cfg)

    def __call__(self, inputs, output_hidden_states=False):
        # inputs are text strings
        assert isinstance(inputs, list)
        # ...ModelForSequenceClassification makes sure to score hiddens
        # from the last non-padding token position
        input_ids = assert_type(
            torch.Tensor,
            self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")[
                "input_ids"
            ],
        ).to(self.transformer.device)

        outputs = self.transformer(input_ids, output_hidden_states=True)

        # gather hiddens at the last non-padding token position
        hiddens = torch.stack(
            outputs.hidden_states
        )  # [num_layers, n, seq_len, hidden_size]
        seq_lens = input_ids.ne(assert_type(int, self.tokenizer.pad_token_id)).sum(
            dim=-1
        )
        last_non_pad_idx = seq_lens - 1
        last_hidden_states = hiddens[:, torch.arange(len(inputs)), last_non_pad_idx, :]

        logodds = outputs.logits[:, 1] - outputs.logits[:, 0]
        return (
            (logodds, last_hidden_states.unbind(0)) if output_hidden_states else logodds
        )

    def to_dict(self):
        return self.cfg.to_dict()


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
