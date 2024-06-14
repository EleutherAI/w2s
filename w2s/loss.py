import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union
from w2s.sft_utils import literal

from simple_parsing import Serializable


@dataclass
class LossConfig(Serializable):
    def to_dict(self):
        irrelevant_fields = []
        return {k: v for k, v in vars(self).items() if k not in irrelevant_fields}

@dataclass
class LogConfidenceLossConfig(LossConfig):
    logconf_weight: float = 0.5
    logconf_warmup_steps: int = 100
    balance_batch: bool = False

@dataclass
class ConfidenceWindowLossConfig(LossConfig):
    radius: Union[float, literal("midweak")] = 0.15

    def to_dict(self):
        return {"radius": self.radius if isinstance(self.radius, float) else "midweak"}

@dataclass
class LogEntropyLossConfig(LogConfidenceLossConfig):
    pass

@dataclass
class CrossEntropyLossConfig(LossConfig):
    pass

@dataclass
class KLDivergenceLossConfig(LossConfig):
    pass

LOSS_CONFIGS = {
    "logconf": LogConfidenceLossConfig, 
    "window": ConfidenceWindowLossConfig,
    "entropy": LogEntropyLossConfig,
    "xent": CrossEntropyLossConfig,
    "kl": KLDivergenceLossConfig,
}


def confidence_window_loss(
    logits,
    labels,
    radius: float = 0.15,
):
    """
    Use cross-entropy loss only for the examples where the model is uncertain.
    """
    logits = logits.float()
    labels = labels.float()

    preds = torch.softmax(logits, dim=-1)

    uncertain = (preds.max(dim=-1).values < 0.5 + radius)

    target = torch.stack([1.0 - labels, labels], dim=1)

    loss = torch.nn.functional.cross_entropy(
        logits[uncertain], 
        target[uncertain], 
        reduction="sum"
    )

    return loss / logits.shape[0]


def cross_entropy_loss(
    logits,
    labels,
):
    logits = logits.float()
    labels = labels.float()

    target = torch.stack([1.0 - labels, labels], dim=1)
    return torch.nn.functional.cross_entropy(logits, target)


def kl_divergence_loss(
    logits,
    labels,
):
    logits = logits.float()
    labels = labels.float()

    target = torch.stack([1.0 - labels, labels], dim=1)
    log_preds = torch.log_softmax(logits, dim=-1)

    return F.kl_div(log_preds, target, reduction="batchmean")


def log_confidence_loss(
    logits,
    labels,
    step: int,
    warmup_steps: int = 200,
    aux_coef: float = 0.5,
    balance_batch: bool = False,
    harden: bool = True,
    buffer: list = None,
    buffer_size: int = 32,
):
    """
    This is similar to the loss in Burns et al., except that it also optionally
    balances the labels by mean-subtracting in log-odds space.
    """
    logits = logits.float()
    labels = labels.float()
    if balance_batch:
        logodds_labels = torch.log(labels + 1e-7) - torch.log(1 - labels + 1e-7)
        labels = torch.sigmoid(logodds_labels - logodds_labels.mean())
        prior = 0.5
    else:
        prior = labels.mean() if labels.shape[0] > 1 else 0.5

    coef = aux_coef * min(1.0, step / warmup_steps) if warmup_steps > 0 else aux_coef
    preds = torch.softmax(logits, dim=-1)
    buffer += list(preds[:, 0].detach())
    buffer = buffer[-buffer_size:]

    if harden:
        threshold = torch.quantile(torch.stack(buffer), prior)
        target_preds = torch.cat(
            [(preds[:, 0] >= threshold)[:, None], (preds[:, 0] < threshold)[:, None]],
            dim=1,
        )
    else:
        target_preds = preds

    labels_binary = torch.stack([1.0 - labels, labels], dim=1)
    target = labels_binary * (1 - coef) + target_preds.detach() * coef
    return torch.nn.functional.cross_entropy(logits, target)