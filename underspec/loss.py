import torch


def log_confidence_loss(
    logits,
    labels,
    step: int,
    warmup_steps: int = 200,
    aux_coef: float = 0.5,
):
    logits = logits.float()
    labels = labels.float()

    coef = aux_coef * min(1.0, step / warmup_steps)
    preds = torch.softmax(logits, dim=-1)

    threshold = torch.quantile(preds[:, 0], labels.mean())
    strong_preds = torch.cat(
        [(preds[:, 0] >= threshold)[:, None], (preds[:, 0] < threshold)[:, None]],
        dim=1,
    )
    labels_binary = torch.stack([1.0 - labels, labels], dim=1)
    target = labels_binary * (1 - coef) + strong_preds.detach() * coef
    return torch.nn.functional.cross_entropy(logits, target)
