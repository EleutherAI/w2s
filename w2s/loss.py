import torch


def log_entropy_loss(
    logits,
    labels,
    step: int,
    warmup_steps: int = 40,
    aux_coef: float = 0.75,  # 0.75 -> loss(logit, label) is convex only for very confident labels
    balance_batch: bool = False,
):
    """
    This is similar to the loss in Burns et al., except that it uses an
    entropy term instead of a hardened quasi-entropy term and also optionally
    balances the labels by mean-subtracting in log-odds space.
    """
    logits = logits.float()
    labels = labels.float()
    if balance_batch:
        logodds_labels = torch.log(labels + 1e-7) - torch.log(1 - labels + 1e-7)
        labels = torch.sigmoid(logodds_labels - logodds_labels.mean())

    coef = aux_coef * min(1.0, step / warmup_steps) if warmup_steps > 0 else aux_coef
    preds = torch.softmax(logits, dim=-1)

    labels_binary = torch.stack([1.0 - labels, labels], dim=1)
    target = labels_binary * (1 - coef) + preds.detach() * coef
    return torch.nn.functional.cross_entropy(logits, target)


def log_confidence_loss(
    logits,
    labels,
    step: int,
    warmup_steps: int = 200,
    aux_coef: float = 0.5,
    balance_batch: bool = False,
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
        prior = labels.mean()

    coef = aux_coef * min(1.0, step / warmup_steps) if warmup_steps > 0 else aux_coef
    preds = torch.softmax(logits, dim=-1)

    threshold = torch.quantile(preds[:, 0], prior)
    strong_preds = torch.cat(
        [(preds[:, 0] >= threshold)[:, None], (preds[:, 0] < threshold)[:, None]],
        dim=1,
    )
    labels_binary = torch.stack([1.0 - labels, labels], dim=1)
    target = labels_binary * (1 - coef) + strong_preds.detach() * coef
    return torch.nn.functional.cross_entropy(logits, target)


def log_confidence_loss2(
    logits,
    labels,
    step: int,
    warmup_steps: int = 200,
    aux_coef: float = 0.5,
    balance_batch: bool = False,
):
    """
    This one uses a batch-independent threshold of 0.5, and then finally optionally balances
    the batch by mean-subtracting the log-odds of the target.
    """
    logits = logits.float()
    labels = labels.float()

    coef = aux_coef * min(1.0, step / warmup_steps) if warmup_steps > 0 else aux_coef
    preds = torch.softmax(logits, dim=-1)

    threshold = 0.5
    strong_preds = torch.cat(
        [(preds[:, 0] >= threshold)[:, None], (preds[:, 0] < threshold)[:, None]],
        dim=1,
    )
    labels_binary = torch.stack([1.0 - labels, labels], dim=1)
    target = labels_binary * (1 - coef) + strong_preds.detach() * coef

    if balance_batch:
        logodds_target = torch.log(target) - torch.log1p(-target)
        target = torch.sigmoid(
            logodds_target - logodds_target.mean(dim=0, keepdim=True)
        )

    return torch.nn.functional.cross_entropy(logits, target)
