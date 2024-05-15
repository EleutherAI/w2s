import torch
import torch.nn.functional as F

def oracle_filter(train_labels: torch.Tensor, train_preds: torch.Tensor, q: float = 0.5):
    """Remove points whose labels are far the ground truth labels."""

    # Number of points to return
    n = round(q * len(train_labels))
    assert n > 0

    train_preds_vector = torch.stack([1 - train_preds, train_preds], dim=-1).to(train_labels.device)

    # Use the cross entropy of the train labels and preds to measure misclassification error
    dists = F.cross_entropy(train_preds_vector, train_labels, reduction='none')

    # Return points that are closest to their average neighbor
    return dists.topk(n, largest=False).indices

def rand_filter(train_preds: torch.Tensor, q: float = 0.5):
    """Remove a fraction q of labels randomly."""

    # Number of points to return
    n = round(q * len(train_preds))
    assert n > 0

    # Generate random indices
    indices = torch.randperm(len(train_preds))[:n]

    # Return the selected indices
    return indices