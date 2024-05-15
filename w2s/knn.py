import numpy as np
import torch
from datasets import Dataset
from scipy.sparse.csgraph import connected_components
from tqdm.auto import tqdm
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)

from .utils import assert_type


def lcc_mask(adj: torch.Tensor):
    """Mask for membership in the largest connected component"""
    num_cmps, cmps = connected_components(adj.cpu(), connection="strong")
    cmp_sizes = np.bincount(cmps, minlength=num_cmps)
    return torch.from_numpy(cmps == cmp_sizes.argmax()).to(adj.device)


def topo_cc(x: torch.Tensor, y: torch.Tensor, *, k: int = 5):
    """TopoCC label filtering algorithm."""
    if k == -1:
        # Use all points
        return torch.ones(len(x), dtype=torch.bool)

    # All pairwise distances, leaving out the diagonal
    dists = torch.cdist(x, x).fill_diagonal_(torch.inf)

    if dists.shape[0] < k:
        print(f"Warning: Not enough points for k={k} in TopoCC, using all {dists.shape[0]}.")
        k = dists.shape[0]

    # Find indices of `k` nearest neighbors
    indices = dists.topk(k, largest=False).indices

    # Create kNN adjacency matrix
    adj = indices.new_zeros(len(x), len(x), dtype=torch.bool)
    adj.scatter_(1, indices, True)

    cls_mask = y[:, None] > 0.5
    pos_mask = lcc_mask(adj & cls_mask)
    neg_mask = lcc_mask(adj & ~cls_mask)
    return neg_mask | pos_mask


def topo_relabel(
    x: torch.Tensor, y: torch.Tensor, kcc, kzeta
):
    """Remove points whose labels are far the average of their neighbors' labels."""
 
    C = topo_cc(x, y, k=kcc)
    x_C, y_C = x[C], y[C]

    # Zeta filtering
    dists = torch.cdist(x_C, x_C).fill_diagonal_(torch.inf)
    if dists.shape[0] < kzeta:
        print(f"Warning: Not enough points for k={k} in ZetaFilter, using all {dists.shape[0]} from CC.")
        kzeta = dists.shape[0]

    indices = dists.topk(kzeta, largest=False).indices

    # Compute average neighbor
    knn_labels = y_C[indices].float().mean(1)

    return knn_labels


def zeta_relabel(
    x: torch.Tensor, y: torch.Tensor, kzeta
):
    return topo_relabel(x, y, kcc=-1, kzeta=kzeta)


def topofilter(
    x: torch.Tensor, y: torch.Tensor, contamination: float = 0.1, *, kcc: int = None, k: int = 5
):
    if kcc is None:
        kcc = k

    knn_labels = topo_relabel(x, y, kcc=kcc, kzeta=k)
    dists = torch.abs(y_C - knn_labels)

    # Remove points that are furthest from their average neighbor
    cc_removed = len(x) - len(x_C)
    remaining = round(len(x) * contamination) - cc_removed
    n = max(remaining, 0)

    filtered = dists.topk(n).indices.cpu()
    C_indices = C.nonzero().squeeze(1).cpu()
    return np.delete(C_indices, filtered)


@torch.no_grad()
def gather_hiddens(model: PreTrainedModel, dataset: Dataset):
    dataset = dataset.with_format("torch", device="cuda")

    cfg = assert_type(PretrainedConfig, model.config)
    D = assert_type(int, cfg.hidden_size)
    L = assert_type(int, cfg.num_hidden_layers)

    buffer = torch.empty(len(dataset), D, device=model.device, dtype=model.dtype)
    for i, ex in enumerate(tqdm(dataset)):
        ex = assert_type(dict, ex)

        out = model(ex["input_ids"][None], output_hidden_states=True)
        buffer[i] = out.hidden_states[L // 2][0, -1]  # Final token

    return buffer


def cummean(x):
    """Compute cumulative mean of `x` across the last dimension."""
    return x.cumsum(-1) / torch.arange(
        1, x.shape[-1] + 1, device=x.device, dtype=x.dtype
    )


def zeta_filter(x: torch.Tensor, y: torch.Tensor, *, k: int = 0, q: float = 0.5):
    """Remove points whose labels are far the average of their neighbors' labels."""

    # Number of points to return
    n = round(q * len(x))
    assert n > 0

    # All pairwise distances, leaving out the diagonal
    dists = torch.cdist(x, x).fill_diagonal_(torch.inf)

    # Choose k automatically with LOOCV if needed
    if k < 1:
        # Compute a prediction for each point and each value of k
        preds = cummean(y[dists.argsort()][:, :-1])  # [n, n - 1]

        # Brier score loss for each value of k
        losses = torch.square(preds - y[:, None]).mean(0)

        # Choose the best one
        k = int(losses.argmin()) + 1
        print(f"Chose k = {k} with LOOCV")

    # Find indices of `k` nearest neighbors
    indices = dists.topk(k, largest=False).indices

    # Compute how far each point is from its average neighbor
    knn_labels = y[indices].mean(1)
    dists = torch.abs(y - knn_labels)

    # Return points that are closest to their average neighbor
    return dists.topk(n, largest=False).indices
