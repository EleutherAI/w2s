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
    # All pairwise distances, leaving out the diagonal
    dists = torch.cdist(x, x).fill_diagonal_(torch.inf)

    # Find indices of `k` nearest neighbors
    indices = dists.topk(k, largest=False).indices

    # Create kNN adjacency matrix
    adj = indices.new_zeros(len(x), len(x), dtype=torch.bool)
    adj.scatter_(1, indices, True)

    cls_mask = y[:, None] > 0.5
    pos_mask = lcc_mask(adj & cls_mask)
    neg_mask = lcc_mask(adj & ~cls_mask)
    return neg_mask | pos_mask


def topofilter(
    x: torch.Tensor, y: torch.Tensor, contamination: float = 0.1,
    *, k_cc: int = 5, k_zeta: int = 5,
):
    """Remove points whose labels are far the average of their neighbors' labels."""

    C = topo_cc(x, y, k=k_cc)
    x_C, y_C = x[C], y[C]

    # Zeta filtering
    dists = torch.cdist(x_C, x_C).fill_diagonal_(torch.inf)
    indices = dists.topk(k_zeta, largest=False).indices

    # Compute how far each point is from its average neighbor
    knn_labels = y_C[indices].float().mean(1)
    dists = torch.abs(y_C - knn_labels)

    # Remove points that are furthest from their average neighbor
    cc_removed = len(x) - len(x_C)
    remaining = round(len(x) * contamination) - cc_removed
    n = max(remaining, 0)

    if n == 0:
        print("TopoCC overshot contamination. No additional points removed.")
        print(f"frac removed = {cc_removed / len(x)}")

    filtered = dists.topk(n).indices.cpu()
    C_indices = C.nonzero().squeeze(1).cpu()
    return np.delete(C_indices, filtered)


def topolabel(
    x0: torch.Tensor, y0: torch.Tensor, x: torch.Tensor,
    *, k_cc: int = 5, k_zeta: int = 5,
):
    """Relabel points x to the average of their neighbors' labels within x0 CC."""

    C = topo_cc(x0, y0, k=k_cc)
    x_C, y_C = x0[C], y0[C]

    # Zeta filtering
    dists = torch.cdist(x, x_C)
    indices = dists.topk(k_zeta, largest=False).indices

    # Compute how far each point is from its average neighbor
    knn_labels = y_C[indices].float().mean(1)
    return knn_labels