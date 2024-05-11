import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)

from .utils import assert_type


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


def knn_average(x: torch.Tensor, y: torch.Tensor, k: int):
    """Compute average of `y` of `k` nearest neighbors of `x`."""

    # Find indices of `k` nearest neighbors
    indices = torch.cdist(x, x).topk(k, largest=False).indices

    # Compute average of `y` of `k` nearest neighbors
    return y[indices].mean(1)


def zeta_filter(x: torch.Tensor, y: torch.Tensor, k: int, q: float = 0.5):
    """Compute average of `y` of `k` nearest neighbors of `x`."""

    # Number of points to return
    n = round(q * len(x))
    assert k > 0 and n > 0

    # Find indices of `k` nearest neighbors
    dists = torch.cdist(x, x).fill_diagonal_(torch.inf)
    indices = dists.topk(k, largest=False).indices

    # Compute how far each point is from its average neighbor
    knn_labels = y[indices].mean(1)
    dists = torch.abs(y - knn_labels)

    # Return points that are closest to their average neighbor
    return dists.topk(n, largest=False).indices
