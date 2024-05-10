import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)

from .utils import assert_type


@torch.inference_mode()
def gather_hiddens(model: PreTrainedModel, dataset: Dataset):
    dataset = dataset.with_format("torch", device="cuda")
    model.to("cuda")

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
