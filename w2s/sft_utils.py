import gc
from pathlib import Path

import pynvml
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PretrainedConfig, Trainer

from w2s.roc_auc import roc_auc
from w2s.utils import assert_type


def compute_acc_and_auroc(eval_pred):
    predictions, labels = map(torch.from_numpy, eval_pred)

    hard_labels = (labels > 0.5).long()
    return dict(
        accuracy=predictions.argmax(dim=1).eq(hard_labels).float().mean(),
        auroc=roc_auc(hard_labels, predictions[:, 1]),
    )


@torch.no_grad()
def gather_hiddens(model: torch.nn.Module, dataset: Dataset):
    dataset = dataset.with_format("torch", device="cuda")

    cfg = assert_type(PretrainedConfig, model.config)
    D = assert_type(int, cfg.hidden_size)
    L = assert_type(int, cfg.num_hidden_layers)

    buffer = torch.empty(L, len(dataset), D, device=model.device, dtype=model.dtype)
    for i, ex in enumerate(tqdm(dataset)):
        ex = assert_type(dict, ex)

        out = model(ex["input_ids"][None], output_hidden_states=True)
        buffer[i] = torch.stack(out.hidden_states)[:, 0, -1]  # Final token

    return buffer


def move_best_ckpt(trainer: Trainer):
    checkpoints = list(Path(trainer.args.output_dir).glob("checkpoint-*"))
    if not checkpoints:
        print("No checkpoints found, saving final model")
        trainer.save_model(f"{trainer.args.output_dir}/best-ckpt")
        trainer._save_optimizer_and_scheduler(f"{trainer.args.output_dir}/best-ckpt")
        return

    if not trainer.args.load_best_model_at_end or not checkpoints:
        checkpoints = list(Path(trainer.args.output_dir).glob("checkpoint-*"))
        # get the largest checkpoint
        best_ckpt = max(checkpoints, key=lambda x: int(x.stem.split("-")[-1]))
        best_ckpt.rename(Path(trainer.args.output_dir) / "best-ckpt")
        return

    path = trainer.state.best_model_checkpoint
    perf = trainer.state.best_metric
    assert path is not None, "No best checkpoint found"
    assert perf is not None, "No best metric"

    src = Path(path)
    dest = src.parent / "best-ckpt"
    src.rename(dest)
    print(f"Best model (auroc {perf:.3f}) saved at: {dest}")


def clear_mem(verbose: bool = False):
    """
    This function is used to clear the memory allocated by PyTorch.
    It does so by calling the garbage collector to release unused GPU memory.
    After clearing the memory, it prints the current amount of memory still
    allocated by PyTorch (post-clean).

    Parameters:
    verbose (bool): Whether to print additional information.
    """

    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
    )

    if verbose:

        def try_attr(x, a):
            try:
                return getattr(x, a)
            except Exception:
                # amazing that this can cause...
                # (AttributeError, OSError, AssertionError, RuntimeError, ModuleNotFoundError)
                return None

        for obj in gc.get_objects():
            if torch.is_tensor(obj) or torch.is_tensor(try_attr(obj, "data")):
                print(type(obj), obj.size(), obj.dtype)


def get_gpu_mem_used() -> float:
    """returns proportion of used GPU memory averaged across all GPUs"""
    prop_sum = 0
    pynvml.nvmlInit()
    try:
        num_devices = pynvml.nvmlDeviceGetCount()
        for i in range(num_devices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            prop_sum += int(meminfo.used) / int(meminfo.total)
    finally:
        pynvml.nvmlShutdown()
    return prop_sum / num_devices
