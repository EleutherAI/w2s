import gc
from pathlib import Path

import pynvml
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PretrainedConfig, Trainer

from w2s.utils import assert_type


# simple_parsing doesn't like typing.Literal (pre-3.12) so I rolled my own
# note: parens, not brackets

# Python 3.11 version:
# literal = lambda *args: StrEnum("option", args)

# Python 3.10 version:
def ident_escape_char(c: str) -> str:
    if c.isalnum() or c == "_":
        return c
    return f"_{ord(c)}_"

def ident_escape(s: str) -> str:
    return "".join(ident_escape_char(c) for c in s)

def literal(s: str):
    return type('LiteralString_' + ident_escape(s), (LiteralString,), {"value": s})

class LiteralString():
    value = ""

    def __init__(self, value):
        if value != self.value:
            raise ValueError(f"Invalid value {value!r} is not literally {self.value!r}")

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return self.value == other


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
    if trainer.args.load_best_model_at_end:
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
