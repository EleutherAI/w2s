from typing import Literal, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from w2s.utils import assert_type


def get_reproducible_generator():
    return torch.Generator(device="cuda").manual_seed(0)


def get_kernel_grads(
    model,
    dataset,
    d_jacobian: int = 1000,
    d_downsample: Union[int, Literal["sqrt"]] = "sqrt",
):
    dataset = dataset.with_format("torch", device="cuda")

    model_n_params = sum(p.numel() for p in model.parameters())
    if d_downsample == "sqrt":
        # e.g. 7B -> 16.7m, 410m -> 4m
        d_downsample = int(200 * model_n_params**0.5)

    # Compute Jacobians
    jac_idxs = torch.randint(0, len(dataset), (d_jacobian,), device=model.device)
    jac_dataset = dataset.select(jac_idxs)

    jac_buffer = torch.empty(
        d_jacobian, d_downsample, device=model.device, dtype=model.dtype
    )
    for i, ex in enumerate(tqdm(jac_dataset, desc="Computing Jacobians")):
        ex = assert_type(dict, ex)

        logits: torch.Tensor = model(
            ex["input_ids"][None], output_hidden_states=False
        ).logits
        logodds = logits.diff(dim=1).squeeze()
        logodds.backward()

        jac_buffer[i] = gather_grad_components(
            model,
            d_downsample,
            get_reproducible_generator(),
            io_device=model.device,
        )
        model.zero_grad()

    # Compute regular grads
    jvp = torch.empty(len(dataset), d_jacobian, device=model.device, dtype=model.dtype)
    for i, ex in enumerate(tqdm(dataset, desc="Computing grads")):
        ex = assert_type(dict, ex)

        loss = model(ex["input_ids"][None], labels=ex["labels"][None]).loss
        loss.backward()

        # TODO: Adam?
        jvp[i] = (
            gather_grad_components(
                model,
                d_downsample,
                get_reproducible_generator(),
                io_device=model.device,
            )
            @ jac_buffer.T
        )
        model.zero_grad()

    return jvp


def gather_grad_components(
    model, d_downsample, generator, io_device: str | int = "cpu", optimizer=None
):
    """
    This avoids concatenating all the grads
    into one tensor before projecting, to save memory.
    This assumes `model` parameters has gradients already computed.

    If optimizer passed is Adam, then we also normalize the gradients by the
    second moment estimate per Adam's update rule.
    """
    proj_updates = []
    model_n_params = sum(p.numel() for p in model.parameters() if p.grad is not None)
    keep_prob = d_downsample / model_n_params

    for param in model.parameters():
        if param.grad is None:
            continue

        # NOTE: this produces indices that are not unique for the benefit of speed
        # it's around ~1.8x faster for d_downsample=4_300_000 (out of 500_000_000)
        # as compared to using a while loop to sample new indices until full
        # (randperm is much slower still)
        n_keep = int(np.ceil(keep_prob * param.numel()))
        indices = torch.randint(
            0, param.numel(), (n_keep,), generator=generator, device=io_device
        )

        update = param.grad.flatten()[indices].to(io_device)

        if isinstance(optimizer, torch.optim.Adam):
            step = optimizer.state[param].get("step", 0)
            if step > 0:
                # normalize based on raw second moment estimates
                beta2 = float(optimizer.param_groups[0]["betas"][1])
                exp_avg_sq = optimizer.state[param]["exp_avg_sq"]
                exp_avg_sq = exp_avg_sq.flatten()[indices].to(io_device)
                corrected_exp_avg = torch.sqrt(exp_avg_sq / (1 - beta2**step))
            else:
                corrected_exp_avg = update.abs()

            eps = float(optimizer.param_groups[0]["eps"])
            update = update / (corrected_exp_avg + eps)

        proj_updates.append(update)

    proj_updates = torch.cat(proj_updates)
    # We have a few more than d_downsample updates, so we pick some to drop
    # NOTE: For speed reasons (~1.5x compared to randperm) we choose to do the
    # less diverse thing:
    # We keep only the last d_downsample updates
    # We keep the later gradients because they tend to be larger in magnitude
    return proj_updates[-d_downsample:]
