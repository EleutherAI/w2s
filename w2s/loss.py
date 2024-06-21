from typing import Optional

import torch
from einops import rearrange
from transformers import Trainer


class CustomLossTrainer(Trainer):
    def __init__(
        self,
        loss_name: str,
        *args,
        resume_from_optimizer_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.resume_from_optimizer_checkpoint = resume_from_optimizer_checkpoint

    def compute_loss(self, model, inputs, return_outputs=False):
        if (
            self.state.global_step == 0
            and self.resume_from_optimizer_checkpoint is not None
            and self.optimizer is not None
        ):
            # check if adam exp buffer is empty, and then load the optimizer state if it is
            if not isinstance(self.optimizer, torch.optim.AdamW):
                assert isinstance(self.optimizer.optimizer, torch.optim.AdamW)
            self.optimizer: torch.optim.AdamW
            state = self.optimizer.state[self.optimizer.param_groups[0]["params"][0]]
            if "exp_avg" not in state:
                # update the step, exp_avg, and exp_avg_sq of the optimizer state
                print(
                    "Loading optimizer state from",
                    self.resume_from_optimizer_checkpoint,
                )
                state_dict = torch.load(
                    self.resume_from_optimizer_checkpoint,
                    map_location=self.model.device,
                )["state"]
                trainable_params = (
                    p for p in self.model.parameters() if p.requires_grad
                )
                for state, p in zip(state_dict.values(), trainable_params):  # type: ignore
                    self.optimizer.state[p] = state  # type: ignore
                self.resume_from_optimizer_checkpoint = None

        if self.state.global_step == 1 and self.optimizer is not None:
            state = self.optimizer.state[self.optimizer.param_groups[0]["params"][1]]
            if "exp_avg" in state:
                print(f"Adam buffer dtype: {state['exp_avg'].dtype}")

        return self.compute_loss_custom(model, inputs, return_outputs)

    def compute_loss_custom(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()

        outputs = model(**inputs)
        if self.loss_name in {"xent", "kl"}:
            aux_weight = 0
        elif self.loss_name == "logconf":
            aux_weight = 0.5
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

        loss = log_confidence_loss(
            outputs.logits, labels, self.state.global_step, aux_coef=aux_weight, subtract_label_ent=self.loss_name == "kl"
        )
            

        return (loss, outputs) if return_outputs else loss


class DivDisTrainer(CustomLossTrainer):
    """
    Three loss terms for diversification:
    1. Cross-entropy on trusted data
    2. Independence on unlabeled target data:
        For each pair of heads i and j, KL divergence between
        empirical_P(pred_i, pred_j) and empirical_P(pred_i) * empirical_P(pred_j)
    3. Regularization on unlabeled target data:
        KL divergence between empirical_P(pred_i) and a guess about the label distribution
    However the above loss doesn't work well with small minibatch sizes
    (which we need for memory reasons) so we keep a detached buffer of
    target probs that we concatenate with the minibatch before computing the loss
    """

    def __init__(
        self,
        max_buffer_size: int = 100,
        indep_coef: float = 1.0,
        reg_coef: float = 1.0,
        reg_guess: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_buffer = []
        self.max_buffer_size = max_buffer_size
        self.indep_coef = indep_coef
        self.reg_coef = reg_coef
        self.reg_guess = reg_guess

    def compute_loss_custom(self, model, inputs, return_outputs=False):
        # we use label -1 for target data
        labels = inputs.pop("labels").float()
        target_idxs = labels == -1

        outputs = model(**inputs)
        trusted_logits = outputs.logits[~target_idxs]
        h = trusted_logits.shape[-2]
        trusted_logits = trusted_logits.reshape(-1, trusted_logits.shape[-1])
        trusted_labels = labels[~target_idxs].repeat_interleave(
            h, dim=0
        )  # [b,] -> [b * h,]

        xent_loss = (
            log_confidence_loss(
                trusted_logits, trusted_labels, self.state.global_step, aux_coef=0
            )
            if trusted_logits.shape[0] > 0
            else 0
        )

        if target_idxs.any():
            target_logits = outputs.logits[target_idxs]
            target_probs = torch.softmax(target_logits, dim=-1)
            cat_target_probs = torch.cat([target_probs, *self.target_buffer], dim=0)
            indep_loss = mutual_info_loss(cat_target_probs)

            if self.reg_guess is None:
                # assume uniform distribution
                reg_guess = torch.ones_like(target_probs) / target_probs.shape[-1]
            else:
                reg_guess = (
                    self.reg_guess.expand_as(target_probs)
                    .type_as(target_probs)
                    .to(target_probs.device)
                )
            # KL divergence over the last dimension, averaged over heads and batch
            reg_loss = (
                (
                    reg_guess
                    * (reg_guess.log() - torch.log_softmax(target_logits, dim=-1))
                )
                .sum(dim=-1)
                .mean()
            )

            self.target_buffer.append(target_probs.detach())
            self.target_buffer = self.target_buffer[-self.max_buffer_size :]
        else:
            indep_loss, reg_loss = 0, 0
        loss = xent_loss + self.indep_coef * indep_loss + self.reg_coef * reg_loss
        return (loss, outputs) if return_outputs else loss


def mutual_info_loss(probs):
    """Input: predicted probabilites on target batch."""
    B, H, D = probs.shape  # B=batch_size, H=heads, D=pred_dim
    marginal_p = probs.mean(dim=0)  # H, D
    marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p)  # H, H, D, D
    marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # H^2, D^2
    joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(dim=0)  # H, H, D, D
    joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)")  # H^2, D^2
    kl_divs = joint_p * (joint_p.log() - marginal_p.log())
    kl_grid = rearrange(kl_divs.sum(dim=-1), "(h g) -> h g", h=H)  # H, H
    pairwise_mis = torch.triu(
        kl_grid, diagonal=1
    )  # Get only off-diagonal KL divergences
    return pairwise_mis.mean()


def log_confidence_loss(
    logits,
    labels,
    step: int,
    warmup_steps: int = 200,
    aux_coef: float = 0.5,
    subtract_label_ent: bool = False,
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
    labels_one_hot = torch.stack([1.0 - labels, labels], dim=1)
    target = labels_one_hot * (1 - coef) + strong_preds.detach() * coef
    loss = torch.nn.functional.cross_entropy(logits, target)
    if subtract_label_ent:
        avg_label_ent = -torch.sum(labels_one_hot * torch.log(labels_one_hot + 1e-10), dim=1).mean()
        loss = loss - avg_label_ent
    return loss
