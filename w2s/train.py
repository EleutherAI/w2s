from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from datasets import Sequence, Value
from peft import (
    AutoPeftModelForCausalLM,
    AutoPeftModelForSequenceClassification,
    LoraConfig,
    get_peft_model,
)
from simple_parsing import Serializable, field, parse
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalLoopOutput
from transformers.trainer_pt_utils import find_batch_size

import wandb

from .ds_registry import load_and_process_dataset
from .knn import gather_hiddens, zeta_filter
from .loss import log_confidence_loss
from .roc_auc import roc_auc


@dataclass
class TrainConfig(Serializable):
    weak_name: str
    """Name of the weak model to use."""

    dataset: str = field(positional=True)
    """Name of the dataset to use."""

    contamination: float = field(default=0.0)
    """What fraction of data points to remove as outliers."""

    minibatch_size: int = 8
    """Size of the minibatches to use during training."""

    outlier_k: int = field(default=0)
    """Number of neighbors to consider when removing outliers."""

    run_name: str = ""
    """Name of the run."""


class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        frac = self.state.global_step / self.state.max_steps
        loss = log_confidence_loss(outputs.logits, labels, frac)

        return (loss, outputs) if return_outputs else loss


class DistillationChessTrainer(Trainer):
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.pop("labels")

    #     outputs = model(**inputs)
    #     frac = self.state.global_step / self.state.max_steps
    #     breakpoint()
    #     loss = log_confidence_loss(outputs.logits, labels, frac)

    #     return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
            self,
            dataloader,
            description,
            ignore_keys,
            metric_key_prefix,
        ) -> EvalLoopOutput:
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        args = self.args
        
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        model.eval()
        eval_dataset = getattr(dataloader, "dataset", [])
        num_samples = len(eval_dataset)

        running_mean_loss = 0.0
        running_mean_acc = 0.0
        predictions = []
        for _, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, False, ignore_keys=ignore_keys)
            mask = labels != -100
            batch_predictions = logits[mask].argmax(dim=-1) # type: ignore
            predictions.append(batch_predictions.cpu().numpy())

            running_mean_loss += (loss.item() / len(dataloader)) # type: ignore
            running_mean_acc += batch_predictions.eq(labels[mask]).float().mean().item() / len(dataloader) # type: ignore

        metrics = dict(
            eval_accuracy=running_mean_acc,
            eval_loss=running_mean_loss
        )
        predictions=np.concatenate(predictions, axis=0)
        
        return EvalLoopOutput(predictions=predictions, label_ids=np.array([]), metrics=metrics, num_samples=num_samples)
    

# Works for both Llama and Qwen architectures
LORA_MODULES = [
    "gate_proj",
    "down_proj",
    "up_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


def move_best_ckpt(trainer: Trainer):
    path = trainer.state.best_model_checkpoint
    perf = trainer.state.best_metric
    assert path is not None, "No best checkpoint found"
    assert perf is not None, "No best metric"

    src = Path(path)
    dest = src.parent / "best-ckpt"
    src.rename(dest)
    print(f"Best model (auroc {perf:.3f}) saved at: {dest}")


def lolcat(lol1, lol2):
    # list-of-list concatenation along the second dimension
    assert len(lol1) == len(lol2)
    return [l1 + l2 for l1, l2 in zip(lol1, lol2)]


def lolconst(lol, const):
    return [[const for _ in l_] for l_ in lol]


def train(cfg: TrainConfig):
    lora_cfg = LoraConfig(target_modules=LORA_MODULES)

    # STRONG_NAME = "meta-llama/Meta-Llama-3-8B"
    STRONG_NAME="Qwen/Qwen1.5-0.5B"
    strong_tokenizer = AutoTokenizer.from_pretrained(STRONG_NAME)
    weak_tokenizer = AutoTokenizer.from_pretrained(cfg.weak_name)

    # Make sure that the pad token is set
    if strong_tokenizer.pad_token_id is None:
        strong_tokenizer.pad_token = strong_tokenizer.eos_token
    if weak_tokenizer.pad_token_id is None:
        weak_tokenizer.pad_token = weak_tokenizer.eos_token

    splits = load_and_process_dataset(
        cfg.dataset, split_sizes=dict(train=20_000, test=1_000)
    )

    if "txt" in splits["train"].column_names:
        task = "classify"
    elif "ctx" in splits["train"].column_names:
        task = "generate"
    else:
        raise ValueError(
            f"Unrecognized dataset columns: {splits['train'].column_names}"
        )

    def weak_processor(examples):
        if task == "generate":
            ctx_out = weak_tokenizer(examples["ctx"], truncation=True)
            trg_out = weak_tokenizer(
                examples["target"], truncation=True, add_special_tokens=False
            )
            return dict(
                input_ids=lolcat(ctx_out["input_ids"], trg_out["input_ids"]),
                attention_mask=lolcat(
                    ctx_out["attention_mask"], trg_out["attention_mask"]
                ),
                labels=lolcat(
                    lolconst(ctx_out["input_ids"], -100), trg_out["input_ids"]
                ),
            )

        out = weak_tokenizer(examples["txt"], truncation=True)
        out["labels"] = examples["hard_label"]
        return out

    def compute_metrics(eval_pred):
        predictions, labels = map(torch.from_numpy, eval_pred)
        if task == "generate":
            print(eval_pred)
            return dict(
                accuracy=predictions.argmax(dim=1).eq(labels).float().mean(),
                loss=eval_pred.loss,
            )
        return dict(
            accuracy=predictions.argmax(dim=1).eq(labels).float().mean(),
            auroc=roc_auc(labels, predictions[:, 1]),
        )

    cols = ["ctx", "target"] if task == "generate" else ["hard_label", "txt"]
    test = splits["test"].select(range(100)).select_columns(cols)
    train = splits["train"].select(range(200)).select_columns(cols)

    weak_test = test.map(weak_processor, batched=True).cast_column(
        "labels", Sequence(Value("int64")) if task == "generate" else Value("int64")
    )
    weak_train = train.map(weak_processor, batched=True).cast_column(
        "labels", Sequence(Value("int64")) if task == "generate" else Value("int64")
    )

    root = Path("results") / cfg.dataset
    training_args = TrainingArguments(
        str(root / "floor"),
        adam_beta2=0.95,
        gradient_accumulation_steps=8 // cfg.minibatch_size,
        evaluation_strategy="epoch",
        # evaluation_strategy="steps",
        # eval_steps=5,
        label_names=["labels"],
        load_best_model_at_end=True,
        logging_steps=50,
        metric_for_best_model="auroc" if task == "classify" else "accuracy",
        per_device_train_batch_size=cfg.minibatch_size,
        per_device_eval_batch_size=cfg.minibatch_size,
        run_name=cfg.dataset + "/floor" + cfg.run_name,
        save_strategy="epoch",
        save_total_limit=1,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        warmup_steps=100,
        weight_decay=0.01,
        bf16_full_eval=True,
    )
    if task == "generate":
        # training_args.eval_accumulation_steps = 6
        training_args.prediction_loss_only=True
        # add custom eval script that accumulates

    # Gather weak labels
    label_dir = root / "floor/preds"
    if label_dir.exists():
        print(f"Loading weak labels from {label_dir}")
        train_labels = torch.load(label_dir / "train.pt")
        test_labels = torch.load(label_dir / "test.pt")
    else:
        should_train = True
        weak_path = root / "floor/best-ckpt"
        if not weak_path.exists():
            autoclass = (
                AutoModelForCausalLM
                if task == "generate"
                else AutoModelForSequenceClassification
            )
            weak_model = autoclass.from_pretrained(cfg.weak_name, torch_dtype="auto")
            if task == "classify":
                # HuggingFace init for the head is too large
                weak_model.score.weight.data *= 0.01
                weak_model.config.problem_type = "single_label_classification"

            weak_model = get_peft_model(weak_model, lora_cfg)
        else:
            print("Loading weak model from:", weak_path)
            should_train = False

            autoclass = (
                AutoPeftModelForCausalLM
                if task == "generate"
                else AutoPeftModelForSequenceClassification
            )
            weak_model = autoclass.from_pretrained(weak_path, torch_dtype="auto")

        weak_model.config.pad_token_id = weak_tokenizer.pad_token_id

        data_collator = (
            DataCollatorForTokenClassification(weak_tokenizer)
            if task == "generate"
            else DataCollatorWithPadding(weak_tokenizer)
        )

        if task == "generate":
            trainer = DistillationChessTrainer(
                args=training_args,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                eval_dataset=weak_test,
                model=weak_model,
                tokenizer=weak_tokenizer,
                train_dataset=weak_train,
            )
        else:
            trainer = DistillationTrainer(
                args=training_args,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                eval_dataset=weak_test,
                model=weak_model,
                tokenizer=weak_tokenizer,
                train_dataset=weak_train,
            )
        if should_train:
            print("\n\033[32m===== Training weak model =====\033[0m")
            trainer.train()
            wandb.finish()
            move_best_ckpt(trainer)

        print("Gathering weak labels")
        train_logits = trainer.predict(weak_train).predictions
        test_logits = trainer.predict(weak_test).predictions

        if task == "classify":
            # Convert to probabilities, then keep only the positive probs
            _, train_labels = torch.from_numpy(train_logits).softmax(-1).unbind(-1)
            _, test_labels = torch.from_numpy(test_logits).softmax(-1).unbind(-1)
        if task == "generate":
            # Convert to hard labels
            train_labels = torch.from_numpy(train_logits)
            test_labels = torch.from_numpy(test_logits)

        label_dir.mkdir(parents=True, exist_ok=True)
        torch.save(train_labels, label_dir / "train.pt")
        torch.save(test_labels, label_dir / "test.pt")

    def strong_processor(examples):
        if task == "generate":
            ctx_out = strong_tokenizer(examples["ctx"], truncation=True)
            trg_out = strong_tokenizer(
                examples["target"], truncation=True, add_special_tokens=False
            )
            return dict(
                input_ids=lolcat(ctx_out["input_ids"], trg_out["input_ids"]),
                attention_mask=lolcat(
                    ctx_out["attention_mask"], trg_out["attention_mask"]
                ),
                hard_label=lolcat(
                    lolconst(ctx_out["input_ids"], -100), trg_out["input_ids"]
                ),
            )

        return strong_tokenizer(examples["txt"], truncation=True)

    strong_train = (
        train.map(strong_processor, batched=True)
        .rename_column("hard_label", "labels")
        .cast_column(
            "labels", Sequence(Value("int64")) if task == "generate" else Value("int64")
        )
    )
    ceil_test = (
        test.map(strong_processor, batched=True)
        .rename_column("hard_label", "labels")
        .cast_column(
            "labels", Sequence(Value("int64")) if task == "generate" else Value("int64")
        )
    )

    strong_ckpt = root / "ceil" / "best-ckpt"
    if strong_ckpt.exists():
        print(f"Strong ceiling model already exists at {strong_ckpt}")
    else:
        print("\n\033[32m===== Training strong ceiling model =====\033[0m")
        strong_model = AutoModelForSequenceClassification.from_pretrained(
            STRONG_NAME, torch_dtype="auto", device_map={"": "cuda"}
        )
        # HuggingFace init for the head is too large
        strong_model.score.weight.data *= 0.01
        strong_model.config.pad_token_id = strong_tokenizer.pad_token_id

        training_args.output_dir = str(root / "ceil")
        training_args.run_name = cfg.dataset + "/ceil" + cfg.run_name

        data_collator = (
            DataCollatorForTokenClassification(strong_tokenizer)
            if task == "generate"
            else DataCollatorWithPadding(strong_tokenizer)
        )

        if task == "generate":
            trainer = DistillationChessTrainer(
                args=training_args,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                eval_dataset=ceil_test,
                model=get_peft_model(strong_model, lora_cfg),
                tokenizer=strong_tokenizer,
                train_dataset=strong_train,
            )
        else:
            trainer = DistillationTrainer(
                args=training_args,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                eval_dataset=ceil_test,
                model=get_peft_model(strong_model, lora_cfg),
                tokenizer=strong_tokenizer,
                train_dataset=strong_train,
            )
    
        trainer.train()
        wandb.finish()
        move_best_ckpt(trainer)

    print("\n\033[32m===== Training w2s model =====\033[0m")
    autoclass = (
        AutoModelForCausalLM
        if task == "generate"
        else AutoModelForSequenceClassification
    )
    
    strong_model = autoclass.from_pretrained(
        STRONG_NAME, torch_dtype="auto", device_map={"": "cuda"}
    )
    # HuggingFace init for the head is too large
    strong_model.score.weight.data *= 0.01
    strong_model.config.pad_token_id = strong_tokenizer.pad_token_id

    # Weak to strong generalization
    acts_path = root / "ceil/acts.pt"
    if acts_path.exists():
        print(f"Loading strong activations from {acts_path}")
        train_acts = torch.load(acts_path, map_location=strong_model.device)
    else:
        print("Gathering strong activations")
        train_acts = gather_hiddens(strong_model, strong_train)
        torch.save(train_acts, acts_path)

    w2s_train = strong_train.remove_columns("labels").add_column(
        "labels", train_labels.numpy()
    )
    if cfg.contamination > 0.0:
        y = train_labels.to(train_acts.device)
        top = zeta_filter(train_acts, y, k=cfg.outlier_k, q=1.0 - cfg.contamination)
        w2s_train = w2s_train.select(top.tolist())

    # Check gt metrics every 100 steps during w2s training.
    # We can overfit to the weak labels before a single epoch.
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = 100
    training_args.save_steps = 100

    training_args.label_names = ["labels"]
    training_args.output_dir = str(root / "w2s") + cfg.run_name
    training_args.run_name = cfg.dataset + "/w2s" + cfg.run_name

    if task == "generate":
        trainer = DistillationChessTrainer(
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            eval_dataset=ceil_test,
            model=get_peft_model(strong_model, lora_cfg),
            tokenizer=strong_tokenizer,
            train_dataset=w2s_train,
        )
    else:
        trainer = DistillationTrainer(
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            eval_dataset=ceil_test,
            model=get_peft_model(strong_model, lora_cfg),
            tokenizer=strong_tokenizer,
            train_dataset=w2s_train,
        )
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    train(parse(TrainConfig))
