from typing import Optional

from w2s.model import MODEL_REGISTRY


def set_default_args(args: dict, model_name: str, run_name: Optional[str] = None):
    """Set default arguments for training a model."""
    # set defaults
    args["num_train_epochs"] = args.get("num_train_epochs", 1)
    args["per_device_train_batch_size"] = args.get("per_device_train_batch_size", 8)
    args["per_device_eval_batch_size"] = args.get("per_device_eval_batch_size", 32)
    args["gradient_accumulation_steps"] = args.get("gradient_accumulation_steps", 4)
    args["warmup_steps"] = args.get("warmup_steps", 40)
    args["lr_scheduler_type"] = args.get("lr_scheduler_type", "cosine")
    args["weight_decay"] = args.get("weight_decay", 0.01)
    args["eval_strategy"] = args.get("eval_strategy", "steps")
    args["eval_steps"] = args.get("eval_steps", 50)
    args["save_strategy"] = args.get("save_strategy", "steps")
    args["save_steps"] = args.get("save_steps", 50)
    args["logging_steps"] = args.get("logging_steps", 25)
    args["load_best_model_at_end"] = args.get("load_best_model_at_end", False)
    args["save_total_limit"] = args.get("save_total_limit", 1)
    args["adam_beta2"] = args.get("adam_beta2", 0.95)
    args["tf32"] = args.get("tf32", True)
    args["label_names"] = args.get("label_names", ["labels"])
    args["learning_rate"] = args.get("learning_rate", MODEL_REGISTRY[model_name]["lr"])
    if run_name is not None:
        args["run_name"] = run_name
    return args
