import subprocess
from multiprocessing import Process

# Define the datasets and respective GPU ids
configs = [
    # ("boolq", 2),
    # ("anli-r2", 8),
    # ("cosmos_qa", 4),
    ("mc_taco", 4),
    ("sciq", 4),
    # ("paws", 16),
    ("twitter-sentiment", 8),
    ("wic", 8),
]

gpu_ids = range(len(configs))

# Define the base command
base_command = (
    "CUDA_VISIBLE_DEVICES={gpu_id} "
    "python run.py "
    "--dataset {dataset} "
    "--weak_model_name Qwen/Qwen1.5-0.5B "
    "--strong_model_name meta-llama/Meta-Llama-3-8B "
    "--n_epochs 3 "
    "--n_train 10_000 "
    "--n_val 1000 "
    "--n_test 5_000 "
    "--n_predict 0 "
    "--eval_every 100 "
    "--save_every 100 "
    "--save_total_limit 1 "
    "--logconf_warmup_steps 80 "
    "--logconf_weight 0.75 "
    "--strong_weight 0.5 "
    "--minibatch_size {minibatch_size} "
    "--weak_lr 5e-4 "
    "--strong_lr 8e-5 "
    '--run_name "log_entropy" '
)


def run_command(command):
    subprocess.run(command, shell=True, check=True)


# List to hold processes
processes = []

# Loop over datasets and gpu_ids
for (dataset, minibatch_size), gpu_id in zip(configs, gpu_ids):
    command = base_command.format(
        gpu_id=gpu_id, dataset=dataset, minibatch_size=minibatch_size
    )
    print(f"Running command: {command}")  # Debug print
    p = Process(target=run_command, args=(command,))
    p.start()
    processes.append(p)

# Wait for all processes to complete
for p in processes:
    p.join()
