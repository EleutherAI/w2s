import subprocess
from multiprocessing import Process

# Define the datasets and respective GPU ids
datasets = [
    "boolq",
    "anli-r2",
    "cosmos_qa",
    "mc_taco",
    "sciq",
    "paws",
    "twitter-sentiment",
    "wic",
]

gpu_ids = range(len(datasets))

# Define the base command
base_command = (
    "CUDA_VISIBLE_DEVICES={gpu_id} "
    "python run.py "
    "--dataset {dataset} "
    "--weak_model_name Qwen/Qwen1.5-0.5B "
    "--strong_model_name meta-llama/Meta-Llama-3-8B "
    "--n_epochs 2 "
    "--n_train 20_000 "
    "--n_val 500 "
    "--n_test 1000 "
    "--n_predict 0 "
    "--eval_every 25 "
    "--save_every 25 "
    "--logconf_warmup_steps 200 "
    "--logconf_weight 0.5 "
    "--strong_weight 0.5 "
    "--minibatch_size 4 "
    '--run_name "logconf_no_warmup" '
)


def run_command(command):
    subprocess.run(command, shell=True, check=True)


# List to hold processes
processes = []

# Loop over datasets and gpu_ids
for dataset, gpu_id in zip(datasets, gpu_ids):
    command = base_command.format(gpu_id=gpu_id, dataset=dataset)
    print(f"Running command: {command}")  # Debug print
    p = Process(target=run_command, args=(command,))
    p.start()
    processes.append(p)

# Wait for all processes to complete
for p in processes:
    p.join()
