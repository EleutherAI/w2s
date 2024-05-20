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
    "python train.py {dataset} "
    "--weak_name Qwen/Qwen1.5-0.5B "
    "--strong_model meta-llama/Meta-Llama-3-8B "
    "--minibatch_size 4 "
    '--run_name "_estop_xent_llama3_s2s" '
    "--s2s_iter 10"
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
