import subprocess
from multiprocessing import Process
from sys import argv

# Define the datasets and respective GPU ids
configs = [
    ("boolq", 2),
    ("anli-r2", 8),
    ("cosmos_qa", 4),
    ("mc_taco", 4),
    ("sciq", 4),
    ("paws", 16),
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
    "--loss window "
    "--minibatch_size {minibatch_size} "
    "--weak_lr 5e-4 "
    "--strong_lr 8e-5 "
    '--run_name "window4" '
)


def run_command(command):
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    # get GPU ID arguments
    if len(argv) > 1:
        included_gpu_ids = list(map(int, argv[1:]))
        assert all(
            gpu_id in gpu_ids for gpu_id in included_gpu_ids
        ), f"Invalid GPU IDs: {included_gpu_ids}"
    else:
        included_gpu_ids = gpu_ids

    # List to hold processes
    processes = []

    # Loop over datasets and gpu_ids
    for (dataset, minibatch_size), gpu_id in zip(configs, gpu_ids):
        if gpu_id not in included_gpu_ids:
            continue
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
