import subprocess
from multiprocessing import Process
import sys

# Define the datasets and respective GPU ids
# list of tuples with dataset name and minibatch size
configs = [
    ("boolq", 1),
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
    "--minibatch_size {minibatch_size} "
    "{argv} "
)


def run_command(command):
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    if "--gpus" in sys.argv:
        i = sys.argv.index("--gpus")
        included_gpu_ids = [int(gpu_id) for gpu_id in sys.argv[i + 1].split(",")]
        other_argv = sys.argv[1:i] + sys.argv[i + 2:]
        
    else:
        included_gpu_ids = gpu_ids
        other_argv = sys.argv[1:]

    argv = " ".join(other_argv)

    # List to hold processes
    processes = []

    # Loop over datasets and gpu_ids
    for (dataset, minibatch_size), gpu_id in zip(configs, gpu_ids):
        if gpu_id not in included_gpu_ids:
            continue
        command = base_command.format(
            gpu_id=gpu_id, 
            dataset=dataset, 
            minibatch_size=minibatch_size,
            argv=argv,
        )
        squished = command.replace(" ", "_").replace("/", "_").replace("=", "_")
        command += f" | tee logs/{squished}.log"
        print(f"Running command: {command}")  # Debug print
        p = Process(target=run_command, args=(command,))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()
