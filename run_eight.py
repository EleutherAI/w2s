import subprocess
from multiprocessing import Process

# Define the datasets and respective GPU ids
configs = [
    (1, 0, "sft_0"),
    (128, 12, "sft_128"),
    (512, 6, "sft_512"),
    (2000, 3, "sft_2000"),
    (6000, 2, "sft_6000"),
    (12000, 1, "sft_8000"),
    (32000, 1, "sft_32000"),
    (128000, 1, "sft_128000"),
]

gpu_ids = range(len(configs))

# Define the base command
base_command = (
    "CUDA_VISIBLE_DEVICES={gpu_id} "
    "python train_transformer_reporter.py "
    "amazon_polarity_misleading "
    "{n_train} 2000 "
    "--weak_model_name Qwen/Qwen1.5-0.5B "
    # "--strong_model_name meta-llama/Meta-Llama-3-8B "
    "--strong_model_name mistralai/Mistral-7B-v0.1 "
    "--w2s_num_train_epochs {n_epochs} "
    "--oracle_num_train_epochs 1 "
    "--oracle_warmup_steps 0 "
    "--load_best_model_at_end False "
    "--eval_steps 10 "
    "--save_steps 10 "
    "--save_total_limit 1 "
    "--per_device_train_batch_size 4 "
    "--gradient_accumulation_steps 8 "
    '--run_name "{run_name}" '
)


def run_command(command):
    subprocess.run(command, shell=True, check=True)


# List to hold processes
processes = []

# Loop over datasets and gpu_ids
for gpu_id, (n_train, n_epochs, run_name) in zip(gpu_ids, configs):
    command = base_command.format(
        gpu_id=gpu_id, n_train=n_train, run_name=run_name, n_epochs=n_epochs
    )
    print(f"Running command: {command}")  # Debug print
    p = Process(target=run_command, args=(command,))
    p.start()
    processes.append(p)

# Wait for all processes to complete
for p in processes:
    p.join()
