import subprocess
from multiprocessing import Process
from pathlib import Path

configs = [
    (1, 0, "ss_contains_0_matching"),
    (32, 48, "ss_contains_32x48_matching"),
    (128, 12, "ss_contains_128x12_matching"),
    (512, 4, "ss_contains_512x4_matching"),
    (512, 1, "ss_contains_512_matching"),
    (2000, 1, "ss_contains_2000_matching"),
    (2000, 4, "ss_contains_2000x4_matching"),
    (8000, 1, "ss_contains_8000_matching"),
]

gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]


base_command = (
    "CUDA_VISIBLE_DEVICES={gpu_id} "
    "python train_transformer_reporter.py "
    "{weak_ds_path} "
    "{oracle_ds_path} "
    "{test_ds_path} "
    "{n_train} 2000 2000 "
    "--strong_model_name meta-llama/Meta-Llama-3-8B "
    # "--strong_model_name mistralai/Mistral-7B-v0.1 "
    "--w2s_num_train_epochs {n_epochs} "
    "--oracle_num_train_epochs 1 "
    "--oracle_warmup_steps 0 "
    "--eval_steps 50 "
    "--save_steps 50 "
    "--save_total_limit 1 "
    "--per_device_train_batch_size 1 "
    "--per_device_eval_batch_size 16 "
    "--gradient_accumulation_steps 32 "
    "--results_folder /mnt/ssd-1/alexm/w2s/results/sciq_support_contains "
    '--run_name "{run_name}" '
)


def run_command(command):
    subprocess.run(command, shell=True, check=True)


floor_command = (
    "CUDA_VISIBLE_DEVICES=0 "
    "python run_simple_sft.py "
    "sciq_support_contains "
    "--per_device_train_batch_size 4 "
    "--gradient_accumulation_steps 8 "
)

weak_ds_path = "/mnt/ssd-1/alexm/w2s/results/sciq_support_contains/weak_train"
oracle_ds_path = "/mnt/ssd-1/alexm/w2s/results/sciq_support_contains/weak_train"
test_ds_path = "/mnt/ssd-1/alexm/w2s/results/sciq_support_contains/weak_test"

if (
    Path(weak_ds_path).is_dir()
    and Path(oracle_ds_path).is_dir()
    and Path(test_ds_path).is_dir()
):
    print("Weak dataset exists, skipping floor model training")
else:
    print("Training weak floor model")
    run_command(floor_command)

processes = []

# Loop over datasets and gpu_ids
for gpu_id, (n_train, n_epochs, run_name) in zip(gpu_ids, configs):
    command = base_command.format(
        gpu_id=gpu_id,
        n_train=n_train,
        run_name=run_name,
        n_epochs=n_epochs,
        weak_ds_path=weak_ds_path,
        oracle_ds_path=oracle_ds_path,
        test_ds_path=test_ds_path,
    )
    print(f"Running command: {command}")  # Debug print
    p = Process(target=run_command, args=(command,))
    p.start()
    processes.append(p)

# Wait for all processes to complete
for p in processes:
    p.join()
