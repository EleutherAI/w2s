import subprocess
from multiprocessing import Process

# Define the datasets and respective GPU ids
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # NOTE
weak_labels_dir = "amazon_polarity_title_only_denoised"  # NOTE

base_command = (
    "CUDA_VISIBLE_DEVICES={gpu_id} "
    "python train_transformer_reporter.py "
    "{weak_ds_path} "
    "{oracle_ds_path} "
    "{test_ds_path} "
    "{n_train} 1025 2686 "  # NOTE
    "--seed {seed} "
    "--strong_model_name meta-llama/Meta-Llama-3-8B "
    "--reporter_method ActiveSftReporter "  # NOTE
    "--w2s_num_train_epochs {n_epochs} "
    "--oracle_warmup_steps 0 "
    "--eval_steps 50 "
    "--save_steps 50 "
    "--save_total_limit 1 "
    "--per_device_train_batch_size 1 "
    "--per_device_eval_batch_size 16 "
    "--gradient_accumulation_steps 32 "
    f"--results_folder /mnt/ssd-1/alexm/w2s/results/{weak_labels_dir} "
    '--run_name "amt_denoised_{n_train}x{n_epochs}_seed{seed}_active" '  # NOTE
)


def run_command(command):
    subprocess.run(command, shell=True, check=True)


weak_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_labels_dir}/weak_train"
oracle_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_labels_dir}/weak_train"
test_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_labels_dir}/weak_test"

processes = []

# Loop over datasets and gpu_ids
for gpu_id, seed in zip(gpu_ids, range(1000, 1000 + len(gpu_ids))):
    command = base_command.format(
        gpu_id=gpu_id,
        n_train=1,  # NOTE
        n_epochs=0,  # NOTE
        seed=seed,
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
