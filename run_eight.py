import subprocess
from multiprocessing import Process

# Define the datasets and respective GPU ids
gpu_ids = [0, 1]  # NOTE
weak_labels_dir = "amazon_polarity_title_only"  # NOTE
cfgs = [
    # CFG 1: LP(weak), FT(GT), FT(weak) with new head, FT(GT)
    [
        {
            "modules_with_grad": "head",
            "type": "weak",
            "size": 1024,
            "sampling": "most_confident_label",
            "sample_temp": 0.1,
            "num_train_epochs": 3,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 32,
            "sampling": "least_confident_pred",
            "sample_temp": 0.1,
            "num_train_epochs": 100,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "reinit_head": True,
            "type": "weak",
            "size": 1024,
            "sampling": "most_confident_label",
            "sample_temp": 0.1,
            "num_train_epochs": 1,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 32,
            "sampling": "least_confident_pred",
            "sample_temp": 0.1,
            "num_train_epochs": 50,
            "n_test": 500,
        },
    ],
    # CFG 1: LP(weak), FT(GT), FT(weak) with new head, FT(GT)
    [
        {
            "modules_with_grad": "head",
            "type": "weak",
            "size": 1024,
            "sampling": "most_confident_label",
            "sample_temp": 0.5,
            "num_train_epochs": 3,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 32,
            "sampling": "least_confident_pred",
            "sample_temp": 0.5,
            "num_train_epochs": 100,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "reinit_head": True,
            "type": "weak",
            "size": 1024,
            "sampling": "most_confident_label",
            "sample_temp": 0.5,
            "num_train_epochs": 1,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 32,
            "sampling": "least_confident_pred",
            "sample_temp": 0.5,
            "num_train_epochs": 50,
            "n_test": 500,
        },
    ],
]
base_command = (
    "CUDA_VISIBLE_DEVICES={gpu_id} "
    "python train_transformer_reporter.py "
    "{weak_ds_path} "
    "{oracle_ds_path} "
    "{test_ds_path} "
    "8_000 8_000 2686 "  # NOTE
    "--seed {seed} "
    "--strong_model_name meta-llama/Meta-Llama-3-8B "
    "--reporter_stages {reporter_stages} "
    "--num_train_epochs 1 "
    "--eval_steps 50 "
    "--save_steps 50 "
    "--save_total_limit 1 "
    "--per_device_train_batch_size 1 "
    "--per_device_eval_batch_size 3 "
    "--gradient_accumulation_steps 32 "
    f"--results_folder /mnt/ssd-1/alexm/w2s/results/{weak_labels_dir} "
    '--run_name "{run_name}" '  # NOTE
)


def run_command(command):
    subprocess.run(command, shell=True, check=True)


weak_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_labels_dir}/weak_train"
oracle_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_labels_dir}/weak_train"
test_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_labels_dir}/weak_test"

processes = []

# Loop over datasets and gpu_ids
for gpu_id, (i, cfg) in zip(gpu_ids, enumerate(cfgs)):
    seed_offset = 1
    command = base_command.format(
        gpu_id=gpu_id,
        weak_ds_path=weak_ds_path,
        oracle_ds_path=oracle_ds_path,
        test_ds_path=test_ds_path,
        seed=i + seed_offset,
        reporter_stages=len(cfg),
        run_name=f"am_title_temp_sweep_s{seed_offset}_" + str(i),
    )
    for j, stage in enumerate(cfg):
        prefix = f"stage{j}_"
        for key, value in stage.items():
            if value is True:
                command += f"--{prefix}{key} "
            else:
                command += f"--{prefix}{key} {value} "

    print(f"Running command: {command}")  # python  Debug print
    p = Process(target=run_command, args=(command,))
    p.start()
    processes.append(p)

# Wait for all processes to complete
for p in processes:
    p.join()
