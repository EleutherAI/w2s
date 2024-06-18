import subprocess
from multiprocessing import Process

# Define the datasets and respective GPU ids
gpu_ids = [0, 1, 2]  # NOTE
weak_labels_dir = "amazon_polarity_title_only"  # NOTE
cfgs = [
    # CFG 1: LP(GT), FT(weak) frozen head, FT(oracle) reinit head
    [
        {
            "modules_with_grad": "head",
            "type": "weak",
            "size": 32,
            "weight_decay": 1.0,
            "sampling": "random",
            "num_train_epochs": 100,
            "n_test": 500,
        },
        {
            "modules_with_grad": "body",
            "type": "weak",
            "size": 128,
            "sampling": "most_confident_label",
            "num_train_epochs": 16,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "reinit_head": True,
            "type": "oracle",
            "size": 32,
            "sampling": "least_confident_pred",
            "sample_temp": 0.5,
            "num_train_epochs": 16,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 32,
            "sampling": "least_confident_pred",
            "sample_temp": 0.5,
            "reuse_optimizer_checkpoint": True,
            "num_warmup_steps": 0,
            "num_train_epochs": 16,
        },
    ],
    # CFG 2: LP(weak), FT(weak) frozen head, FT(oracle)
    [
        {
            "modules_with_grad": "head",
            "type": "weak",
            "size": 32,
            "sampling": "most_confident_label",
            "weight_decay": 1.0,
            "num_train_epochs": 100,
            "n_test": 500,
        },
        {
            "modules_with_grad": "body",
            "type": "weak",
            "size": 128,
            "sampling": "most_confident_label",
            "num_train_epochs": 16,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 32,
            "sampling": "least_confident_pred",
            "num_train_epochs": 16,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 32,
            "sampling": "least_confident_pred",
            "reuse_optimizer_checkpoint": True,
            "num_warmup_steps": 0,
            "num_train_epochs": 16,
        },
    ],
    # CFG 4: LP with weak, FT with gt, FT with weak with random head, FT with gt again but body, then ft with gt  # noqa
    [
        {
            "modules_with_grad": "head",
            "type": "weak",
            "size": 1024,
            "sampling": "most_confident_label",
            "num_train_epochs": 3,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 32,
            "sampling": "least_confident_pred",
            "num_train_epochs": 60,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "reinit_head": True,
            "type": "weak",
            "size": 1024,
            "sampling": "most_confident_label",
            "num_train_epochs": 1,
            "n_test": 500,
        },
        {
            "modules_with_grad": "body",
            "type": "oracle",
            "size": 16,
            "sampling": "least_confident_pred",
            "sample_temp": 0.5,
            "num_train_epochs": 50,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 16,
            "sampling": "least_confident_pred",
            "sample_temp": 0.5,
            "num_train_epochs": 50,
            "n_test": 500,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 16,
            "sampling": "least_confident_pred",
            "sample_temp": 0.5,
            "reuse_optimizer_checkpoint": True,
            "num_warmup_steps": 0,
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
    command = base_command.format(
        gpu_id=gpu_id,
        weak_ds_path=weak_ds_path,
        oracle_ds_path=oracle_ds_path,
        test_ds_path=test_ds_path,
        seed=i + 2,
        reporter_stages=len(cfg),
        run_name="am_title_online_sampling_s2_" + str(i),
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
