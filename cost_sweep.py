import copy

import pandas as pd

temp = 0.25
# CFG 1: LP(weak), FT(GT), FT(weak) with new head, FT(GT)
stages_list = [
    [
        {
            "modules_with_grad": "head",
            "type": "weak",
            "size": 1024,
            "sampling": "most_confident_label",
            "sample_temp": temp,
            "num_train_epochs": 3,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 32,
            "sampling": "least_confident_pred",
            "sample_temp": temp,
            "num_train_epochs": 100,
        },
        {
            "modules_with_grad": "all",
            "reinit_head": True,
            "type": "weak",
            "size": 1024,
            "sampling": "most_confident_label",
            "sample_temp": temp,
            "num_train_epochs": 1,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 32,
            "sampling": "least_confident_pred",
            "sample_temp": temp,
            "num_train_epochs": 50,
        },
    ],
]
stages_list = [
    [
        {
            "modules_with_grad": "all",
            "type": "weak",
            "sampling": "random",
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "sampling": "random",
        },
    ],
]

salience_df = pd.read_json("results/salience_results.json", lines=True)
salience_df = salience_df[salience_df["against"] == "oracle"]
salience_df.set_index("ds_name", inplace=True)

weak_ds_list = ["boolq_Qwen1.5-0.5B"]
weak_ds_list += [
    f"{ds_name}_{prompt}"
    for ds_name in [
        "ethics_deontology_excuse_only",
        "amazon_polarity_title_only",
        "sciq_support_contains",
        "paws_consistency",
    ]
    for prompt in ["weak_amplified", "gt_amplified"]
]
# weak_ds_list += [f"{ds_name}_{prompt}" for ds_name in
# ["paws_consistency", "ethics_deontology_excuse_only",
# "amazon_polarity_title_only", "sciq_support_contains"]
#  for prompt in ["weak_amplified", "both_amplified", "neither_amplified", "gt_amplified"]]

for weak_ds in weak_ds_list:
    earliest_step = int(salience_df.loc[weak_ds, "earliest_good_step"])  # type: ignore

    base_command = (
        "python train_transformer_reporter.py "
        "{weak_ds_path} "
        "{oracle_ds_path} "
        "{test_ds_path} "
        "10_000 10_000 1000 "
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
        f"--results_folder /mnt/ssd-1/alexm/w2s/results/{weak_ds} "
        '--run_name "{run_name}" '
    )

    weak_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_ds}/weak_train"
    oracle_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_ds}/weak_train"
    test_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_ds}/weak_test"

    def get_command(stages, num_weak, num_oracle):
        stages = copy.deepcopy(stages)
        # keep only the stages where there is any data to run them with
        stages = [
            stage
            for stage in stages
            if (stage["type"] == "weak" and num_weak > 0)
            or (stage["type"] == "oracle" and num_oracle > 0)
        ]
        total_points = earliest_step * 2
        for stage in stages:
            # NOTE: this preserves number of total datapoints, but not:
            #   - relative ratios of where each type of data is used
            #   - the total number of batches for size<batch_size
            # note that it also often can use more data than specified by num_weak and num_oracle
            # because of multiple stages, but also could be less because of sampling data repeatedly
            # so you should look at logs for actual number of each
            num = num_weak if stage["type"] == "weak" else num_oracle
            num_points = round(total_points * num / (num_weak + num_oracle))
            num_epochs = max(num_points / num, 1)
            stage["size"] = num
            stage["num_train_epochs"] = num_epochs

        seed = 4
        # run_name = f"oc={oracle_cost}_b={budget}_sr={spending_frac}_cfg0_s{seed}"
        run_name = f"nw={num_weak}_no={num_oracle}_seq_sft_s{seed}"
        command = base_command.format(
            weak_ds_path=weak_ds_path,
            oracle_ds_path=oracle_ds_path,
            test_ds_path=test_ds_path,
            seed=seed,
            reporter_stages=len(stages),
            run_name=run_name,
        )
        for j, stage in enumerate(stages):
            prefix = f"stage{j}_"
            for key, value in stage.items():
                if value is True:
                    command += f"--{prefix}{key} "
                else:
                    command += f"--{prefix}{key} {value} "

        return command

    pairs = [
        # (2, 100),
        # (2, 120),
        # (2, 200),
        (50, 10),
        # (2, 70),
        # (2, 400),
        (800, 8),
        (450, 50),
        (800, 20),
        # (3000, 300),
        # (2000, 130),
        (1000, 25),
        # (100, 800),
        (2500, 120),
        (100, 500),
        # (400, 400),
        (4000, 700),
        (6000, 1000),
        (100, 100),
        # (10, 100),
        # (100, 1000),
        (1000, 10),
        (1000, 100),
        (500, 100),
        # (6500, 2),
        (7000, 10),
        # (6400, 20),
        (7000, 100),
        (6800, 300),
        # (6600, 2000),
        (6800, 5000),
        (1000, 5500),
        # (7000, 100),
        # (2, 7000),
        # (2, 5000),
        # (20, 7000),
        (1000, 4),
        (5000, 20),
    ]
    pairs += [(0, num_oracle) for num_oracle in [10, 100, 300, 1000, 5000]]
    pairs += [(num_weak, 0) for num_weak in [100, 600, 3000]]
    for stages in stages_list:
        for num_weak, num_oracle in pairs:
            cmd = get_command(stages, num_weak, num_oracle)
            if cmd:
                print(cmd)

    stages = [
        {
            "modules_with_grad": "head",
            "size": 1024,
            "sampling": "random",
            "num_train_epochs": 3,
        },
        {
            "modules_with_grad": "all",
            "size": 32,
            "sampling": "random",
            "num_train_epochs": 100,
        },
        {
            "modules_with_grad": "all",
            "reinit_head": True,
            "size": 1024 + 1600,
            "num_train_epochs": 1,
        },
    ]
