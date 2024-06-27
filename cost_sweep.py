import copy

temp = 0.25
# CFG 1: LP(weak), FT(GT), FT(weak) with new head, FT(GT)
# stages_list = [
#     [
#         {
#             "modules_with_grad": "head",
#             "type": "weak",
#             "size": 1024,
#             "sampling": "most_confident_label",
#             "sample_temp": temp,
#             "num_train_epochs": 3,
#         },
#         {
#             "modules_with_grad": "all",
#             "type": "oracle",
#             "size": 32,
#             "sampling": "least_confident_pred",
#             "sample_temp": temp,
#             "num_train_epochs": 100,
#         },
#         {
#             "modules_with_grad": "all",
#             "reinit_head": True,
#             "type": "weak",
#             "size": 1024,
#             "sampling": "most_confident_label",
#             "sample_temp": temp,
#             "num_train_epochs": 1,
#         },
#         {
#             "modules_with_grad": "all",
#             "type": "oracle",
#             "size": 32,
#             "sampling": "least_confident_pred",
#             "sample_temp": temp,
#             "num_train_epochs": 50,
#         },
#     ],
# ]
stages_list = [
    [
        {
            "modules_with_grad": "all",
            "type": "weak",
            "size": 1024,
            "sampling": "random",
            "num_train_epochs": 3,
        },
        {
            "modules_with_grad": "all",
            "type": "oracle",
            "size": 32,
            "sampling": "random",
            "num_train_epochs": 100,
        },
    ],
]

for weak_labels_dir in [
    "amazon_polarity_title_only",
    "sciq_support_contains",
    "ethics_deontology_excuse_only",
    "boolq_Qwen1.5-0.5B",
    "paws_consistency_weak_amplified",
]:
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
        f"--results_folder /mnt/ssd-1/alexm/w2s/results/{weak_labels_dir} "
        '--run_name "{run_name}" '
    )

    weak_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_labels_dir}/weak_train"
    oracle_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_labels_dir}/weak_train"
    test_ds_path = f"/mnt/ssd-1/alexm/w2s/results/{weak_labels_dir}/weak_test"

    def get_command(stages, num_weak, num_oracle):
        stages = copy.deepcopy(stages)
        # keep only the stages where there is any data to run them with
        stages = [
            stage
            for stage in stages
            if (stage["type"] == "weak" and num_weak > 0)
            or (stage["type"] == "oracle" and num_oracle > 0)
        ]
        for stage in stages:
            total_points = stage["size"] * stage["num_train_epochs"]
            # NOTE: this preserves number of total datapoints, but not:
            #   - relative ratios of where each type of data is used
            #   - the total number of batches for size<batch_size
            # note that it also often can use more data than specified by num_weak and num_oracle
            # because of multiple stages, but also could be less because of sampling data repeatedly
            # so you should look at logs for actual number of each
            if stage["type"] == "weak":
                stage["size"] = num_weak
                stage["num_train_epochs"] = max(total_points // num_weak, 1)
            elif stage["type"] == "oracle":
                stage["size"] = num_oracle
                stage["num_train_epochs"] = max(total_points // num_oracle, 1)

        seed = 3
        # run_name = f"oc={oracle_cost}_b={budget}_sr={spending_frac}_cfg0_s{seed}"
        run_name = f"nw={num_weak}_no={num_oracle}_cfg0_s{seed}"
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
        (2, 100),
        (2, 120),
        (2, 200),
        (2, 50),
        (2, 70),
        (2, 400),
        (2, 800),
        (450, 50),
        (800, 20),
        (3000, 300),
        (2000, 130),
        (1000, 25),
        (100, 800),
        (2500, 100),
        (100, 500),
        (400, 400),
        (4000, 700),
        (6000, 1000),
        (100, 100),
        (10, 100),
        (100, 1000),
        (1000, 10),
        (1000, 100),
        (7000, 2),
        (6500, 2),
        (7000, 10),
        (6400, 20),
        (7000, 100),
        (6800, 300),
        (6600, 2000),
        (6800, 5000),
        (1000, 5500),
        (100, 7000),
        (2, 7000),
        (2, 5000),
        (20, 7000),
        (4, 2000),
        (4, 4000),
    ]
    pairs += [(0, num_oracle) for num_oracle in [10, 100, 300, 1000, 3000, 10000]]
    pairs += [(num_weak, 0) for num_weak in [100, 600, 3000, 10000]]
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
