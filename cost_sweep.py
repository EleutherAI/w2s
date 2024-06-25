import copy

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

for weak_labels_dir in [
    "amazon_polarity_title_only",
    "sciq_support_contains",
    "ethics_deontology_excuse_only",
]:
    base_command = (
        "python train_transformer_reporter.py "
        "{weak_ds_path} "
        "{oracle_ds_path} "
        "{test_ds_path} "
        "9_000 9_000 2686 "
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
        if num_weak > 9_000 or num_oracle > 9000:
            return

        stages = copy.deepcopy(stages)
        for stage in stages:
            total_points = stage["size"] * stage["num_train_epochs"]
            # NOTE: this preserves number of total datapoints, but not:
            #   - relative ratios of where each type of data is used
            #   - the total number of batches for size<batch_size
            # note that it also often can use more data than specified by num_weak and num_oracle
            # because of multiple stages, but also could be less because of sampling data repeatedly
            # so you should look at logs for actual number of each
            if stage["type"] == "weak":
                if num_weak == 0:
                    # NOTE: we just skip these runs where there's
                    # a stage expecting weak data but none provided
                    return
                stage["size"] = num_weak
                stage["num_train_epochs"] = max(total_points // num_weak, 1)
            elif stage["type"] == "oracle":
                if num_oracle == 0:
                    return
                stage["size"] = num_oracle
                stage["num_train_epochs"] = max(total_points // num_oracle, 1)

        seed = 1
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

    for stages in stages_list:
        for num_weak, num_oracle in [
            (100, 100),
            (10, 100),
            (100, 1000),
            (1000, 10),
            (1000, 100),
        ]:
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

    # for budget in [100, 1000, 2500, 10_000, 90_000]:
    #     # get w2s perf
    #     for stage in stages:
    #         stage["type"] = "weak"
    #     print(get_command(stages, 10, budget, 0.0))
    #     # get ceiling perf
    #     for stage in stages:
    #         stage["type"] = "oracle"
    #     print(get_command(stages, 10, budget, 1.0))
