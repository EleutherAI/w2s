import copy

# For amazon:
# Use variations of cfg 0:
# We know a ceiling
# For each cost ratio and budget we want to know how close each method gets to that ceiling
# Weak labels are fixed at $10 per label
# Oracle labels vary from like $20 per label to $1,000,000 per label
# Budget varies from like $1,000 to $10,000,000
# So e.g. for ocost = $1,000 and budget = $10,000, we could try spending 0%, 10%, 50%, 90%, 100% of
#   our budget on oracle labels, with various methods
# This quickly gets huge (4 budgets x 4 oracle costs x 5 spending ratios x 4 methods = 320 ft
#   runs for amazon alone, x3 = 960)
#  - Get rid of 0% and 100% (I can add some more special cases later)
#  - Focus on one or two methods (start with just my best cfg)
#  - oracle costs: [100, 10_000, 1_000_000]
#  - budgets: [1000, 100_000, 10_000_000]
#  - only keep cases where budget >= 10x oracle_cost and num_weak <= 10_000 and num_oracle <= 2000
# around 30 ft runs then

# Define the datasets and respective GPU ids
weak_labels_dir = "ethics_deontology_excuse_only"  # NOTE

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

weak_cost = 10
oracle_costs = [100, 4_000, 160_000]
budgets = [1000, 100_000, 10_000_000]
spending_ratios = [0.1, 0.5, 0.99]

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


def get_command(stages, oracle_cost, budget, spending_ratio):
    num_oracle = int(budget * spending_ratio // oracle_cost)
    num_weak = (budget - num_oracle * oracle_cost) // weak_cost

    if budget < 10 * oracle_cost:
        return
    if num_weak > 9_000 or num_oracle > 2000:
        return

    stages = copy.deepcopy(stages)
    for stage in stages:
        total_points = stage["size"] * stage["num_train_epochs"]
        if stage["type"] == "weak":
            stage["size"] = num_weak
            stage["num_train_epochs"] = max(total_points // num_weak, 1)
        elif stage["type"] == "oracle":
            stage["size"] = num_oracle
            stage["num_train_epochs"] = max(total_points // num_oracle, 1)

    seed = 0
    run_name = f"oc={oracle_cost}_b={budget}_sr={spending_ratio}_cfg0_s{seed}"
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


# for stages in stages_list:
#     for oracle_cost in oracle_costs:
#         for budget in budgets:
#             for spending_ratio in spending_ratios:
#                 cmd = get_command(stages, oracle_cost, budget, spending_ratio)
#                 if cmd:
#                     print(cmd)

for budget in [1000, 10_000, 100_000]:
    # get w2s perf
    print(get_command(stages_list[0], 10, budget, 0.0))
    # get ceiling perf
    print(get_command(stages_list[0], 10, budget, 1.0))
