from w2s.gpu_pool import gpu_map
from w2s.ds_registry import VALID_DATASETS
from datetime import datetime
import sys

def add_log(command, ds, task):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rname = task.split("--run_name ")[1]
    log_file = f"logs/{timestamp}_{ds}_{rname}"
    if len(log_file) > 200:
        log_file = log_file[:200]
    command += f" > {log_file}.log 2>&1"

    return command

date = "0612"
shared_date = "0610"

tasks = [
    f"--loss xent --run_name repro_{date}_xent",
    f"--s2s_iters 2 --run_name repro_{date}_strong2strong",
    f"--probe_relabel --probe knn --run_name repro_{date}_probe_knn",
    f"--probe_relabel --probe logreg --run_name repro_{date}_probe_logreg",
    f"--probe_filter --probe knn --run_name repro_{date}_filter_knn",
    f"--probe_filter --probe logreg --run_name repro_{date}_filter_logreg",
    f"--probe_filter --probe topo --run_name repro_{date}_filter_topo",
    f"--loss window --radius midweak --run_name repro_{date}_window_mid",
    f"--loss entropy --run_name repro_{date}_entropy",
]

jobs = []

configs = []

for i, ds in enumerate(VALID_DATASETS):
    if ds not in [c[0] for c in configs]:
        configs.append((i, ds, 1))

# # knn relabel
# task = tasks[2]
# for i, ds, minibatch in configs[-n:-7]:
#     jobs.append(
#         add_log(
#             f"python run.py --dataset {ds} --minibatch_size {minibatch} {task} --shared_folder repro_{shared_date}",
#             ds,
#             task
#         )
#     )

# logreg relabel
task = tasks[3]
for i, ds, minibatch in configs:
    jobs.append(
        add_log(
            f"python run.py --dataset {ds} --minibatch_size {minibatch} {task} --shared_folder repro_{shared_date}",
            ds,
            task
        )
    )

# dream
i, ds, minibatch = configs[2]
for task in tasks[4:]:
    jobs.append(
        add_log(
            f"python run.py --dataset {ds} --minibatch_size {minibatch} {task} --shared_folder repro_{shared_date}",
            ds,
            task
        )
    )

# quartz
i, ds, minibatch = configs[14]
for task in tasks:
    jobs.append(
        add_log(
            f"python run.py --dataset {ds} --minibatch_size {minibatch} {task} --load_best_model_at_end False --shared_folder repro_{shared_date}",
            ds,
            task
        )
    )


if __name__ == "__main__":
    # usage: python run_friday.py 1,2,3,4,5,6,7
    if len(sys.argv) == 1:
        # default to all GPUs
        gpus = range(8)
    else:
        gpus = [int(gpu) for gpu in sys.argv[1].split(",")]

    for job in jobs:
        print(job)
    print()
    print(f"Running on GPUs: {gpus}")

    gpu_map(gpus, jobs)