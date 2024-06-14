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

date = "0611"
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

configs = [
    # ("boolq", 1),
    # ("anli-r2", 8),
    # ("cosmos_qa", 4),
    # ("mc_taco", 4),
    # ("sciq", 4),
    # ("paws", 16),
    # ("twitter-sentiment", 8),
    # ("wic", 8),
]

for ds in VALID_DATASETS:
    if ds not in [c[0] for c in configs]:
        configs.append((ds, 1))


for task in tasks:
    for ds, minibatch in configs:
        jobs.append(
            add_log(
                f"python run.py --dataset {ds} --minibatch_size {minibatch} {task} --shared_folder repro_{shared_date}",
                ds,
                task
            )
        )

print(len(jobs))

halfpoint = len(jobs) // 2

# ran on adam-ord
jobs = jobs[:halfpoint]

# ran successfully Monday night
jobs = [job for job in jobs if not "xent" in job]

print(len(jobs))

halfpoint = len(jobs) // 2

# CHANGE ME on shared-ord
jobs = jobs[:halfpoint]

print(len(jobs))

count = 0 
for job in jobs:
    if "strong2strong" in job:
        count += 3
    else:
        count += 1
    if "boolq" in job:
        print(count, job)
print(count)

# if __name__ == "__main__":
#     # usage: python run_friday.py 1,2,3,4,5,6,7
#     if len(sys.argv) == 1:
#         # default to all GPUs
#         gpus = range(8)
#     else:
#         gpus = [int(gpu) for gpu in sys.argv[1].split(",")]

#     for job in jobs:
#         print(job)
    # print()
    # print(f"Running on GPUs: {gpus}")

    # gpu_map(gpus, jobs)