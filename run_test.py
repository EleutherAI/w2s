from w2s.gpu_pool import gpu_map

jobs = [
    "sleep 1; echo 'Job 1 done'",
    "sleep 1; echo 'Job 2 done'",
    "sleep 1; echo 'Job 3 done'",
]

if __name__ == "__main__":
    gpus = range(8)
    gpu_map(gpus, jobs)