from multiprocessing import Manager, Process
import subprocess

# Function that runs the job on a GPU
def run_on_gpu(gpu: int, job: str):
    print(f"Starting on GPU {gpu}: {job}")
    command = f"CUDA_VISIBLE_DEVICES={gpu} {job}"
    subprocess.run(command, shell=True, check=True)
    print(f"Finished on GPU {gpu}: {job}")

# Worker function that gets jobs and runs them on a specific GPU
def worker(gpu, jobs, lock):
    while True:
        with lock:
            if not jobs:
                print(f"GPU {gpu} has no more jobs.")
                return  # No more jobs to process
            job = jobs.pop(0)
        
        run_on_gpu(gpu, job)

def gpu_map(gpus, jobs):
    # Create a shared job list and a lock
    manager = Manager()
    jobs = manager.list(jobs)
    lock = manager.Lock()

    # Create and start worker processes, each assigned to a specific GPU
    processes = []
    for gpu in gpus:
        p = Process(target=worker, args=(gpu, jobs, lock))
        processes.append(p)
        p.start()

    # Wait for all worker processes to finish
    for p in processes:
        p.join()

    print("All jobs finished.")
