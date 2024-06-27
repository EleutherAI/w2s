import argparse
import subprocess
import time

import schedule

parser = argparse.ArgumentParser()
parser.add_argument("cmds_file", type=str)
parser.add_argument("--delay-hours", type=float, default=0)
parser.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5,6,7")
args = parser.parse_args()
cmds_file = args.cmds_file
gpu_ids = list(map(int, args.gpu_ids.split(",")))
print(gpu_ids)
# List of commands to execute
commands = open(cmds_file).read().strip().split("\n")
time.sleep(args.delay_hours * 60 * 60)

recently_used_gpus = dict()


def check_gpu_memory():
    for gpu_index, last_used_time in list(recently_used_gpus.items()):
        if time.time() - last_used_time > 60 * 5:
            del recently_used_gpus[gpu_index]
    try:
        # Check GPU memory usage using nvidia-smi command
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("Failed to run nvidia-smi")
            return

        # Parse the output to get free memory values
        free_memory_list = [int(x) for x in result.stdout.strip().split("\n")]

        for i, free_memory in enumerate(free_memory_list):
            if free_memory > 40000 and i not in recently_used_gpus and i in gpu_ids:
                print(f"GPU {i} has {free_memory} MB free. Running the next command.")
                recently_used_gpus[i] = time.time()
                run_next_command(i)
                break
    except Exception as e:
        print(f"Error checking GPU memory: {e}")


def run_next_command(gpu_index):
    if commands:
        # Get the next command from the queue
        command = commands.pop(0)
        command = f"CUDA_VISIBLE_DEVICES={gpu_index} {command}"

        # Run the command as a subprocess
        subprocess.Popen(command, shell=True)
        print(f"Running command: {command} on GPU {gpu_index}")
    else:
        print("No more commands to run.")


# Schedule the GPU check every minute
schedule.every(5).seconds.do(check_gpu_memory)

print("Scheduler started. Checking GPU status every minute.")
while True:
    schedule.run_pending()
    time.sleep(1)
