import subprocess
from multiprocessing import Process
from sys import argv

# Define the datasets and respective GPU ids
# list of tuples with dataset name and minibatch size
configs_A = [
    ("boolq", 2),
    ("anli-r2", 8),
    ("cosmos_qa", 4),
    ("mc_taco", 4),
    ("sciq", 4),
    ("paws", 16),
    ("twitter-sentiment", 8),
    ("wic", 8),
]

ds_B = [
    'cola',
    'dream',
    'ethics-deontology',
    'ethics-justice',
    'ethics-virtue',
    'ethics-utilitarianism',
    'hellaswag',
    'multirc',
]

ds_C = [
    'openbookqa',
    'quail',
    'quartz',
    'social_i_qa',
    'sst2',
    'sciq_with_support',
    'anthropic_hh',
    'amazon_polarity',
 ]

configs_B = [(ds, 8) for ds in ds_B]
configs_C = [(ds, 8) for ds in ds_C]

middles = [
    "--loss xent",
    "--loss logconf --logconf_warmup_steps 80",
    "--loss logconf",
    "--loss entropy --logconf_warmup_steps 80",
    "--loss entropy",
    "--loss window --radius midweak",
    "--loss window --radius .05",
    "--loss window --radius .15",
    "--loss window --radius .3",
]

# Define the base command
base_command = (
    "CUDA_VISIBLE_DEVICES={gpu_id} "
    "python run.py "
    "--dataset {dataset} "
    "--weak_model_name Qwen/Qwen1.5-0.5B "
    "--strong_model_name meta-llama/Meta-Llama-3-8B "
    "--n_epochs 4 "
    "--n_train 10_000 "
    "--n_val 1000 "
    "--n_test 5_000 "
    "--n_predict 0 "
    "--eval_every 100 "
    "--save_every 100 "
    "--save_total_limit 1 "
    "{middle} "
    "--minibatch_size {minibatch_size} "
    "--weak_lr 5e-4 "
    "--strong_lr 8e-5 "
    '--run_name "weekend_{m}_{c}" '
)


def run_command(command):
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    for m, middle in enumerate(middles):
        for c, configs in enumerate([configs_A, configs_B, configs_C]):        
            # List to hold processes
            processes = []
            
            gpu_ids = range(len(configs))

            # Loop over datasets and gpu_ids
            for (dataset, minibatch_size), gpu_id in zip(configs, gpu_ids):
                command = base_command.format(
                    gpu_id=gpu_id, dataset=dataset,
                    middle=middle,
                    minibatch_size=minibatch_size,
                    m=m, c=c,
                )
                print(f"Running command: {command}")  # Debug print
                p = Process(target=run_command, args=(command,))
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join()
