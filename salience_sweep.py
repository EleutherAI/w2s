# python salience.py results/amazon_polarity_title_only_both_amplified 2000 200 --use_weak_label 
#  --eval_steps 10 --save_steps 10 --num_train_epochs 1 --per_device_train_batch_size 1 
#  --per_device_eval_batch_size 3 --gradient_accumulation_steps 32
base_command = (
    "python salience.py "
    "{weak_ds_path} "
    "2000 200 "
    "--eval_steps 10 "
    "--save_steps 10 "
    "--quantize "
    "--num_train_epochs 1 "
    "--per_device_train_batch_size 1 "
    "--per_device_eval_batch_size 3 "
    "--gradient_accumulation_steps 32 "
)

for ds_name in ["amazon_polarity_title_only", "ethics_deontology_excuse_only", "sciq_support_contains"]:
    for prompt in ["weak_amplified", "both_amplified", "neither_amplified", "gt_amplified"]:
        weak_ds_path = f"results/{ds_name}_{prompt}"
        cmd = base_command.format(weak_ds_path=weak_ds_path)
        print(cmd)
        cmd += " --use_weak_label "
        print(cmd)
