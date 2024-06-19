# Weak-to-Strong Generalization

Source code for experiments from [this blog post](https://blog.eleuther.ai/weak-to-strong/), based in part on [openai/weak-to-strong](https://github.com/openai/weak-to-strong).

## Installation

`pip install -e .`

If you run into problems, try installing inside a conda or venv environment.

## Running experiments

Basic invocation: 

`python run.py --dataset sciq --run_name my_run`

List of datasets: `from w2s.ds_registry import VALID_DATASETS`

Additional args to reproduce blog post experiments:

```sh
--loss xent
--s2s_iters 2
--probe_relabel --probe knn
--probe_relabel --probe logreg
--probe_filter --probe knn
--probe_filter --probe logreg
--probe_filter --probe topo
--loss window --radius midweak
--loss entropy
```

There is `--help` available via `simpleparsing`. For individual loss functions and probes, try e.g. `python run.py --probe topo --help`.

Defaults are set in `sft_config.py`, `probe.py`, and `loss.py`. 

LoRA is on by default (rank 8). Pass `--disable_lora` to switch it off, although this is somewhat untested. For architectures other than Llama, Mistral, and Qwen, you will need to set `ModelConfig.lora_modules` in the arguments to `w2s.sft.train()`.

## Output and shared folders

Strong student results are stored in `./results/[run_name]/[dataset]/`. (You can set a different `--run_name` per experiment so that they don't overwrite each other.)

Basic metrics, like test AUC and accuracy, are in `w2s/results.json`. `wandb` is used for detailed logging if available.

Floor and ceiling results, weak supervisor predictions, and activations are stored in a shared folder so that they can be reused across experiments. By default this is `./results/[shared_folder]/[dataset]/`; the default `--shared_folder` is `shared`. You should change this if you change the weak or strong model, or anything else about the weak model training setup.

## Troubleshooting

Llama 3 is gated, see [here](https://huggingface.co/docs/hub/models-gated#access-gated-models-as-a-user) for details.

Loss and probe parameters are set from the CLI via `simpleparsing` [subgroups](https://github.com/lebrice/SimpleParsing/blob/master/examples/subgroups/README.md).