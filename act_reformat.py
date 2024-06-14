from w2s.ds_registry import VALID_DATASETS
from tqdm import tqdm
import torch

datasets = VALID_DATASETS.copy()
# datasets.remove('quartz')
datasets.remove('amazon_polarity_gt')
datasets.remove('amazon_polarity_weak')

def shared_dir(ds):
    if ds == 'quartz':
        return 'repro_0613e'
    return 'repro_0610'

def act_dir(ds):
    return f"./results/{shared_dir(ds)}/{ds}/w2s/activations/"

def load_act(dataset):
    acts_dir = act_dir(dataset)
    act = torch.load(acts_dir + f"train.pt", map_location="cpu")

    return act

def save_act(dataset, act):
    acts_dir = act_dir(dataset)
    print(f"Saving activations to {acts_dir}")
    print(f"Activations shape: {act.shape}")
    for layer in range(len(act[0])):
        layer_act = act[:, layer]
        print(f"Layer {layer} shape: {layer_act.shape}")
        # copy
        layer_act = layer_act.clone().detach().cpu()
        #torch.save(layer_act, acts_dir + f"train_{layer}.pt")
        layer_act_np = layer_act.float().numpy()
        # save as numpy
        with open(acts_dir + f"train_{layer}.npy", 'wb') as f:
            layer_act_np.tofile(f)



for i, ds in enumerate(datasets):
    print(f"[{i}/{len(datasets)}] Loading activations for {ds}...", flush=True)
    act = load_act(ds)
    print("Activations loaded", flush=True)
    save_act(ds, act)
    print("Activations saved", flush=True)