from w2s.relabel import *

def make_grid(param_lists):
    keys = list(param_lists.keys())
    values = list(param_lists.values())
    for i, v in enumerate(values):
        if not isinstance(v, list):
            values[i] = [v]
    return [{k: v for k, v in zip(keys, vals)} for vals in product(*values)]

def relabel_sweep(cfg, dataset, root):
    method = cfg.method
    param_lists = cfg.to_dict()
    # remove RelabelSweepConfig fields
    for field in fields(RelabelSweepConfig):
        del param_lists[field.name]

    print(f"Scanning {dataset}")
    print(f"Loading dataset {dataset}")
    splits = load_and_process_dataset(dataset, split_sizes=dict(train=20_000, test=1_000))
    train, test = splits["train"], splits["test"]

    acts = torch.load(root / "ceil/acts.pt")
    wk = torch.load(root / "floor/preds/train.pt")
    gt = torch.tensor(train['hard_label'])

    acts = acts.float().cuda()
    wk = wk.float().to(acts.device)
    gt = gt.float().to(acts.device)

    if cfg.save_labels:
        acts_test = torch.load(root / "ceil/acts_test.pt")
        wk_test = torch.load(root / "floor/preds/test.pt")

        acts_test = acts_test.float().to(acts.device)
        wk_test = wk_test.float().to(acts.device)

    grid = make_grid(param_lists)

    if cfg.save_labels:
        assert len(grid) == 1, "Can only save labels for one set of parameters"

    method_fn, _ = METHODS[method]

    result_params = defaultdict(list)
    result_metrics = defaultdict(list)

    print(f"Running relabel sweep for {method} on {dataset}")
    for params in tqdm(grid):
        new_labels = method_fn(acts, wk, **params)
 
        if cfg.save_labels:
            save_path = root / 'floor/preds' / f'{method}_train.pt'
            torch.save(new_labels, save_path)
            print(f"Saved train predictions to {save_path}")
            new_labels_test = method_fn(acts_test, wk_test, **params)
            save_path = root / 'floor/preds' / f'{method}_test.pt'
            torch.save(new_labels_test, save_path)
            print(f"Saved test predictions to {save_path}")

        metrics = compute_metrics(new_labels, gt)
        for k, v in params.items():
            result_params[k].append(v)
        for k, v in metrics.items():
            result_metrics[k].append(v)

    baseline_metrics = compute_metrics(wk, gt)
    result_params = dict(result_params)
    result_metrics = dict(result_metrics)
    return asdict(RelabelResult(
        dataset=dataset,
        method=method,
        params=result_params,
        metrics=result_metrics,
        baseline_metrics=baseline_metrics,
    ))

def main(method):
    _, method_param_cls = METHODS[method]
    cfg = parse(method_param_cls)  # e.g. TopoParams


    out_dir = RESULT_DIR.format(user=cfg.outuser)
    out_file = Path(out_dir) / cfg.outfile
    
    # get existing results
    if out_file.exists():
        with open(out_file, 'r') as f:
            results = json.load(f)
    else:
        results = []

    datasets = get_datasets(cfg.resusers)

    print(f"Found {len(datasets)} datasets:")
    print(list(datasets.keys()))
    print("at paths:")
    for path in datasets.values():
        print(path)

    for dataset, root in datasets.items():
        result = relabel_sweep(cfg, dataset, root)
        results.append(result)
        with open(out_file, 'w') as f:
            json.dump(results, f)

if __name__ == "__main__":
    assert sys.argv[1] == '--method'
    method = sys.argv[2]
    print(f"Running relabel sweep for method {method}")
    main(method)