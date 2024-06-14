#!/bin/bash

time python run_eight.py --s2s_iters 2 --run_name repro_0606_strong2strong
time python run_eight.py --probe_relabel --probe knn --run_name repro_0606_probe_knn
time python run_eight.py --probe_relabel --probe logreg --run_name repro_0606_probe_logreg
time python run_eight.py --probe_filter --probe knn --run_name repro_0606_filter_knn
time python run_eight.py --probe_filter --probe logreg --run_name repro_0606_filter_logreg
time python run_eight.py --probe_filter --probe topo --run_name repro_0606_filter_topo
time python run_eight.py --loss window --radius midweak --run_name repro_0606_window_mid
time python run_eight.py --loss entropy --run_name repro_0606_entropy
time python run_eight.py --loss xent --run_name repro_0606_xent