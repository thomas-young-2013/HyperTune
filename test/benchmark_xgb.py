"""
example cmdline:

python test/benchmark_xgb.py --mth hyperband --datasets spambase --R 27 --n_jobs 4 --n_workers 1 --runtime_limit 60 --rep 1 --start_id 0

"""

import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency
from tuner import mth_dict
from benchmark_xgb_utils import run_exp

parser = argparse.ArgumentParser()
parser.add_argument('--mth', type=str)
parser.add_argument('--datasets', type=str)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_jobs', type=int, default=4)

parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=0)
parser.add_argument('--n_workers', type=int)        # must set
parser.add_argument('--max_local_workers', type=int, default=8)

parser.add_argument('--runtime_limit', type=int, default=60)
parser.add_argument('--time_limit_per_trial', type=int, default=600)

parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
algo_name = args.mth
test_datasets = args.datasets.split(',')
print("datasets num=", len(test_datasets))
R = args.R
eta = args.eta
n_jobs = args.n_jobs                                # changed according to dataset

ip = args.ip
port = args.port
n_workers = args.n_workers  # Caution: must set for saving result to different dirs
max_local_workers = args.max_local_workers

runtime_limit = args.runtime_limit                  # changed according to dataset
time_limit_per_trial = args.time_limit_per_trial    # changed according to dataset

rep = args.rep
start_id = args.start_id

print(ip, port, n_jobs, n_workers, test_datasets, algo_name)
print(R, eta)
print(runtime_limit)
for para in (algo_name, ip, port, n_jobs, R, eta, n_workers, runtime_limit):
    assert para is not None

mth_info = mth_dict[algo_name]
if len(mth_info) == 2:
    algo_class, parallel_strategy = mth_info
    algo_kwargs = dict()
elif len(mth_info) == 3:
    algo_class, parallel_strategy, algo_kwargs = mth_info
else:
    raise ValueError('error mth info: %s' % mth_info)
# objective_func, config_space, random_state, method_id, runtime_limit, time_limit_per_trial, ip, port
# are filled in run_exp()
algo_kwargs['R'] = R
algo_kwargs['eta'] = eta
algo_kwargs['restart_needed'] = True

from tuner.mq_random_search import mqRandomSearch
from tuner.mq_bo import mqBO
if algo_class in (mqRandomSearch, mqBO):
    print('set algo_class n_workers:', n_workers)
    algo_kwargs['n_workers'] = n_workers

run_exp(test_datasets, algo_class, algo_kwargs, algo_name, n_workers, parallel_strategy,
        R, n_jobs, runtime_limit, time_limit_per_trial, start_id, rep, ip, port,
        eta=eta, pre_sample=False, run_test=True, max_local_workers=max_local_workers)
