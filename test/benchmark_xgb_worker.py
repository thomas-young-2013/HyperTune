"""
example cmdline:

python test/benchmark_xgb_worker.py --dataset covtype --R 27 --n_jobs 16 --parallel async --n_workers 8 --ip 127.0.0.1 --port 13579

"""

import os
import time
import sys
import argparse
import traceback
import pickle as pkl
import numpy as np
from functools import partial
from multiprocessing import Process

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency
from tuner.mq_mf_worker import mqmfWorker
from tuner.async_mq_mf_worker import async_mqmfWorker
from utils import load_data
from benchmark_xgb_utils import mf_objective_func


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_jobs', type=int, default=15)

parser.add_argument('--ip', type=str)
parser.add_argument('--port', type=int)
parser.add_argument('--n_workers', type=int)        # must set
parser.add_argument('--parallel', type=str, choices=['sync', 'async'])

args = parser.parse_args()
dataset = args.dataset
R = args.R
eta = args.eta
n_jobs = args.n_jobs                                # changed according to dataset

ip = args.ip
port = args.port
n_workers = args.n_workers
parallel_strategy = args.parallel

print(ip, port, n_jobs, n_workers, parallel_strategy)
print(R, eta)
for para in (ip, port, n_jobs, R, eta, n_workers, parallel_strategy):
    assert para is not None

pre_sample = False
run_test = True
assert parallel_strategy in ['sync', 'async']
if pre_sample and eta is None:
    raise ValueError('eta must not be None if pre_sample=True')


def worker_run(i):
    if parallel_strategy == 'sync':
        worker = mqmfWorker(objective_function, ip, port)
    elif parallel_strategy == 'async':
        worker = async_mqmfWorker(objective_function, ip, port)
    else:
        raise ValueError('Error parallel_strategy: %s.' % parallel_strategy)
    worker.run()
    print("Worker %d exit." % (i,))


x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset)

if pre_sample:
    raise NotImplementedError
else:
    objective_function = partial(mf_objective_func, total_resource=R,
                                 x_train=x_train, x_val=x_val, x_test=x_test,
                                 y_train=y_train, y_val=y_val, y_test=y_test,
                                 n_jobs=n_jobs, run_test=run_test)

worker_pool = []
for i in range(n_workers):
    w = Process(target=worker_run, args=(i,))
    worker_pool.append(w)
    w.start()
