"""
example cmdline:

python test/nas_benchmarks/benchmark_nasbench201.py --dataset cifar10-valid --mths hyperband --R 27 --n_workers 4 --runtime_limit 86400 --rep 1 --start_id 0

"""

import os
import sys
import time
import argparse
import traceback
import numpy as np
import pickle as pkl
from functools import partial

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency
from test.nas_benchmarks.nasbench201_utils import load_nasbench201, get_nasbench201_configspace, objective_func
from test.nas_benchmarks.simulation_utils import run_in_parallel, run_async
from test.utils import seeds, timeit
from test.benchmark_process_record import remove_partial, get_incumbent
from tuner import mth_dict

dataset_choices = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120', 'all']
parser = argparse.ArgumentParser()
parser.add_argument('--mths', type=str, default='hyperband')
parser.add_argument('--dataset', type=str, default='all', choices=dataset_choices)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--n_workers', type=int)        # must set
parser.add_argument('--runtime_limit', type=int, default=0)
parser.add_argument('--time_limit_per_trial', type=int, default=999999)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--data_path', type=str, default='../nas_data/NAS-Bench-201-v1_1-096897.pth')

args = parser.parse_args()
mths = args.mths.split(',')
print("mths:", mths)
datasets = args.dataset.split(',')
if datasets[0] == 'all':
    datasets = ['cifar10-valid', 'cifar100', 'ImageNet16-120']
R = args.R
eta = args.eta
n_workers = args.n_workers  # Caution: must set for saving result to different dirs
time_limit_per_trial = args.time_limit_per_trial
rep = args.rep
start_id = args.start_id
data_path = args.data_path

print(n_workers)
print(R, eta)
for para in (R, eta, n_workers):
    assert para is not None

for algo_name in mths:
    assert algo_name in mth_dict.keys()


def evaluate_simulation(algo_class, algo_kwargs, method_id, n_workers, seed, parallel_strategy):
    print(method_id, n_workers, seed)

    assert parallel_strategy in ['sync', 'async']

    if parallel_strategy == 'sync':
        algo_class.run_in_parallel = run_in_parallel
    elif parallel_strategy == 'async':
        algo_class.run = run_async
    while True:
        try:
            port = 13579 + np.random.RandomState(int(time.time() * 10000 % 10000)).randint(2000)
            print('port =', port)
            algo = algo_class(
                objective_func=objective_function,  # must set for simulation
                config_space=cs,
                R=R,
                eta=eta,
                random_state=seed,
                method_id=method_id,
                restart_needed=True,
                time_limit_per_trial=time_limit_per_trial,
                runtime_limit=runtime_limit,
                port=port,
                **algo_kwargs,
            )
        except EOFError:
            print('EOFError: try next port.')
        else:
            break
    algo.n_workers = n_workers  # must set for simulation
    algo.run()
    try:
        algo.logger.info('===== bracket status: %s' % algo.get_bracket_status(algo.bracket))
    except Exception as e:
        pass
    try:
        algo.logger.info('===== brackets status: %s' % algo.get_brackets_status(algo.brackets))
    except Exception as e:
        pass
    return algo.recorder


with timeit('load nasbench201'):
    model_name = 'nasbench201'
    cs = get_nasbench201_configspace()
    api = load_nasbench201(path=data_path)

with timeit('%s all' % datasets):
    for dataset in datasets:
        # set runtime_limit
        if args.runtime_limit == 0:
            if dataset in ['cifar10', 'cifar10-valid']:
                runtime_limit = 86400
            elif dataset == 'cifar100':
                runtime_limit = 172800
            elif dataset == 'ImageNet16-120':
                runtime_limit = 432000
            else:
                raise ValueError
        else:
            runtime_limit = args.runtime_limit

        # set dataset
        objective_function = partial(objective_func, total_resource=R, eta=eta, api=api, dataset=dataset)

        with timeit('%s all' % dataset):
            for algo_name in mths:
                with timeit('%s %d %d' % (algo_name, start_id, rep)):
                    mth_info = mth_dict[algo_name]
                    if len(mth_info) == 2:
                        algo_class, parallel_strategy = mth_info
                        algo_kwargs = dict()
                    elif len(mth_info) == 3:
                        algo_class, parallel_strategy, algo_kwargs = mth_info
                    else:
                        raise ValueError('error mth info: %s' % mth_info)

                    from tuner.mq_random_search import mqRandomSearch
                    from tuner.mq_bo import mqBO
                    if algo_class in (mqRandomSearch, mqBO):
                        print('set algo_class n_workers:', n_workers)
                        algo_kwargs['n_workers'] = n_workers

                    print('===== start eval %s: rep=%d, runtime_limit=%d, time_limit_per_trial=%d'
                          % (dataset, rep, runtime_limit, time_limit_per_trial))
                    for i in range(start_id, start_id + rep):
                        seed = seeds[i]

                        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                        if R != 27:
                            method_str = '%s-%d-n%d' % (algo_name, R, n_workers)
                        else:
                            method_str = '%s-n%d' % (algo_name, n_workers)
                        method_id = method_str + '-%s-%d-%s' % (dataset, seed, timestamp)

                        with timeit('%d %s' % (i, method_id)):
                            recorder = evaluate_simulation(
                                algo_class, algo_kwargs, method_id, n_workers, seed, parallel_strategy
                            )

                        dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model_name, dataset, runtime_limit, method_str)
                        file_name = 'record_%s.pkl' % (method_id,)
                        try:
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)
                        except FileExistsError:
                            pass
                        with open(os.path.join(dir_path, file_name), 'wb') as f:
                            pkl.dump(recorder, f)
                        print(dir_path, file_name, 'saved!', flush=True)

                    try:
                        remove_partial(model_name, dataset, [method_str], runtime_limit, R)
                        get_incumbent(model_name, dataset, [method_str], runtime_limit)
                    except Exception as e:
                        print('benchmark process record failed: %s' % (traceback.format_exc(),))
