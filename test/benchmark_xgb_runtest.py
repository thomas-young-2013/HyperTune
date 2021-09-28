"""
example cmdline:

python test/benchmark_xgb_runtest.py --datasets covtype --mths hyperband-n1 --rep 1 --start_id 0

python test/benchmark_xgb_runtest.py --datasets covtype --show_mode 1

"""
import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency
from utils import load_data, setup_exp, check_datasets, seeds

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str)
parser.add_argument('--mths', type=str)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--show_mode', type=int, default=0)
parser.add_argument('--runtime_limit', type=int)    # if you don't want to use default setup

args = parser.parse_args()
test_datasets = args.datasets.split(',')
mths = args.mths.split(',')
rep = args.rep
start_id = args.start_id
show_mode = args.show_mode

print(test_datasets)


def test_func(config, x_train, x_test, y_train, y_test):
    from tuner.xgb_model import XGBoost
    conf_dict = config.get_dictionary()
    model = XGBoost(**conf_dict, n_jobs=n_jobs, seed=47)
    model.fit(x_train, y_train)
    # test
    y_pred = model.predict(x_test)
    perf = balanced_accuracy_score(y_test, y_pred)
    return perf


if show_mode == 1:
    for dataset in test_datasets:
        # setup
        _, runtime_limit, _ = setup_exp(dataset, 1, 1, 1)
        if args.runtime_limit is not None:
            runtime_limit = args.runtime_limit
        for mth in mths:
            perfs = []
            dir_path = 'data/benchmark_xgb/%s-%d/%s/' % (dataset, runtime_limit, mth)
            for file in os.listdir(dir_path):
                if file.startswith('incumbent_test_perf_%s-%s-' % (mth, dataset)) and file.endswith('.pkl'):
                    with open(os.path.join(dir_path, file), 'rb') as f:
                        perf = pkl.load(f)
                    perfs.append(perf)
            m = np.mean(perfs).item()
            s = np.std(perfs).item()
            print(dataset, mth, perfs, u'%.4f\u00B1%.4f' % (m, s))
    exit()


check_datasets(test_datasets)
for dataset in test_datasets:
    # setup
    n_jobs, runtime_limit, _ = setup_exp(dataset, 4, 1, 1)
    if args.runtime_limit is not None:
        runtime_limit = args.runtime_limit
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset)
    for mth in mths:
        print('===== start test %s %s: rep=%d, n_jobs=%d' % (mth, dataset, rep, n_jobs))
        for i in range(start_id, start_id + rep):
            seed = seeds[i]

            dir_path = 'data/benchmark_xgb/%s-%d/%s/' % (dataset, runtime_limit, mth)
            for file in os.listdir(dir_path):
                if file.startswith('incumbent_new_record_%s-%s-%d-' % (mth, dataset, seed)) \
                        and file.endswith('.pkl'):
                    # load config
                    with open(os.path.join(dir_path, file), 'rb') as f:
                        record = pkl.load(f)
                    print(dataset, mth, seed, 'loaded!', record, flush=True)

                    # run test
                    config = record['configuration']
                    perf = test_func(config, x_train, x_test, y_train, y_test)
                    print(dataset, mth, seed, 'perf =', perf)

                    # save perf
                    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                    method_id = mth + '-%s-%d-%s' % (dataset, seed, timestamp)
                    perf_file_name = 'incumbent_test_perf_%s.pkl' % (method_id,)
                    with open(os.path.join(dir_path, perf_file_name), 'wb') as f:
                        pkl.dump(perf, f)
                    print(dir_path, perf_file_name, 'saved!', flush=True)
