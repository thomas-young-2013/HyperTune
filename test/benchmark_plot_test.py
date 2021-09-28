"""
run benchmark_process_record.py first to get new_record file

example cmdline:

python test/benchmark_plot_test.py --dataset covtype --R 27

"""
import argparse
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from utils import setup_exp, create_plot_points


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--mths', type=str)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--runtime_limit', type=int)    # if you don't want to use default setup
parser.add_argument('--model', type=str, default='xgb')
parser.add_argument('--default_value', type=float, default=0.0)

args = parser.parse_args()
dataset = args.dataset
mths = args.mths.split(',')
R = args.R
model = args.model
default_value = args.default_value

print('start', dataset)
# setup
_, runtime_limit, _ = setup_exp(dataset, 1, 1, 1)
if args.runtime_limit is not None:
    runtime_limit = args.runtime_limit
point_num = 300


result = dict()
for mth in mths:
    stats = []
    dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, runtime_limit, mth)
    for file in os.listdir(dir_path):
        if file.startswith('new_record_%s-%s-' % (mth, dataset)) and file.endswith('.pkl'):
            with open(os.path.join(dir_path, file), 'rb') as f:
                raw_recorder = pkl.load(f)
            recorder = []
            for record in raw_recorder:
                # if record.get('n_iteration') is not None and record['n_iteration'] < R:
                #     print('error abandon record by n_iteration:', R, mth, record)
                #     continue
                if record['global_time'] > runtime_limit:
                    print('abandon record by runtime_limit:', runtime_limit, mth, record)
                    continue
                recorder.append(record)
            recorder.sort(key=lambda rec: rec['global_time'])
            # print([(rec['global_time'], rec['return_info']['loss']) for rec in recorder])
            print('new recorder len:', mth, len(recorder), len(raw_recorder))

            best_val_perf = recorder[0]['return_info']['loss']
            timestamps = [recorder[0]['global_time']]
            test_perfs = [recorder[0]['return_info'].get('test_perf', None)]
            if test_perfs[0] is None:
                raise ValueError('%s\n%s does not have test_perf!' % (recorder[0], mth))
            for rec in recorder[1:]:
                val_perf = rec['return_info']['loss']
                if val_perf < best_val_perf:
                    best_val_perf = val_perf
                    timestamps.append(rec['global_time'])
                    test_perfs.append(rec['return_info']['test_perf'])
            stats.append((timestamps, test_perfs))
    x, m, s = create_plot_points(stats, 0, runtime_limit, point_num=point_num, default=default_value)
    result[mth] = (x, m, s)

# print last test perf
print('===== mth - last test perf =====')
for mth in mths:
    x, m, s = result[mth]
    m = m[-1]
    s = s[-1]
    perfs = None
    if dataset in ['cifar10', 'cifar10-valid', 'cifar100', 'ImageNet16-120']:
        print(dataset, mth, perfs, u'%.2f\u00B1%.2f' % (m, s))
    else:
        print(dataset, mth, perfs, u'%.4f\u00B1%.4f' % (m, s))

