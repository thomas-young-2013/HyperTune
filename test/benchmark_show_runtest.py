"""
example cmdline:

python test/benchmark_show_runtest.py --model xgb --datasets covtype --mths hyperband-n8


"""
import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency
from utils import setup_exp

default_mths = 'random-n8,hyperband-n8'

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str)
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--runtime_limit', type=int)    # if you don't want to use default setup
parser.add_argument('--model', type=str, default='xgb')
parser.add_argument('--std_scale', type=float, default=0.5)

args = parser.parse_args()
test_datasets = args.datasets.split(',')
mths = args.mths.split(',')
model = args.model
std_scale = args.std_scale

print(test_datasets)
if std_scale != 1:
    print('=== Caution: std_scale:', std_scale)


for dataset in test_datasets:
    # setup
    _, runtime_limit, _ = setup_exp(dataset, 1, 1, 1)
    if args.runtime_limit is not None:
        runtime_limit = args.runtime_limit
    for mth in mths:
        perfs = []
        dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, runtime_limit, mth)
        for file in os.listdir(dir_path):
            if file.startswith('incumbent_test_perf_%s-%s-' % (mth, dataset)) and file.endswith('.pkl'):
                with open(os.path.join(dir_path, file), 'rb') as f:
                    perf = pkl.load(f)
                perfs.append(perf)
        m = np.mean(perfs).item()
        s = np.std(perfs).item()
        if dataset in ['cifar10-valid', 'cifar100', 'ImageNet16-120', 'penn']:
            print(dataset, mth, perfs, u'%.2f\u00B1%.2f' % (m, s))
        else:
            print(dataset, mth, perfs, u'%.4f\u00B1%.4f' % (m, s))
