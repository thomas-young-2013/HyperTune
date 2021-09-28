"""
run benchmark_process_record.py first to get new_record file

example cmdline:

python test/benchmark_plot.py --dataset covtype --R 27

"""
import argparse
import os
import time
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from utils import setup_exp, descending, create_plot_points

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


def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    color_list = ['purple', 'royalblue', 'green', 'brown', 'red', 'orange', 'yellowgreen', 'black', 'yellow']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x', 'd']

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]

    for name in m_list:
        if name.startswith('hyperband'):
            fill_values(name, 2)
        else:
            print('color not defined:', name)
            fill_values(name, 1)
    return color_dict, marker_dict


def get_mth_legend(mth):
    return mth


def plot_setup(_dataset):
    if _dataset == 'covtype':
        plt.ylim(-0.940, -0.880)
    elif _dataset == 'pokerhand':
        plt.ylim(-1.001, -0.951)
    elif _dataset.startswith('HIGGS'):
        plt.ylim(-0.756, -0.746)
    elif _dataset.startswith('hepmass'):
        plt.ylim(-0.8755, -0.8725)
    elif _dataset == 'cifar10-valid':
        plt.ylim(-91.65, -90.85)
    elif _dataset == 'cifar100':
        plt.ylim(-73.7, -70.7)
    elif _dataset == 'ImageNet16-120':
        plt.ylim(-47.0, -45.0)
    plt.xlim(0, runtime_limit)


print('start', dataset)
# setup
_, runtime_limit, _ = setup_exp(dataset, 1, 1, 1)
if args.runtime_limit is not None:
    runtime_limit = args.runtime_limit
plot_setup(dataset)
color_dict, marker_dict = fetch_color_marker(mths)
point_num = 300
lw = 2
markersize = 6
markevery = int(point_num / 10)
alpha = 0.15

plot_list = []
legend_list = []
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
            timestamp = [rec['global_time'] for rec in recorder]
            perf = descending([rec['return_info']['loss'] for rec in recorder])
            stats.append((timestamp, perf))
    x, m, s = create_plot_points(stats, 0, runtime_limit, point_num=point_num, default=default_value)
    result[mth] = (x, m, s)
    # plot
    plt.plot(x, m, lw=lw, label=get_mth_legend(mth),
             #color=color_dict[mth], marker=marker_dict[mth],
             markersize=markersize, markevery=markevery)
    #plt.fill_between(x, m - s, m + s, alpha=alpha, facecolor=color_dict[mth])

# calculate speedup
speedup_algo = 1
print('===== mth - baseline - speedup ===== speedup_algo =', speedup_algo)
for mth in mths:
    for baseline in mths:
        baseline_perf = result[baseline][1][-1]
        if speedup_algo == 1:   # algo 1
            baseline_time = None
            x, m, s = result[baseline]
            x, m, s = x.tolist(), m.tolist(), s.tolist()
            for xi, mi, si in zip(x, m, s):
                if mi <= baseline_perf:
                    baseline_time = xi
                    break
            assert baseline_time is not None
        else:
            raise ValueError
        x, m, s = result[mth]
        x, m, s = x.tolist(), m.tolist(), s.tolist()
        mth_time = baseline_time
        for xi, mi, si in zip(x, m, s):
            if mi <= baseline_perf:
                mth_time = xi
                break
        speedup = baseline_time / mth_time
        print("%s %s %.2f" % (mth, baseline, speedup))

# print last val perf
print('===== mth - last val perf =====')
for mth in mths:
    x, m, s = result[mth]
    m = m[-1]
    s = s[-1]
    perfs = None
    if dataset in ['cifar10', 'cifar10-valid', 'cifar100', 'ImageNet16-120']:
        print(dataset, mth, perfs, u'%.2f\u00B1%.2f' % (m, s))
    else:
        print(dataset, mth, perfs, u'%.4f\u00B1%.4f' % (m, s))

# show plot
plt.legend(loc='upper right')
plt.title("%s on %s" % (model, dataset), fontsize=16)
plt.xlabel("Wall Clock Time (sec)", fontsize=16)
plt.ylabel("Validation Error", fontsize=16)
plt.tight_layout()
plt.grid()
plt.show()
