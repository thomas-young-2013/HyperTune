"""
example cmdline:

python test/benchmark_process_record.py --dataset covtype --old 10800 --new 10800 --R 27

"""
import argparse
import os
import numpy as np
import pickle as pkl


# step 1: cut off
def cut_off(model, dataset, mths, old_runtime_limit, new_runtime_limit):
    print('===== step 1: cut off')
    print(old_runtime_limit, new_runtime_limit)
    if old_runtime_limit != new_runtime_limit:
        for mth in mths:
            old_dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, old_runtime_limit, mth)
            new_dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, new_runtime_limit, mth)
            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)
            else:
                raise Exception('please checkout. new dir already exists!')
            for file in os.listdir(old_dir_path):
                if file.startswith('record_%s-%s-' % (mth, dataset)) and file.endswith('.pkl'):
                    with open(os.path.join(old_dir_path, file), 'rb') as f:
                        raw_recorder = pkl.load(f)
                    recorder = []
                    for record in raw_recorder:
                        if record['global_time'] > new_runtime_limit:
                            # print('abandon record by new_runtime_limit:', new_runtime_limit, mth, record)
                            continue
                        recorder.append(record)
                    # write new
                    with open(os.path.join(new_dir_path, file), 'wb') as f:
                        pkl.dump(recorder, f)
                    print('recorder len:', mth, len(recorder), len(raw_recorder))


# step 2: remove partial validation in hyperband
def remove_partial(model, dataset, mths, new_runtime_limit, R):
    print('===== step 2: remove part validation in hyperband')
    for mth in mths:
        new_dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, new_runtime_limit, mth)
        for file in os.listdir(new_dir_path):
            if file.startswith('record_%s-%s-' % (mth, dataset)) and file.endswith('.pkl'):
                with open(os.path.join(new_dir_path, file), 'rb') as f:
                    raw_recorder = pkl.load(f)
                recorder = []
                for record in raw_recorder:
                    if record.get('n_iteration') is not None:
                        if record['n_iteration'] < R:
                            if mth.startswith('random') or mth == 'smac':
                                print('error abandon record by n_iteration:', R, mth, record)
                            continue
                        if record['n_iteration'] > R:
                            raise ValueError('please check R in settings.', R, mth, record)
                    recorder.append(record)
                # write new
                new_file = 'new_' + file
                with open(os.path.join(new_dir_path, new_file), 'wb') as f:
                    pkl.dump(recorder, f)
                print('recorder len:', mth, len(recorder), len(raw_recorder))


# step 3: get incumbent config
def get_incumbent(model, dataset, mths, new_runtime_limit):
    print('===== step 3: get incumbent config')
    for mth in mths:
        new_dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, new_runtime_limit, mth)
        for file in os.listdir(new_dir_path):
            if file.startswith('new_record_%s-%s-' % (mth, dataset)) and file.endswith('.pkl'):
                with open(os.path.join(new_dir_path, file), 'rb') as f:
                    raw_recorder = pkl.load(f)
                incumbent_perf = np.inf
                incumbent_record = None
                for record in raw_recorder:
                    perf = record['return_info']['loss']
                    if perf < incumbent_perf:
                        incumbent_perf = perf
                        incumbent_record = record
                # write new
                new_file = 'incumbent_' + file
                with open(os.path.join(new_dir_path, new_file), 'wb') as f:
                    pkl.dump(incumbent_record, f)
                print(mth, file, incumbent_record)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--mths', type=str)
    parser.add_argument('--R', type=int, default=27)
    parser.add_argument('--old', type=int)
    parser.add_argument('--new', type=int)
    parser.add_argument('--model', type=str, default='xgb')

    args = parser.parse_args()
    dataset = args.dataset
    mths = args.mths.split(',')
    R = args.R
    old_runtime_limit = args.old
    new_runtime_limit = args.new
    model = args.model
    for para in (dataset, old_runtime_limit, new_runtime_limit):
        assert para is not None

    # process
    cut_off(model, dataset, mths, old_runtime_limit, new_runtime_limit)
    remove_partial(model, dataset, mths, new_runtime_limit, R)
    get_incumbent(model, dataset, mths, new_runtime_limit)
