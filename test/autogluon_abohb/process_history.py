"""
example cmdline:

python test/autogluon_abohb/process_history.py --model xgb --dataset pokerhand --time 7200 --mths abohb_aws-n8 --R 27

"""
import argparse
import os
import numpy as np
import pickle as pkl
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--mths', type=str, default='abohb_aws-n8')
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--time', type=int)
parser.add_argument('--model', type=str, default='xgb')
parser.add_argument('--simulation_factor', type=int, default=1)  # simulation sleep time factor for nasbench

args = parser.parse_args()
dataset = args.dataset
mths = args.mths.split(',')
R = args.R
runtime_limit = args.time
model = args.model
simulation_factor = args.simulation_factor
if model not in ('nasbench101', 'nasbench201', 'math'):
    simulation_factor = 1
else:
    print('simulation factor:', simulation_factor)
for para in (dataset, runtime_limit):
    assert para is not None


for mth in mths:
    dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, runtime_limit, mth)
    for file in os.listdir(dir_path):
        if file.startswith('history_%s-%s-' % (mth, dataset)) and file.endswith('.pkl'):
            with open(os.path.join(dir_path, file), 'rb') as f:
                save_item = pkl.load(f)
                if isinstance(save_item, tuple):
                    start_time, history = save_item
                elif isinstance(save_item, OrderedDict):
                    print('Warning: no start time in history file.')
                    history = save_item
                    start_time = history['0'][0]['time_step'] - 60
                else:
                    raise ValueError('Unknown save type: %s' % (type(save_item)))

            recorder = []
            cnt = 0
            for history_list in history.values():
                for history_dict in history_list:
                    cnt += 1
                    epoch = history_dict.pop('epoch')
                    if epoch < R:
                        continue
                    elif epoch > R:
                        raise ValueError('please check R in settings.', R, mth, history_list)

                    time_step = history_dict.pop('time_step')
                    performance = history_dict.pop('performance')
                    test_perf = history_dict.pop('test_perf', None)
                    eval_time = history_dict.pop('eval_time')
                    history_dict.pop('terminated')
                    history_dict.pop('bracket')

                    runtime = (time_step - start_time)
                    if runtime > runtime_limit:
                        print('abandon record by runtime:', runtime, runtime_limit)
                        continue

                    record = {
                        'time_consumed': eval_time * simulation_factor,
                        'configuration': history_dict,
                        'global_time': runtime * simulation_factor,
                        'n_iteration': epoch,
                        'return_info': {
                            'loss': -performance,    # minimize
                            'test_perf': test_perf,  # already processed
                        },
                    }
                    recorder.append(record)

            recorder.sort(key=lambda rec: rec['global_time'])
            # write new
            new_file = 'new_record_' + file[8:]
            with open(os.path.join(dir_path, new_file), 'wb') as f:
                pkl.dump(recorder, f)
            print('recorder len:', mth, len(recorder), cnt)
