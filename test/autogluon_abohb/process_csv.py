"""
example cmdline:

python test/autogluon_abohb/process_csv.py --model xgb --dataset pokerhand --time 7200 --mths abohb_aws-n8 --R 27

"""
import argparse
import os
import numpy as np
import pickle as pkl
import pandas as pd

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
        if file.startswith('result_%s-%s-' % (mth, dataset)) and file.endswith('.csv'):
            new_file = 'new_record_' + file[7:-4] + '.pkl'
            # history.pkl has higher priority
            if new_file in os.listdir(dir_path):
                print('%s already exists. pass.' % new_file)
                continue

            df = pd.read_csv(os.path.join(dir_path, file), header='infer', sep=',')

            recorder = []
            for idx, row in df.iterrows():  # for row in df.itertuples(): getattr(row, 'col_name')
                epoch = row['epoch']
                if epoch < R:
                    continue
                elif epoch > R:
                    raise ValueError('please check R in settings.', R, mth, idx, row)

                # time_step = row['time_step']
                runtime = row['runtime']
                performance = row['performance']
                test_perf = row['test_perf'] if 'test_perf' in row else None
                eval_time = row['eval_time']

                if runtime > runtime_limit:
                    print('abandon record by runtime:', runtime, runtime_limit)
                    continue

                record = {
                    'time_consumed': eval_time * simulation_factor,
                    'configuration': None,  # todo
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
            with open(os.path.join(dir_path, new_file), 'wb') as f:
                pkl.dump(recorder, f)
            print('recorder len:', mth, len(recorder), df.shape[0])
