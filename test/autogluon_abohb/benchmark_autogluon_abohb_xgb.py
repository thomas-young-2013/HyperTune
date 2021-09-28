"""
example cmdline:

python test/autogluon_abohb/benchmark_autogluon_abohb_xgb.py --R 27 --reduction_factor 3 --brackets 4 --num_cpus 16 --n_workers 8 --n_jobs 16 --timeout 10800 --dataset covtype --rep 1 --start_id 0

"""

import autogluon.core as ag
import os
import sys
import time
import logging
import yaml
import argparse
import numpy as np
import pickle as pkl
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency
from tuner.xgb_model import XGBoost
from test.utils import load_data, setup_exp, check_datasets, seeds


logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description='Runs autogluon to optimize the hyperparameters')
parser.add_argument('--num_trials', default=999999, type=int,
                    help='number of trial tasks. It is enough to either set num_trials or timout.')
parser.add_argument('--timeout', default=60, type=int,
                    help='runtime of autogluon in seconds. It is enough to either set num_trials or timout.')
parser.add_argument('--num_gpus', type=int, default=0,
                    help='number of GPUs available to a given trial.')
parser.add_argument('--num_cpus', type=int, default=16,
                    help='number of CPUs available to a given trial.')  # autogluon use this to infer n_workers
parser.add_argument('--hostfile', type=argparse.FileType('r'))
parser.add_argument('--store_results_period', type=int, default=100,
                    help='If specified, results are stored in intervals of '
                         'this many seconds (they are always stored at '
                         'the end)')
parser.add_argument('--scheduler', type=str, default='hyperband_promotion',
                    choices=['hyperband_stopping', 'hyperband_promotion'],
                    help='Asynchronous scheduler type. In case of doubt leave it to the default')
parser.add_argument('--reduction_factor', type=int, default=3,
                    help='Reduction factor for successive halving')
parser.add_argument('--brackets', type=int, default=4,
                    help='Number of brackets. Setting the number of brackets to 1 means '
                         'that we run effectively successive halving')
parser.add_argument('--min_resource_level', type=int, default=1,
                    help='Minimum resource level (i.e epochs) on which a configuration is evaluated on.')
parser.add_argument('--searcher', type=str, default='bayesopt',
                    choices=['random', 'bayesopt'],
                    help='searcher to sample new configurations')

parser.add_argument('--dataset', type=str)
parser.add_argument('--R', type=int, default=27)    # remember to set reduction_factor(eta) and brackets(s_max+1)
parser.add_argument('--n_workers', type=int)        # must set for saving results, but no use in autogluon
parser.add_argument('--n_jobs', type=int, default=15)    # xgb n_jobs
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()

algo_name = 'abohb_aws'
dataset = args.dataset
R = args.R
eta = args.reduction_factor
n_workers = args.n_workers  # Caution: must set for saving result to different dirs
rep = args.rep
start_id = args.start_id

runtime_limit = args.timeout
n_jobs = args.n_jobs

print(n_workers, dataset)
print(R)
for para in (R, n_workers):
    assert para is not None

# for autogluon
assert args.num_cpus >= n_jobs


@ag.args(  # config space
    n_estimators=ag.space.Int(lower=100, upper=1000, default=500),
    max_depth=ag.space.Int(lower=1, upper=12),
    learning_rate=ag.space.Real(lower=1e-3, upper=0.9, default=0.1, log=True),
    min_child_weight=ag.space.Real(lower=0, upper=10, default=1),
    subsample=ag.space.Real(lower=0.1, upper=1, default=1),
    colsample_bytree=ag.space.Real(lower=0.1, upper=1, default=1),
    gamma=ag.space.Real(lower=0, upper=10, default=0),
    reg_alpha=ag.space.Real(lower=0, upper=10, default=0),
    reg_lambda=ag.space.Real(lower=1, upper=10, default=1),
    epochs=R,   # max resource
)
def objective_function(args, reporter):
    s_max = int(log(args.epochs) / log(eta))
    iterate_r = [int(r) for r in np.logspace(0, s_max, s_max + 1, base=eta)]

    # load data
    lt = time.time()
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset)
    print('load data %s time: %.2fs' % (dataset, time.time() - lt))

    # Hyperparameters to be optimized
    params = dict()
    params['n_estimators'] = args.n_estimators
    params['max_depth'] = args.max_depth
    params['learning_rate'] = args.learning_rate
    params['min_child_weight'] = args.min_child_weight
    params['subsample'] = args.subsample
    params['colsample_bytree'] = args.colsample_bytree
    params['gamma'] = args.gamma
    params['reg_alpha'] = args.reg_alpha
    params['reg_lambda'] = args.reg_lambda

    total_resource = args.epochs
    for n_resource in iterate_r:
        t_eval_start = time.time()

        t0 = time.time()
        # sample train data. the test data after split is sampled train data
        if n_resource < total_resource:
            ratio = n_resource / total_resource
            print('sample data: ratio =', ratio, n_resource, total_resource, flush=True)
            _x, sample_x, _y, sample_y = train_test_split(x_train, y_train, test_size=ratio,
                                                          stratify=y_train, random_state=1)
        else:
            print('sample data: use full dataset', n_resource, total_resource, flush=True)
            sample_x, sample_y = x_train, y_train
        t1 = time.time()
        print('=== sample time = %.2fs. resource=%f/%f.' % (t1 - t0, n_resource, total_resource), flush=True)

        model = XGBoost(**params, n_jobs=n_jobs, seed=47)
        model.fit(sample_x, sample_y)

        t2 = time.time()
        print('=== train time = %.2fs. resource=%f/%f.' % (t2 - t1, n_resource, total_resource), flush=True)

        # evaluate on validation data
        y_pred = model.predict(x_val)
        perf = balanced_accuracy_score(y_val, y_pred)  # Caution: maximized

        t3 = time.time()
        print('predict time = %.2fs.' % (t3 - t2), flush=True)

        test_perf = 10000.0
        if n_resource == total_resource:
            y_test_pred = model.predict(x_test)
            test_perf = -balanced_accuracy_score(y_test, y_test_pred)  # minimize
            t4 = time.time()
            print('test time = %.2fs.' % (t4 - t3), flush=True)

        t_eval_end = time.time()
        eval_time = t_eval_end - t_eval_start
        print('Report %d/%d: config=%s, perf=%f, test_perf=%f.'
              % (n_resource, total_resource, params, perf, test_perf), flush=True)
        reporter(
            epoch=n_resource,
            performance=float(perf),    # Caution: maximized
            eval_time=eval_time,
            time_step=t_eval_end,
            test_perf=test_perf,
            **params,
        )


def evaluate(hyperband_type, method_id, seed, dir_path):
    def callback(training_history, start_timestamp, config_history, state_dict):
        # This callback function will be executed every time AutoGluon collected some new data.
        # In this example we will parse the training history into a .csv file and save it to disk, such that we can
        # for example plot AutoGluon's performance during the optimization process.
        # If you don't care about analyzing performance of AutoGluon online, you can also ignore this callback function and
        # just save the training history after AutoGluon has finished.
        import pandas as pd
        task_dfs = []

        # this function need to be changed if you return something else than accuracy
        def compute_error(df):
            return -df["performance"]  # Caution: minimize

        def compute_runtime(df, start_timestamp):
            return df["time_step"] - start_timestamp

        for task_id in training_history:
            task_df = pd.DataFrame(training_history[task_id])
            task_df = task_df.assign(task_id=task_id,
                                     runtime=compute_runtime(task_df, start_timestamp),
                                     error=compute_error(task_df),
                                     target_epoch=task_df["epoch"].iloc[-1])
            task_dfs.append(task_df)

        result = pd.concat(task_dfs, axis="index", ignore_index=True, sort=True)

        # re-order by runtime
        result = result.sort_values(by="runtime")

        # calculate incumbent best -- the cumulative minimum of the error.
        result = result.assign(best=result["error"].cummin())

        csv_path = os.path.join(dir_path, "result_%s.csv" % method_id)
        result.to_csv(csv_path)
        print('result saved to', csv_path)

    scheduler = ag.scheduler.HyperbandScheduler(objective_function,
                                                resource={'num_cpus': args.num_cpus, 'num_gpus': args.num_gpus},
                                                # Autogluon runs until it either reaches num_trials or time_out
                                                num_trials=args.num_trials,
                                                time_out=runtime_limit,
                                                # This argument defines the metric that will be maximized.
                                                # Make sure that you report this back in the objective function.
                                                reward_attr='performance',
                                                # The metric along we make scheduling decision. Needs to be also
                                                # reported back to AutoGluon in the objective function.
                                                time_attr='epoch',
                                                brackets=args.brackets,
                                                checkpoint=None,
                                                searcher=args.searcher,  # Defines searcher for new configurations
                                                search_options=dict(random_seed=seed, first_is_default=False),
                                                dist_ip_addrs=dist_ip_addrs,
                                                training_history_callback=callback,
                                                training_history_callback_delta_secs=args.store_results_period,
                                                reduction_factor=args.reduction_factor,
                                                type=hyperband_type,
                                                # defines the minimum resource level for Hyperband,
                                                # i.e the minimum number of epochs
                                                grace_period=args.min_resource_level,
                                                random_seed=seed,
                                                )
    # set config space seed
    scheduler.searcher.gp_searcher.configspace_ext.hp_ranges_ext.config_space.seed(seed)
    scheduler.searcher.gp_searcher.configspace_ext.hp_ranges.config_space.seed(seed)
    scheduler.searcher.gp_searcher.hp_ranges.config_space.seed(seed)
    # run
    scheduler.run()
    scheduler.join_jobs()

    # final training history
    training_history = scheduler.training_history
    print('len of final history: %d' % len(training_history), flush=True)
    return training_history


if __name__ == "__main__":
    # In case you want to run AutoGluon across multiple instances, you need to provide it with a list of
    # the IP addresses of the instances. Here, we assume that the IP addresses are stored in a .yaml file.
    # If you want to run AutoGluon on a single instance just pass None.
    # However, keep in mind that it will still parallelize the optimization process across multiple threads then.
    # If you really want to run it purely sequentially, set the num_cpus equal to the number of VCPUs of the machine.
    dist_ip_addrs = yaml.load(args.hostfile) if args.hostfile is not None else []
    del args.hostfile
    print("Got worker host IP addresses [{}]".format(dist_ip_addrs))

    if args.scheduler == "hyperband_stopping":
        hyperband_type = "stopping"
    elif args.scheduler == "hyperband_promotion":
        hyperband_type = "promotion"
    else:
        raise ValueError(args.scheduler)

    model_name = 'xgb'
    # setup
    for i in range(start_id, start_id + rep):
        seed = seeds[i]

        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        if R != 27:
            method_str = '%s-%d-n%d' % (algo_name, R, n_workers)
        else:
            method_str = '%s-n%d' % (algo_name, n_workers)
        method_id = method_str + '-%s-%d-%s' % (dataset, seed, timestamp)

        dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model_name, dataset, runtime_limit, method_str)
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except FileExistsError:
            pass

        # run
        global_start_time = time.time()
        training_history = evaluate(hyperband_type, method_id, seed, dir_path)

        # save
        file_name = 'history_%s.pkl' % (method_id,)
        save_item = (global_start_time, training_history)
        with open(os.path.join(dir_path, file_name), 'wb') as f:
            pkl.dump(save_item, f)
        print(dir_path, file_name, 'saved!', flush=True)

        if rep > 1:
            sleep_time = min(runtime_limit/5, 1800)
            print('sleep %.2fs now!' % sleep_time, flush=True)
            time.sleep(sleep_time)
