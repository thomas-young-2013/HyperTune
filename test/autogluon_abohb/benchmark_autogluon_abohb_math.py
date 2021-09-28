"""
example cmdline:

python test/autogluon_abohb/benchmark_autogluon_abohb_math.py --R 27 --reduction_factor 3 --brackets 4 --num_cpus 16 --n_workers 8 --timeout 1080 --dataset hartmann --noise_alpha 1000 --rep 1 --start_id 0

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

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency
from test.utils import seeds
from test.math_benchmarks.math_utils import get_math_obj_func_cs


logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description='Runs autogluon to optimize the hyperparameters')
parser.add_argument('--num_trials', default=999999, type=int,
                    help='number of trial tasks. It is enough to either set num_trials or timout.')
parser.add_argument('--timeout', default=0, type=int,
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

dataset_choices = ['hartmann']
parser.add_argument('--dataset', type=str, default='hartmann', choices=dataset_choices)
parser.add_argument('--R', type=int, default=27)    # remember to set reduction_factor(eta) and brackets(s_max+1)
parser.add_argument('--n_workers', type=int)        # must set for saving results, but no use in autogluon
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--simulation_factor', type=int, default=1)  # simulation sleep time factor

parser.add_argument('--noise_alpha', type=float, default=1.0)

args = parser.parse_args()

algo_name = 'abohb_aws'
model_name = 'math'
dataset = args.dataset
R = args.R
eta = args.reduction_factor
n_workers = args.n_workers  # Caution: must set for saving result to different dirs
rep = args.rep
start_id = args.start_id

simulation_factor = args.simulation_factor
print("simulation_factor:", simulation_factor)
# set runtime_limit
runtime_limit = args.timeout
print("runtime_limit:", runtime_limit)

noise_alpha = args.noise_alpha
print("noise_alpha:", noise_alpha)

print(n_workers, dataset)
print(R)
for para in (R, n_workers):
    assert para is not None


def get_cs_dict(problem_str):
    if problem_str == 'hartmann':
        cs_dict = dict(
            x0=ag.space.Real(lower=0, upper=1, default=0.5),
            x1=ag.space.Real(lower=0, upper=1, default=0.5),
            x2=ag.space.Real(lower=0, upper=1, default=0.5),
            x3=ag.space.Real(lower=0, upper=1, default=0.5),
            x4=ag.space.Real(lower=0, upper=1, default=0.5),
            x5=ag.space.Real(lower=0, upper=1, default=0.5),
            epochs=R,  # max resource
        )
        return cs_dict
    else:
        raise ValueError


@ag.args(**get_cs_dict(problem_str=dataset))    # config space
def objective_function(args, reporter):
    s_max = int(log(args.epochs) / log(eta))
    iterate_r = [int(r) for r in np.logspace(0, s_max, s_max + 1, base=eta)]

    # Hyperparameters to be optimized
    config = dict()
    if dataset == 'hartmann':
        config['x0'] = args.x0
        config['x1'] = args.x1
        config['x2'] = args.x2
        config['x3'] = args.x3
        config['x4'] = args.x4
        config['x5'] = args.x5
    else:
        raise ValueError

    last_train_time = 0.0
    total_resource = args.epochs
    for n_resource in iterate_r:
        t_eval_start = time.time()

        extra_conf = dict(initial_run=True)
        result_dict = math_obj_func(config, n_resource, extra_conf)
        perf = -result_dict['objective_value']  # Caution: maximize
        train_time = result_dict['elapsed_time']

        # simulation sleep
        real_sleep_time = train_time - last_train_time
        if real_sleep_time < 0:
            print('Error sleep time=%.2fs! train_time=%.2fs, last_train_time=%.2fs. config=%s'
                  % (real_sleep_time, train_time, last_train_time, config))
            real_sleep_time = 0
        last_train_time = train_time
        print('start sleeping %.2f / %d seconds' % (real_sleep_time, simulation_factor))
        time.sleep(real_sleep_time / simulation_factor)

        t_eval_end = time.time()
        eval_time = t_eval_end - t_eval_start
        print('Report %d/%d: config=%s, perf=%f'
              % (n_resource, total_resource, config, perf), flush=True)
        reporter(
            epoch=n_resource,
            performance=float(perf),    # Caution: maximized
            eval_time=eval_time,
            time_step=t_eval_end,
            **config,
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

    # setup
    for i in range(start_id, start_id + rep):
        seed = seeds[i]

        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        if R != 27:
            method_str = '%s-%d-n%d' % (algo_name, R, n_workers)
        else:
            method_str = '%s-n%d' % (algo_name, n_workers)
        method_id = method_str + '-%s-%.2f-%d-%s' % (dataset, noise_alpha, seed, timestamp)

        dir_path = 'data/benchmark_%s/%s-%.2f-%d/%s/' % (model_name, dataset, noise_alpha, runtime_limit, method_str)
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except FileExistsError:
            pass

        rng = np.random.RandomState(seed)
        problem_kwargs = dict()
        math_obj_func, _ = get_math_obj_func_cs(R, eta, dataset, noise_alpha, rng, problem_kwargs)

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
            sleep_time = 30.0
            print('sleep %.2fs now!' % sleep_time, flush=True)
            time.sleep(sleep_time)
