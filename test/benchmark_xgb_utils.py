import os
import time
import traceback
import pickle as pkl
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from multiprocessing import Process, Manager

from tuner.mq_mf_worker import mqmfWorker
from tuner.async_mq_mf_worker import async_mqmfWorker
from tuner.xgb_model import XGBoost
from utils import load_data, setup_exp, check_datasets, seeds
from benchmark_process_record import remove_partial, get_incumbent


MAX_LOCAL_WORKERS = 8


def mf_objective_func(config, n_resource, extra_conf,
                      total_resource, x_train, x_val, x_test, y_train, y_val, y_test,
                      n_jobs, run_test=True):
    print('objective extra conf:', extra_conf)
    params = config.get_dictionary()

    t0 = time.time()
    # sample train data. the test data after split is sampled train data
    if n_resource < total_resource:
        ratio = n_resource / total_resource
        print('sample data: ratio =', ratio, n_resource, total_resource)
        _x, sample_x, _y, sample_y = train_test_split(x_train, y_train, test_size=ratio,
                                                      stratify=y_train, random_state=1)
    else:
        print('sample data: use full dataset', n_resource, total_resource)
        sample_x, sample_y = x_train, y_train
    t1 = time.time()
    print('sample time = %.2fs. resource=%f/%f.' % (t1 - t0, n_resource, total_resource))

    model = XGBoost(**params, n_jobs=n_jobs, seed=47)
    model.fit(sample_x, sample_y)

    t2 = time.time()
    print('train time = %.2fs. resource=%f/%f.' % (t2 - t1, n_resource, total_resource))

    # evaluate on validation data
    y_pred = model.predict(x_val)
    perf = -balanced_accuracy_score(y_val, y_pred)  # minimize

    t3 = time.time()
    print('predict time = %.2fs.' % (t3 - t2))

    test_perf = None
    if run_test and n_resource == total_resource:
        y_test_pred = model.predict(x_test)
        test_perf = -balanced_accuracy_score(y_test, y_test_pred)  # minimize
        t4 = time.time()
        print('test time = %.2fs.' % (t4 - t3))

    result = dict(
        objective_value=perf,
        early_stop=False,  # for deep learning
        ref_id=None,
        test_perf=test_perf,
    )
    return result


def evaluate_parallel(algo_class, algo_kwargs, method_id, n_workers, dataset, seed, ip, port,
                      parallel_strategy, n_jobs, R, eta=None, pre_sample=False, run_test=True):
    print(method_id, n_workers, dataset, seed)
    if port == 0:
        port = 13579 + np.random.RandomState(int(time.time() * 10000 % 10000)).randint(2000)
    print('ip=', ip, 'port=', port)
    assert parallel_strategy in ['sync', 'async']
    if pre_sample and eta is None:
        raise ValueError('eta must not be None if pre_sample=True')

    x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset)

    if pre_sample:
        raise NotImplementedError
    else:
        objective_function = partial(mf_objective_func, total_resource=R,
                                     x_train=x_train, x_val=x_val, x_test=x_test,
                                     y_train=y_train, y_val=y_val, y_test=y_test,
                                     n_jobs=n_jobs, run_test=run_test)

    def master_run(return_list, algo_class, algo_kwargs):
        algo_kwargs['ip'] = ''
        algo_kwargs['port'] = port
        algo = algo_class(**algo_kwargs)
        algo.run()
        try:
            algo.logger.info('===== bracket status: %s' % algo.get_bracket_status(algo.bracket))
        except Exception as e:
            pass
        try:
            algo.logger.info('===== brackets status: %s' % algo.get_brackets_status(algo.brackets))
        except Exception as e:
            pass
        return_list.extend(algo.recorder)  # send to return list

    def worker_run(i):
        if parallel_strategy == 'sync':
            worker = mqmfWorker(objective_function, ip, port)
        elif parallel_strategy == 'async':
            worker = async_mqmfWorker(objective_function, ip, port)
        else:
            raise ValueError('Error parallel_strategy: %s.' % parallel_strategy)
        worker.run()
        print("Worker %d exit." % (i,))

    manager = Manager()
    recorder = manager.list()   # shared list
    master = Process(target=master_run, args=(recorder, algo_class, algo_kwargs))
    master.start()

    time.sleep(10)  # wait for master init
    worker_pool = []
    for i in range(n_workers):
        worker = Process(target=worker_run, args=(i,))
        worker_pool.append(worker)
        worker.start()

    master.join()   # wait for master to gen result
    for w in worker_pool:
        w.kill()

    return list(recorder)   # covert to list


def run_exp(test_datasets, algo_class, algo_kwargs, algo_name, n_workers, parallel_strategy,
            R, n_jobs, runtime_limit, time_limit_per_trial, start_id, rep, ip, port,
            eta=None, pre_sample=False, run_test=True, max_local_workers=MAX_LOCAL_WORKERS):
    if n_workers > max_local_workers:
        print('Caution: n_workers=%d, max_local_workers=%d' % (n_workers, max_local_workers))
    check_datasets(test_datasets)
    model_name = 'xgb'
    for dataset in test_datasets:
        # setup
        n_jobs, runtime_limit, time_limit_per_trial = setup_exp(dataset, n_jobs, runtime_limit, time_limit_per_trial)
        print('===== start eval %s: rep=%d, n_jobs=%d, runtime_limit=%d, time_limit_per_trial=%d'
              % (dataset, rep, n_jobs, runtime_limit, time_limit_per_trial))
        for i in range(start_id, start_id + rep):
            seed = seeds[i]

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            if R != 27:
                method_str = '%s-%d-n%d' % (algo_name, R, n_workers)
            else:
                method_str = '%s-n%d' % (algo_name, n_workers)
            method_id = method_str + '-%s-%d-%s' % (dataset, seed, timestamp)

            # ip, port are filled in evaluate_parallel()
            algo_kwargs['objective_func'] = None
            algo_kwargs['config_space'] = XGBoost.get_cs()
            algo_kwargs['random_state'] = seed
            algo_kwargs['method_id'] = method_id
            algo_kwargs['runtime_limit'] = runtime_limit
            algo_kwargs['time_limit_per_trial'] = time_limit_per_trial

            n_local_workers = min(n_workers, max_local_workers)
            recorder = evaluate_parallel(
                algo_class, algo_kwargs, method_id, n_local_workers, dataset, seed, ip, port,
                parallel_strategy, n_jobs, R, eta=eta, pre_sample=pre_sample, run_test=run_test,
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

            if rep > 1 or len(test_datasets) > 1:
                sleep_time = min(runtime_limit/5, 1800)
                print('sleep %.2fs now!' % sleep_time, flush=True)
                time.sleep(sleep_time)

        try:
            remove_partial(model_name, dataset, [method_str], runtime_limit, R)
            get_incumbent(model_name, dataset, [method_str], runtime_limit)
        except Exception as e:
            print('benchmark process record failed: %s' % (traceback.format_exc(),))
