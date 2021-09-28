import os
import time
import traceback
import pickle as pkl
import numpy as np
from functools import partial
from multiprocessing import Process, Manager

from tuner.mq_mf_worker_gpu import mqmfWorker_gpu
from tuner.async_mq_mf_worker_gpu import async_mqmfWorker_gpu
from test.utils import setup_exp, seeds
from test.benchmark_process_record import remove_partial, get_incumbent
from resnet_obj import mf_objective_func_gpu
from resnet_model import ResNet32Classifier


def evaluate_parallel(algo_class, algo_kwargs, method_id, n_workers, dataset, seed, ip, port,
                      parallel_strategy, n_jobs, R, eta=3, run_test=True,
                      dir_path=None, file_name=None):
    # dataset / n_jobs are ignored
    assert dir_path is not None
    assert file_name is not None

    print(method_id, n_workers, dataset, seed)
    if port == 0:
        port = 13579 + np.random.RandomState(int(time.time() * 10000 % 10000)).randint(2000)
    print('ip=', ip, 'port=', port)
    assert parallel_strategy in ['sync', 'async']

    model_dir = os.path.join('./data/resnet_save_models', method_id)
    objective_function_gpu = partial(mf_objective_func_gpu, total_resource=R, run_test=run_test,
                                     model_dir=model_dir, eta=eta)

    def master_run(return_list, algo_class, algo_kwargs):
        algo_kwargs['ip'] = ''
        algo_kwargs['port'] = port
        algo = algo_class(**algo_kwargs)

        tmp_path = os.path.join(dir_path, 'tmp')
        algo.set_save_intermediate_record(tmp_path, file_name)

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
        device = 'cuda:%d' % i  # gpu
        if parallel_strategy == 'sync':
            worker = mqmfWorker_gpu(objective_function_gpu, device, ip, port)
        elif parallel_strategy == 'async':
            worker = async_mqmfWorker_gpu(objective_function_gpu, device, ip, port)
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


def run_exp(dataset, algo_class, algo_kwargs, algo_name, n_workers, parallel_strategy,
            R, n_jobs, runtime_limit, time_limit_per_trial, start_id, rep, ip, port,
            eta=3, pre_sample=False, run_test=False):
    # n_jobs / pre_sample are ignored
    assert dataset == 'cifar10'
    model = 'resnet'

    # setup
    print('===== start eval %s: rep=%d, n_jobs=%d, runtime_limit=%d, time_limit_per_trial=%d'
          % (dataset, rep, n_jobs, runtime_limit, time_limit_per_trial))
    for i in range(start_id, start_id + rep):
        seed = seeds[i]

        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        method_str = '%s-n%d' % (algo_name, n_workers)
        method_id = method_str + '-%s-%d-%s' % (dataset, seed, timestamp)

        # ip, port are filled in evaluate_parallel()
        algo_kwargs['objective_func'] = None
        algo_kwargs['config_space'] = ResNet32Classifier.get_hyperparameter_search_space()
        algo_kwargs['random_state'] = seed
        algo_kwargs['method_id'] = method_id
        algo_kwargs['runtime_limit'] = runtime_limit
        algo_kwargs['time_limit_per_trial'] = time_limit_per_trial

        dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, runtime_limit, method_str)
        file_name = 'record_%s.pkl' % (method_id,)

        recorder = evaluate_parallel(
            algo_class, algo_kwargs, method_id, n_workers, dataset, seed, ip, port,
            parallel_strategy, n_jobs, R, eta=eta, run_test=run_test,
            dir_path=dir_path, file_name=file_name,
        )

        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except FileExistsError:
            pass
        with open(os.path.join(dir_path, file_name), 'wb') as f:
            pkl.dump(recorder, f)
        print(dir_path, file_name, 'saved!', flush=True)

        if rep > 1:
            time.sleep(3600)

    try:
        remove_partial(model, dataset, [method_str], runtime_limit, R)
        get_incumbent(model, dataset, [method_str], runtime_limit)
    except Exception as e:
        print('benchmark process record failed: %s' % (traceback.format_exc(),))
