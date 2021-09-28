import time
import traceback
import numpy as np


def run_in_parallel(self, configurations, n_iteration, extra_info=None, initial_run=True):
    """
    simulation function for sync algorithms

    n_workers: number of simulation workers

    objective_func:
        :param: config, n_iteration, extra_conf
        :return: a dict that must contain: 'objective_value', 'elapsed_time'
    """
    assert hasattr(self, 'objective_func') and hasattr(self, 'n_workers')

    n_configuration = len(configurations)
    performance_result = []
    early_stops = []

    # TODO: need systematic tests.
    # check configurations, whether it exists the same configs
    count_dict = dict()
    for i, config in enumerate(configurations):
        if config not in count_dict:
            count_dict[config] = 0
        count_dict[config] += 1

    # incorporate ref info.
    conf_list = []
    for index, config in enumerate(configurations):
        extra_conf_dict = dict()
        if count_dict[config] > 1:
            extra_conf_dict['uid'] = count_dict[config]
            count_dict[config] -= 1

        if extra_info is not None:
            extra_conf_dict['reference'] = extra_info[index]
        extra_conf_dict['need_lc'] = self.record_lc
        extra_conf_dict['method_name'] = self.method_name
        extra_conf_dict['initial_run'] = initial_run  # for loading from checkpoint in DL
        conf_list.append((config, extra_conf_dict))

    # ==================== simulation start ====================
    if not hasattr(self, 'simulation_global_time'):
        self.simulation_global_time = 0
    worker_list = []
    result_num = 0
    result_needed = len(conf_list)
    self.logger.info('Master: %d configs sent.' % result_needed)
    # send all jobs
    while len(conf_list) > 0:
        # assign job
        if len(worker_list) < self.n_workers:
            config, extra_conf = conf_list.pop(0)
            trial_id = self.global_trial_counter
            self.global_trial_counter += 1
            # query result
            result = self.objective_func(config, n_iteration, extra_conf)
            return_info = dict(loss=result['objective_value'],
                               n_iteration=n_iteration,
                               ref_id=result.get('ref_id', None),
                               early_stop=result.get('early_stop', False),
                               trial_state=None,
                               test_perf=result.get('test_perf', None),
                               extra_conf=extra_conf)
            observation = [return_info, result['elapsed_time'], trial_id, config]
            job = dict(
                observation=observation,
                time_remaining=result['elapsed_time'],
            )
            worker_list.append(job)
        # run job with least remaining time
        else:
            worker_id = np.argmin([job['time_remaining'] for job in worker_list]).item()
            run_job = worker_list.pop(worker_id)
            run_time = run_job['time_remaining']
            for job in worker_list:
                job['time_remaining'] -= run_time
                if job['time_remaining'] < 0:
                    self.logger.warning('time_remaining %f < 0. please check!' % job['time_remaining'])
                    job['time_remaining'] = 0
            self.simulation_global_time += run_time
            global_time = self.simulation_global_time
            observation = run_job['observation']
            self.trial_statistics.append((observation, global_time))
            result_num += 1
            self.logger.info('Master: Get the [%d] observation %s. Global time=%.2fs.'
                             % (result_num, str(observation), global_time))

    # complete remaining jobs
    while len(worker_list) > 0:
        # run job with least remaining time
        worker_id = np.argmin([job['time_remaining'] for job in worker_list]).item()
        run_job = worker_list.pop(worker_id)
        run_time = run_job['time_remaining']
        for job in worker_list:
            job['time_remaining'] -= run_time
            if job['time_remaining'] < 0:
                self.logger.warning('time_remaining %f < 0. please check!' % job['time_remaining'])
                job['time_remaining'] = 0
        self.simulation_global_time += run_time
        global_time = self.simulation_global_time
        observation = run_job['observation']
        self.trial_statistics.append((observation, global_time))
        result_num += 1
        self.logger.info('Master: Get the [%d] observation %s. Global time=%.2fs.'
                         % (result_num, str(observation), global_time))

    assert result_num == result_needed
    # ==================== simulation end ====================

    # sort by trial_id. FIX BUG
    self.trial_statistics.sort(key=lambda x: x[0][2])

    # get the evaluation statistics
    for observation, global_time in self.trial_statistics:
        return_info, time_taken, trial_id, config = observation

        performance = return_info['loss']
        if performance < self.global_incumbent:
            self.global_incumbent = performance
            self.global_incumbent_configuration = config

        self.add_history(global_time, self.global_incumbent, trial_id,
                         self.global_incumbent_configuration)
        performance_result.append(return_info)
        early_stops.append(return_info.get('early_stop', False))
        if self.runtime_limit is None or global_time < self.runtime_limit:
            self.recorder.append({'trial_id': trial_id, 'time_consumed': time_taken,
                                  'configuration': config, 'n_iteration': n_iteration,
                                  'return_info': return_info, 'global_time': global_time})

    self.trial_statistics.clear()

    # self.save_intermediate_statistics()
    if self.runtime_limit is not None and self.simulation_global_time > self.runtime_limit:
        raise ValueError('Runtime budget meets!')
    return performance_result, early_stops


def run_async(self):
    """
    simulation function for async algorithms

    n_workers: number of simulation workers

    objective_func:
        :param: config, n_iteration, extra_conf
        :return: a dict that must contain: 'objective_value', 'elapsed_time'
    """
    assert hasattr(self, 'objective_func') and hasattr(self, 'n_workers')
    try:
        self.simulation_global_time = 0
        worker_list = []
        # init async: fill all workers
        while len(worker_list) < self.n_workers:
            # Send new job
            t = time.time()
            config, n_iteration, extra_conf = self.get_job()
            self.logger.info('get_job() cost %.2fs.' % (time.time() - t,))
            trial_id = self.global_trial_counter
            self.global_trial_counter += 1
            # query result
            result = self.objective_func(config, n_iteration, extra_conf)
            return_info = dict(loss=result['objective_value'],
                               n_iteration=n_iteration,
                               ref_id=result.get('ref_id', None),
                               early_stop=result.get('early_stop', False),
                               trial_state=None,
                               test_perf=result.get('test_perf', None),
                               extra_conf=extra_conf)
            observation = [return_info, result['elapsed_time'], trial_id, config]
            job = dict(
                observation=observation,
                time_remaining=result['elapsed_time'],
            )
            worker_list.append(job)
            self.logger.info("Worker %d init." % len(worker_list))
            self.logger.info('Master send job: %s.' % (job,))

        while True:
            if self.runtime_limit is not None and self.simulation_global_time > self.runtime_limit:
                self.logger.info('RUNTIME BUDGET is RUNNING OUT.')
                return

            # run job with least remaining time
            worker_id = np.argmin([job['time_remaining'] for job in worker_list]).item()
            run_job = worker_list.pop(worker_id)
            run_time = run_job['time_remaining']
            for job in worker_list:
                job['time_remaining'] -= run_time
                if job['time_remaining'] < 0:
                    self.logger.warning('time_remaining %f < 0. please check!' % job['time_remaining'])
                    job['time_remaining'] = 0
            self.simulation_global_time += run_time
            global_time = self.simulation_global_time
            observation = run_job['observation']

            return_info, time_taken, trial_id, config = observation
            # update observation
            self.logger.info('Master get observation: %s. Global time=%.2fs.' % (str(observation), global_time))
            n_iteration = return_info['n_iteration']
            perf = return_info['loss']
            t = time.time()
            self.update_observation(config, perf, n_iteration)
            self.logger.info('update_observation() cost %.2fs.' % (time.time() - t,))
            if self.runtime_limit is None or global_time < self.runtime_limit:
                self.recorder.append({'trial_id': trial_id, 'time_consumed': time_taken,
                                      'configuration': config, 'n_iteration': n_iteration,
                                      'return_info': return_info, 'global_time': global_time})
            # if (not hasattr(self, 'R')) or n_iteration == self.R:
            #     self.save_intermediate_statistics()

            # Send new job
            t = time.time()
            config, n_iteration, extra_conf = self.get_job()
            self.logger.info('get_job() cost %.2fs.' % (time.time() - t,))
            trial_id = self.global_trial_counter
            self.global_trial_counter += 1
            # query result
            result = self.objective_func(config, n_iteration, extra_conf)
            return_info = dict(loss=result['objective_value'],
                               n_iteration=n_iteration,
                               ref_id=result.get('ref_id', None),
                               early_stop=result.get('early_stop', False),
                               trial_state=None,
                               test_perf=result.get('test_perf', None),
                               extra_conf=extra_conf)
            observation = [return_info, result['elapsed_time'], trial_id, config]
            job = dict(
                observation=observation,
                time_remaining=result['elapsed_time'],
            )
            worker_list.append(job)
            self.logger.info('Master send job: %s.' % (job,))

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        self.logger.error(traceback.format_exc())

