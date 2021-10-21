import time
import traceback
import numpy as np
from tuner.mq_base_facade import mqBaseFacade
from tuner.utils import sample_configurations, expand_configurations

from openbox.core.sync_batch_advisor import SyncBatchAdvisor, SUCCESS
from openbox.utils.config_space import ConfigurationSpace
from openbox.core.base import Observation


class mqBO(mqBaseFacade):
    """
    synchronous parallel Bayesian Optimization (using OpenBox)
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 n_workers,
                 num_iter=10000,
                 bo_init_num=3,
                 random_state=1,
                 method_id='mqBO',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',
                 **kwargs):
        max_queue_len = max(1000, 3 * n_workers)  # conservative design
        super().__init__(objective_func, method_name=method_id,
                         restart_needed=restart_needed, time_limit_per_trial=time_limit_per_trial,
                         runtime_limit=runtime_limit,
                         max_queue_len=max_queue_len, ip=ip, port=port, authkey=authkey)
        self.seed = random_state
        self.config_space = config_space
        self.config_space.seed(self.seed)

        self.R = R
        self.n_workers = n_workers
        self.bo_init_num = bo_init_num
        # using median_imputation batch_strategy implemented in OpenBox to generate BO suggestions
        if 'task_info' in SyncBatchAdvisor.__init__.__code__.co_varnames:
            # old version OpenBox
            task_info = {'num_constraints': 0, 'num_objs': 1}
            task_kwargs = dict(task_info=task_info)
        else:
            task_kwargs = dict(num_objs=1, num_constraints=0)
        self.config_advisor = SyncBatchAdvisor(config_space,
                                               **task_kwargs,
                                               batch_size=self.n_workers,
                                               batch_strategy='median_imputation',
                                               initial_trials=self.bo_init_num,
                                               init_strategy='random',
                                               optimization_strategy='bo',
                                               surrogate_type='prf',
                                               acq_type='ei',
                                               acq_optimizer_type='local_random',
                                               task_id=self.method_name,
                                               output_dir=self.log_directory,
                                               random_state=random_state,
                                               )

        self.num_iter = num_iter
        self.incumbent_configs = []
        self.incumbent_perfs = []
        self.logger.info('Unused kwargs: %s' % kwargs)

    def run(self):
        try:
            for iter in range(1, 1 + self.num_iter):
                self.logger.info('-' * 50)
                self.logger.info("%s: %d/%d iteration starts" % (self.method_name, iter, self.num_iter))
                start_time = time.time()
                self.iterate()
                time_elapsed = (time.time() - start_time) / 60
                self.logger.info("iteration took %.2f min." % time_elapsed)
                self.save_intermediate_statistics()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.logger.error(traceback.format_exc())
            # clear the immediate result.
            # self.remove_immediate_model()

    def iterate(self):
        configs = self.get_bo_candidates()
        extra_info = None
        ret_val, early_stops = self.run_in_parallel(configs, self.R, extra_info, initial_run=True)
        val_losses = [item['loss'] for item in ret_val]

        self.incumbent_configs.extend(configs)
        self.incumbent_perfs.extend(val_losses)
        self.add_stage_history(self.stage_id, self.global_incumbent)
        self.stage_id += 1
        # self.remove_immediate_model()

        # update bo advisor
        for config, perf in zip(configs, val_losses):
            objs = [perf]
            observation = Observation(
                config=config, objs=objs, constraints=None,
                trial_state=SUCCESS, elapsed_time=None,
            )
            self.config_advisor.update_observation(observation)
            self.logger.info('update BO observation: config=%s, perf=%f' % (str(config), perf))

    def get_bo_candidates(self):
        num_config = self.n_workers
        # get bo configs
        if len(self.incumbent_configs) < self.bo_init_num:
            # fix bug: bo advisor suggests repeated configs if call get_suggestions() repeatedly in initial stage
            bo_configs = list()
        else:
            bo_configs = self.config_advisor.get_suggestions()
            bo_configs = bo_configs[:num_config]  # may exceed num_config in initial random sampling
        self.logger.info('len bo configs = %d.' % len(bo_configs))

        # sample random configs
        configs = expand_configurations(bo_configs, self.config_space, num_config)
        self.logger.info('len total configs = %d.' % len(configs))
        assert len(configs) == num_config
        return configs

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        targets = [self.incumbent_perfs[i] for i in indices[0: num_inc]]
        return configs, targets
