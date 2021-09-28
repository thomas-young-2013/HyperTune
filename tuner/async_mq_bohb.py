import os
import time
import numpy as np

from tuner.async_mq_hb import async_mqHyperband
from tuner.utils import RUNNING, COMPLETED, PROMOTED
from tuner.utils import sample_configuration
from tuner.utils import minmax_normalization, std_normalization
from tuner.acq_maximizer.ei_optimization import RandomSampling

from openbox.utils.util_funcs import get_types
from openbox.utils.config_space import ConfigurationSpace
from openbox.acquisition_function.acquisition import EI
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.utils.config_space.util import convert_configurations_to_array


class async_mqBOHB(async_mqHyperband):
    """
    The implementation of Asynchronous BOHB (combine ASHA and BOHB)
    no median imputation!
    """

    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 skip_outer_loop=0,
                 rand_prob=0.3,
                 bo_init_num=3,
                 random_state=1,
                 method_id='mqAsyncBOHB',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc'):
        super().__init__(objective_func, config_space, R, eta=eta, skip_outer_loop=skip_outer_loop,
                         random_state=random_state, method_id=method_id, restart_needed=restart_needed,
                         time_limit_per_trial=time_limit_per_trial, runtime_limit=runtime_limit,
                         ip=ip, port=port, authkey=authkey)

        self.rand_prob = rand_prob
        self.bo_init_num = bo_init_num
        types, bounds = get_types(config_space)
        self.surrogate = RandomForestWithInstances(types=types, bounds=bounds)
        self.acquisition_function = EI(model=self.surrogate)
        self.acq_optimizer = RandomSampling(self.acquisition_function, config_space,
                                            n_samples=max(5000, 50 * len(bounds)))
        self.rng = np.random.RandomState(self.seed)

    def update_observation(self, config, perf, n_iteration):
        rung_id = self.get_rung_id(self.bracket, n_iteration)

        updated = False
        for job in self.bracket[rung_id]['jobs']:
            _job_status, _config, _perf, _extra_conf = job
            if _config == config:
                assert _job_status == RUNNING
                job[0] = COMPLETED
                job[2] = perf
                updated = True
                break
        assert updated
        # print('=== bracket after update_observation:', self.get_bracket_status(self.bracket))

        if n_iteration == self.R:
            self.incumbent_configs.append(config)
            self.incumbent_perfs.append(perf)
            # train BO surrogate
            train_perfs = np.array(std_normalization(self.incumbent_perfs), dtype=np.float64)
            self.surrogate.train(convert_configurations_to_array(self.incumbent_configs), train_perfs)

    def choose_next(self):
        """
        sample a config according to BOHB. give iterations according to Hyperband strategy.
        """
        next_config = None
        next_n_iteration = self.get_next_n_iteration()
        next_rung_id = self.get_rung_id(self.bracket, next_n_iteration)

        # sample config
        excluded_configs = self.bracket[next_rung_id]['configs']

        if len(self.incumbent_configs) < self.bo_init_num or self.rng.random() < self.rand_prob:
            next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
        else:
            # BO
            start_time = time.time()
            best_index = np.argmin(self.incumbent_perfs)
            best_config = self.incumbent_configs[best_index]
            std_incumbent_value = np.min(std_normalization(self.incumbent_perfs))
            # Update surrogate model in acquisition function.
            self.acquisition_function.update(model=self.surrogate, eta=std_incumbent_value,
                                             num_data=len(self.incumbent_configs))
            candidates = self.acq_optimizer.maximize(best_config=best_config, batch_size=5000)
            time1 = time.time()
            for candidate in candidates:
                if candidate not in excluded_configs:
                    next_config = candidate
                    break
            if next_config is None:
                self.logger.warning('Cannot get a non duplicate configuration from bo candidates. '
                                    'Sample a random one.')
                next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
            time2 = time.time()
            if time2 - start_time > 1:
                self.logger.info('BO opt cost %.2fs. check duplication cost %.2fs. len of incumbent_configs: %d.'
                                 % (time1-start_time, time2-time1, len(self.incumbent_configs)))

        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf
