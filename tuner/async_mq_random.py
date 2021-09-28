import numpy as np
from tuner.async_mq_base_facade import async_mqBaseFacade
from tuner.utils import sample_configuration

from openbox.utils.config_space import ConfigurationSpace


class async_mqRandomSearch(async_mqBaseFacade):
    """
    The implementation of Asynchronous Random Search
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 random_state=1,
                 method_id='mqAsyncRandomSearch',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',
                 **kwargs):
        max_queue_len = 1000   # conservative design
        super().__init__(objective_func, method_name=method_id,
                         restart_needed=restart_needed, time_limit_per_trial=time_limit_per_trial,
                         runtime_limit=runtime_limit,
                         max_queue_len=max_queue_len, ip=ip, port=port, authkey=authkey)
        self.seed = random_state
        self.config_space = config_space
        self.config_space.seed(self.seed)
        self.R = R

        self.incumbent_configs = list()
        self.incumbent_perfs = list()

        self.all_configs = set()

        self.logger.info('Unused kwargs: %s' % kwargs)

    def get_job(self):
        """
        sample a random config
        """
        next_config = sample_configuration(self.config_space, excluded_configs=self.all_configs)
        next_n_iteration = self.R
        next_extra_conf = dict(initial_run=True)

        self.all_configs.add(next_config)

        return next_config, next_n_iteration, next_extra_conf

    def update_observation(self, config, perf, n_iteration):
        assert int(n_iteration) == self.R
        self.incumbent_configs.append(config)
        self.incumbent_perfs.append(perf)
        return

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        perfs = [self.incumbent_perfs[i] for i in indices[0: num_inc]]
        return configs, perfs
