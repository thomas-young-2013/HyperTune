import numpy as np
import random
from ConfigSpace.util import get_one_exchange_neighbourhood
from tuner.async_mq_base_facade import async_mqBaseFacade
from tuner.utils import sample_configuration

from openbox.utils.config_space import ConfigurationSpace


class async_mqEA(async_mqBaseFacade):
    """
    The implementation of Asynchronous Evolutionary Algorithm
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 population_size=30,
                 subset_size=20,
                 epsilon=0.2,
                 strategy='worst',  # 'worst', 'oldest'
                 random_state=1,
                 method_id='mqAsyncEA',
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

        self.age = 0
        self.population = list()
        self.population_size = population_size
        self.subset_size = subset_size
        assert 0 < self.subset_size <= self.population_size
        self.epsilon = epsilon
        self.strategy = strategy
        assert self.strategy in ['worst', 'oldest']
        self.rng = np.random.RandomState(self.seed)

        self.logger.info('Unused kwargs: %s' % kwargs)

    def get_job(self):
        """
        sample a random config
        """
        if len(self.population) < self.population_size:
            # Initialize population
            next_config = sample_configuration(self.config_space, excluded_configs=self.all_configs)
        else:
            # Select a parent by subset tournament and epsilon greedy
            if self.rng.random() < self.epsilon:
                parent_config = random.sample(self.population, 1)[0]['config']
            else:
                subset = random.sample(self.population, self.subset_size)
                subset.sort(key=lambda x: x['perf'])    # minimize
                parent_config = subset[0]['config']

            # Mutation to 1-step neighbors
            next_config = None
            neighbors_gen = get_one_exchange_neighbourhood(parent_config, seed=1)
            for neighbor in neighbors_gen:
                if neighbor not in self.all_configs:
                    next_config = neighbor
                    break
            if next_config is None:  # If all the neighors are evaluated, sample randomly!
                next_config = sample_configuration(self.config_space, excluded_configs=self.all_configs)

        self.all_configs.add(next_config)

        next_n_iteration = self.R
        next_extra_conf = dict(initial_run=True)
        return next_config, next_n_iteration, next_extra_conf

    def update_observation(self, config, perf, n_iteration):
        assert int(n_iteration) == self.R
        self.incumbent_configs.append(config)
        self.incumbent_perfs.append(perf)

        # update population
        self.population.append(dict(config=config, age=self.age, perf=perf))
        self.age += 1

        # Eliminate samples
        if len(self.population) > self.population_size:
            if self.strategy == 'oldest':
                self.population.sort(key=lambda x: x['age'])
                self.population.pop(0)
            elif self.strategy == 'worst':
                self.population.sort(key=lambda x: x['perf'])
                self.population.pop(-1)
            else:
                raise ValueError('Unknown strategy: %s' % self.strategy)
        return

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        perfs = [self.incumbent_perfs[i] for i in indices[0: num_inc]]
        return configs, perfs
