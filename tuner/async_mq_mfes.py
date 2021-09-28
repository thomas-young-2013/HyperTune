import os
import time
import numpy as np
from math import log, ceil
from sklearn.model_selection import KFold
from scipy.optimize import minimize

from tuner.async_mq_hb import async_mqHyperband
from tuner.utils import RUNNING, COMPLETED, PROMOTED
from tuner.utils import sample_configuration
from tuner.utils import minmax_normalization, std_normalization
from tuner.surrogate.rf_ensemble import RandomForestEnsemble
from tuner.surrogate.gp_ensemble import GaussianProcessEnsemble
from tuner.acq_maximizer.ei_optimization import RandomSampling

from openbox.utils.util_funcs import get_types
from openbox.utils.config_space import ConfigurationSpace
from openbox.acquisition_function.acquisition import EI
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.surrogate.base.build_gp import create_gp_model
from openbox.acq_maximizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from openbox.acq_maximizer.random_configuration_chooser import ChooserProb
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.history_container import HistoryContainer


class async_mqMFES(async_mqHyperband):
    """
    The implementation of our method
    """

    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 skip_outer_loop=0,
                 rand_prob=0.3,
                 init_weight=None, update_enable=True,
                 weight_method='rank_loss_p_norm',
                 fusion_method='idp',
                 power_num=3,
                 set_promotion_threshold=True,
                 non_decreasing_weight=False,
                 increasing_weight=True,
                 surrogate_type='prf',  # 'prf', 'gp'
                 acq_optimizer='local_random',  # 'local_random', 'random'
                 use_weight_init=True,
                 weight_init_choosing='proportional',  # 'proportional', 'pow', 'argmax', 'argmax2'
                 median_imputation=None,  # None, 'top', 'corresponding', 'all'
                 test_sh=False,
                 test_random=False,
                 test_bohb=False,
                 test_original_asha=False,
                 random_state=1,
                 method_id='mqAsyncMFES',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc'):

        self.set_promotion_threshold = set_promotion_threshold

        super().__init__(objective_func, config_space, R, eta=eta, skip_outer_loop=skip_outer_loop,
                         random_state=random_state, method_id=method_id, restart_needed=restart_needed,
                         time_limit_per_trial=time_limit_per_trial, runtime_limit=runtime_limit,
                         ip=ip, port=port, authkey=authkey)

        # test version
        self.test_sh = test_sh
        self.test_random = test_random
        self.test_bohb = test_bohb
        self.test_original_asha = test_original_asha

        self.update_enable = update_enable
        self.fusion_method = fusion_method
        # Parameter for weight method `rank_loss_p_norm`.
        self.power_num = power_num
        # Specify the weight learning method.
        self.weight_method = weight_method
        self.weight_update_id = 0
        self.weight_changed_cnt = 0

        self.init_weight = init_weight
        if self.init_weight is None:
            if self.test_bohb:
                self.init_weight = [0.] * self.s_max + [1.]
            elif self.s_max == 0:
                self.init_weight = [1.]
            else:
                self.init_weight = [1. / self.s_max] * self.s_max + [0.]
        assert len(self.init_weight) == (self.s_max + 1)
        self.logger.info("Initialize weight to %s" % self.init_weight)
        types, bounds = get_types(config_space)

        self.rng = np.random.RandomState(seed=self.seed)

        self.surrogate_type = surrogate_type
        if self.surrogate_type == 'prf':
            self.surrogate = RandomForestEnsemble(types, bounds, self.s_max, self.eta,
                                                  self.init_weight, self.fusion_method)
        elif self.surrogate_type == 'gp':
            self.surrogate = GaussianProcessEnsemble(config_space, types, bounds, self.s_max, self.eta,
                                                     self.init_weight, self.fusion_method, self.rng)
        else:
            raise ValueError('Unknown surrogate type: %s' % self.surrogate_type)
        self.acquisition_function = EI(model=self.surrogate)

        self.iterate_id = 0
        self.iterate_r = list()
        self.hist_weights = list()
        self.hist_weights_unadjusted = list()

        # Saving evaluation statistics in Hyperband.
        self.target_x = dict()
        self.target_y = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = list()
            self.target_y[r] = list()

        # BO optimizer settings.
        self.history_container = HistoryContainer(task_id=self.method_name)
        self.sls_max_steps = None
        self.n_sls_iterations = 5
        self.sls_n_steps_plateau_walk = 10
        self.acq_optimizer_type = acq_optimizer
        if self.acq_optimizer_type == 'local_random':
            self.acq_optimizer = InterleavedLocalAndRandomSearch(
                acquisition_function=self.acquisition_function,
                config_space=self.config_space,
                rng=self.rng,
                max_steps=self.sls_max_steps,
                n_steps_plateau_walk=self.sls_n_steps_plateau_walk,
                n_sls_iterations=self.n_sls_iterations,
                rand_prob=0.0,
            )
        elif self.acq_optimizer_type == 'random':
            self.acq_optimizer = RandomSampling(self.acquisition_function, config_space,
                                                n_samples=max(5000, 50 * len(bounds)))
        else:
            raise ValueError
        self.random_configuration_chooser = ChooserProb(prob=rand_prob, rng=self.rng)
        self.random_check_idx = 0

        self.non_decreasing_weight = non_decreasing_weight
        self.increasing_weight = increasing_weight
        assert not (self.non_decreasing_weight and self.increasing_weight)
        self.use_weight_init = use_weight_init
        self.weight_init_choosing = weight_init_choosing
        assert self.weight_init_choosing in ['proportional', 'pow', 'argmax', 'argmax2']
        self.n_init_configs = np.array(
            [len(init_iter_list) for init_iter_list in self.hb_bracket_list],
            dtype=np.float64
        )

        # median imputation
        self.median_imputation = median_imputation
        self.configs_running_dict = dict()
        self.all_configs_running = set()
        assert self.median_imputation in [None, 'top', 'corresponding', 'all']

    def create_bracket(self):
        """
        bracket : list of rungs
        rung: {
            'rung_id': rung id (the lowest rung is 0),
            'n_iteration': iterations (resource) per config for evaluation,
            'jobs': list of [job_status, config, perf, extra_conf],
            'configs': set of all configs in the rung,
            'num_promoted': number of promoted configs in the rung,
            'promotion_start_threshold':
                promotion starts when number of completed/promoted jobs greater than threshold,
        }
        job_status: RUNNING, COMPLETED, PROMOTED
        """
        self.bracket = list()
        s = self.s_max
        # Initial number of iterations per config
        r = self.R * self.eta ** (-s)
        for i in range(s + 1):
            n_iteration = r * self.eta ** (i)
            promotion_start_threshold = self.R * self.eta ** (-i)
            rung = dict(
                rung_id=i,
                n_iteration=n_iteration,
                jobs=list(),
                configs=set(),
                num_promoted=0,
                # set promotion start threshold
                promotion_start_threshold=promotion_start_threshold if self.set_promotion_threshold else 0,
            )
            self.bracket.append(rung)
        self.logger.info('Init bracket: %s.' % str(self.bracket))

    def can_promote(self, rung_id):
        """
        return whether configs can be promoted in current rung
        """
        if self.test_original_asha:
            return True

        # if not enough jobs, do not promote
        num_completed_promoted = len([job for job in self.bracket[rung_id]['jobs']
                                      if job[0] in (COMPLETED, PROMOTED)])
        num_promoted = self.bracket[rung_id]['num_promoted']
        if num_completed_promoted == 0 or (num_promoted + 1) / num_completed_promoted > 1 / self.eta:
            return False

        # prevent error promotion in start stage
        promotion_start_threshold = self.bracket[rung_id]['promotion_start_threshold']
        if num_completed_promoted < promotion_start_threshold:
            return False

        return True

    def get_job(self):
        next_config, next_n_iteration, next_extra_conf = super().get_job()
        # for median imputation
        if self.median_imputation is not None:
            start_time = time.time()
            r = int(next_n_iteration)
            if r not in self.configs_running_dict.keys():
                self.configs_running_dict[r] = set()
            self.configs_running_dict[r].add(next_config)
            self.all_configs_running.add(next_config)

            if self.median_imputation == 'corresponding':
                self.train_surrogate(self.surrogate, r, median_imputation=True)
            elif self.median_imputation == 'top':
                # self.train_surrogate(self.surrogate, self.iterate_r[-1], median_imputation=True, all_impute=True)
                pass    # refit when update_observation
            elif self.median_imputation == 'all':
                pass    # refit when update_observation
            else:
                raise ValueError('Unknown median_imputation: %s' % self.median_imputation)

            self.logger.info('get_job median_imputation cost %.2fs.' % (time.time() - start_time))

        return next_config, next_n_iteration, next_extra_conf

    def train_surrogate(self, surrogate, n_iteration: int, median_imputation: bool, all_impute=False):
        if median_imputation:
            if all_impute:
                configs_running = list(self.all_configs_running)
            else:
                configs_running = list(self.configs_running_dict[n_iteration])
            value_imputed = np.median(self.target_y[n_iteration])
            configs_train = self.target_x[n_iteration] + configs_running
            results_train = self.target_y[n_iteration] + [value_imputed] * len(configs_running)
        else:
            configs_train = self.target_x[n_iteration]
            results_train = self.target_y[n_iteration]
        results_train = np.array(std_normalization(results_train), dtype=np.float64)
        surrogate.train(convert_configurations_to_array(configs_train), results_train, r=n_iteration)

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

        n_iteration = int(n_iteration)

        self.target_x[n_iteration].append(config)
        self.target_y[n_iteration].append(perf)

        if self.median_imputation is not None:
            self.configs_running_dict[n_iteration].remove(config)
            self.all_configs_running.remove(config)

        # Refit the ensemble surrogate model.
        start_time = time.time()
        if self.median_imputation is None:
            self.train_surrogate(self.surrogate, n_iteration, median_imputation=False)
        elif self.median_imputation == 'corresponding':
            self.train_surrogate(self.surrogate, n_iteration, median_imputation=True)
        elif self.median_imputation == 'top':
            if n_iteration != self.iterate_r[-1]:
                self.train_surrogate(self.surrogate, n_iteration, median_imputation=False)
            self.train_surrogate(self.surrogate, self.iterate_r[-1], median_imputation=True, all_impute=True)
        elif self.median_imputation == 'all':
            for r in self.iterate_r:
                self.train_surrogate(self.surrogate, r, median_imputation=True, all_impute=True)
        else:
            raise ValueError('Unknown median_imputation: %s' % self.median_imputation)
        self.logger.info('update_observation training surrogate cost %.2fs.' % (time.time() - start_time))

        if n_iteration == self.R:
            self.incumbent_configs.append(config)
            self.incumbent_perfs.append(perf)
            # Update history container.
            self.history_container.add(config, perf)

            # Update weight
            if self.update_enable and len(self.incumbent_configs) >= 8:  # todo: replace 8 by full observation num
                self.weight_update_id += 1
                self.update_weight()

    def choose_next(self):
        """
        sample a config according to MFES. give iterations according to Hyperband strategy.
        """
        next_config = None
        next_n_iteration = self.get_next_n_iteration()
        next_rung_id = self.get_rung_id(self.bracket, next_n_iteration)

        # sample config
        excluded_configs = self.bracket[next_rung_id]['configs']
        if any([len(y_list) == 0 for y_list in self.target_y.values()]) or self.test_random:
            next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
        else:
            # Like BOHB, sample a fixed percentage of random configurations.
            self.random_check_idx += 1
            if self.random_configuration_chooser.check(self.random_check_idx):
                next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
            else:
                acq_configs = self.get_bo_candidates()
                for config in acq_configs:
                    if config not in self.bracket[next_rung_id]['configs']:
                        next_config = config
                        break
                if next_config is None:
                    self.logger.warning('Cannot get a non duplicate configuration from bo candidates. '
                                        'Sample a random one.')
                    next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)

        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf

    def get_next_n_iteration(self):
        """
        choose next_n_iteration according to weights
        """
        if self.test_sh:
            return self.bracket[0]['n_iteration']

        if self.use_weight_init and len(self.incumbent_configs) >= 3 * 8:  # todo: replace 8 by full observation num
            if self.weight_init_choosing == 'proportional':
                top_k = 2
                new_weights = self.hist_weights_unadjusted[-1]
                choose_weights = np.asarray(new_weights, dtype=np.float64) * self.n_init_configs
                top_idx = np.argsort(choose_weights)[::-1][:top_k]
                # retain top k weights
                for i in range(len(choose_weights)):
                    if i not in top_idx:
                        choose_weights[i] = 0.
                choose_weights = choose_weights / np.sum(choose_weights)
                next_n_iteration = self.rng.choice(self.iterate_r, p=choose_weights)
                self.logger.info('random choosing next_n_iteration=%d. unadjusted_weights: %s. '
                                 'n_init_configs: %s. choose_weights: %s.'
                                 % (next_n_iteration, new_weights, self.n_init_configs, choose_weights))
            elif self.weight_init_choosing == 'pow':
                top_k = 2
                new_weights = self.hist_weights_unadjusted[-1]
                choose_weights = np.array(new_weights, dtype=np.float64) ** self.power_num
                choose_weights[-1] = 0.
                top_idx = np.argsort(choose_weights)[::-1][:top_k]
                # retain top k weights
                for i in range(len(choose_weights)):
                    if i not in top_idx:
                        choose_weights[i] = 0.
                weight_sum = np.sum(choose_weights)
                if weight_sum <= 1e-8:
                    choose_weights = np.array(self.init_weight, dtype=np.float64)
                else:
                    choose_weights = choose_weights / weight_sum
                next_n_iteration = self.rng.choice(self.iterate_r, p=choose_weights)
                self.logger.info('random choosing next_n_iteration=%d. new_weights: %s. choose_weights: %s.'
                                 % (next_n_iteration, new_weights, choose_weights))
            elif self.weight_init_choosing == 'argmax':
                new_weights = self.hist_weights_unadjusted[-1]
                choose_weights = np.asarray(new_weights, dtype=np.float64) * self.n_init_configs
                choose_weights = choose_weights / np.sum(choose_weights)
                idx = np.argmax(choose_weights).item()
                next_n_iteration = self.iterate_r[idx]
                self.logger.info('argmax choosing next_n_iteration=%d. new_weights: %s. '
                                 'n_init_configs: %s. choose_weights: %s.'
                                 % (next_n_iteration, new_weights, self.n_init_configs, choose_weights))
            elif self.weight_init_choosing == 'argmax2':
                new_weights = self.hist_weights_unadjusted[-1]
                idx = np.argmax(new_weights).item()
                next_n_iteration = self.iterate_r[idx]
                self.logger.info('argmax2 choosing next_n_iteration=%d. new_weights: %s.'
                                 % (next_n_iteration, new_weights))
            else:
                raise ValueError('Unknown weight_init_choosing: %s' % self.weight_init_choosing)
            return next_n_iteration

        return super().get_next_n_iteration()

    def get_bo_candidates(self):
        if self.acq_optimizer_type == 'local_random':
            std_incumbent_value = np.min(std_normalization(self.target_y[self.iterate_r[-1]]))
            # Update surrogate model in acquisition function.
            self.acquisition_function.update(model=self.surrogate, eta=std_incumbent_value,
                                             num_data=len(self.incumbent_configs))

            challengers = self.acq_optimizer.maximize(
                runhistory=self.history_container,
                num_points=5000,
            )
            return challengers.challengers
        elif self.acq_optimizer_type == 'random':
            best_index = np.argmin(self.incumbent_perfs)
            best_config = self.incumbent_configs[best_index]
            std_incumbent_value = np.min(std_normalization(self.incumbent_perfs))
            # Update surrogate model in acquisition function.
            self.acquisition_function.update(model=self.surrogate, eta=std_incumbent_value,
                                             num_data=len(self.incumbent_configs))
            candidates = self.acq_optimizer.maximize(best_config=best_config, batch_size=5000)
            return candidates
        else:
            raise ValueError

    @staticmethod
    def calculate_preserving_order_num(y_pred, y_true):
        array_size = len(y_pred)
        assert len(y_true) == array_size

        total_pair_num, order_preserving_num = 0, 0
        for idx in range(array_size):
            for inner_idx in range(idx + 1, array_size):
                if bool(y_true[idx] > y_true[inner_idx]) == bool(y_pred[idx] > y_pred[inner_idx]):
                    order_preserving_num += 1
                total_pair_num += 1
        return order_preserving_num, total_pair_num

    def update_weight(self):
        start_time = time.time()

        max_r = self.iterate_r[-1]
        incumbent_configs = self.target_x[max_r]
        if len(incumbent_configs) < 3:
            return
        test_x = convert_configurations_to_array(incumbent_configs)
        test_y = np.array(self.target_y[max_r], dtype=np.float64)

        r_list = self.surrogate.surrogate_r
        K = len(r_list)

        old_weights = list()
        for i, r in enumerate(r_list):
            _weight = self.surrogate.surrogate_weight[r]
            old_weights.append(_weight)

        if len(test_y) >= 3:
            # refit surrogate model without median imputation
            if self.median_imputation is None:
                test_surrogate = self.surrogate
            else:
                types, bounds = get_types(self.config_space)
                if self.surrogate_type == 'prf':
                    test_surrogate = RandomForestEnsemble(types, bounds, self.s_max, self.eta,
                                                          old_weights, self.fusion_method)
                elif self.surrogate_type == 'gp':
                    test_surrogate = GaussianProcessEnsemble(self.config_space, types, bounds, self.s_max, self.eta,
                                                             old_weights, self.fusion_method, self.rng)
                else:
                    raise ValueError('Unknown surrogate type: %s' % self.surrogate_type)
                for r in self.iterate_r:
                    self.train_surrogate(test_surrogate, r, median_imputation=False)

            # Get previous weights
            if self.weight_method == 'rank_loss_p_norm':
                preserving_order_p = list()
                preserving_order_nums = list()
                for i, r in enumerate(r_list):
                    fold_num = 5
                    if i != K - 1:
                        mean, var = test_surrogate.surrogate_container[r].predict(test_x)
                        tmp_y = np.reshape(mean, -1)
                        preorder_num, pair_num = self.calculate_preserving_order_num(tmp_y, test_y)
                        preserving_order_p.append(preorder_num / pair_num)
                        preserving_order_nums.append(preorder_num)
                    else:
                        if len(test_y) < 2 * fold_num:
                            preserving_order_p.append(0)
                        else:
                            # 5-fold cross validation.
                            kfold = KFold(n_splits=fold_num)
                            cv_pred = np.array([0] * len(test_y))
                            for train_idx, valid_idx in kfold.split(test_x):
                                train_configs, train_y = test_x[train_idx], test_y[train_idx]
                                valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                                types, bounds = get_types(self.config_space)
                                if self.surrogate_type == 'prf':
                                    _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                                elif self.surrogate_type == 'gp':
                                    _surrogate = create_gp_model('gp', self.config_space, types, bounds, self.rng)
                                else:
                                    raise ValueError('Unknown surrogate type: %s' % self.surrogate_type)
                                _surrogate.train(train_configs, train_y)
                                pred, _ = _surrogate.predict(valid_configs)
                                cv_pred[valid_idx] = pred.reshape(-1)
                            preorder_num, pair_num = self.calculate_preserving_order_num(cv_pred, test_y)
                            preserving_order_p.append(preorder_num / pair_num)
                            preserving_order_nums.append(preorder_num)

                self.logger.info('update weight preserving_order_p: %s' % preserving_order_p)
                trans_order_weight = np.array(preserving_order_p)
                power_sum = np.sum(np.power(trans_order_weight, self.power_num))
                new_weights = np.power(trans_order_weight, self.power_num) / power_sum

            elif self.weight_method == 'rank_loss_prob':
                t1 = time.time()
                # For basic surrogate i=1:K-1.
                mean_list, var_list = list(), list()
                prob_list = list()
                for i, r in enumerate(r_list[:-1]):
                    mean, var = test_surrogate.surrogate_container[r].predict(test_x)
                    mean_list.append(np.reshape(mean, -1))
                    var_list.append(np.reshape(var, -1))

                    tmp_y = np.reshape(mean, -1)
                    preorder_num, pair_num = self.calculate_preserving_order_num(tmp_y, test_y)
                    prob_list.append(preorder_num / pair_num)
                self.logger.info('update weight preserving_order prob_list: %s' % prob_list)

                t2 = time.time()

                sample_num = 100
                min_probability_array = [0] * K
                for _ in range(sample_num):
                    order_preseving_nums = list()

                    # For basic surrogate i=1:K-1.
                    for idx in range(K - 1):
                        sampled_y = self.rng.normal(mean_list[idx], var_list[idx])
                        _num, _ = self.calculate_preserving_order_num(sampled_y, test_y)
                        order_preseving_nums.append(_num)

                    fold_num = 5
                    # For basic surrogate i=K. cv
                    if len(test_y) < 2 * fold_num or self.increasing_weight:
                        order_preseving_nums.append(0)
                    else:
                        # 5-fold cross validation.
                        kfold = KFold(n_splits=fold_num)
                        cv_pred = np.array([0] * len(test_y))
                        for train_idx, valid_idx in kfold.split(test_x):    # todo: reduce cost!!!
                            train_configs, train_y = test_x[train_idx], test_y[train_idx]
                            valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                            types, bounds = get_types(self.config_space)
                            if self.surrogate_type == 'prf':
                                _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                            elif self.surrogate_type == 'gp':
                                _surrogate = create_gp_model('gp', self.config_space, types, bounds, self.rng)
                            else:
                                raise ValueError('Unknown surrogate type: %s' % self.surrogate_type)
                            _surrogate.train(train_configs, train_y)
                            _pred, _var = _surrogate.predict(valid_configs)
                            sampled_pred = self.rng.normal(_pred.reshape(-1), _var.reshape(-1))
                            cv_pred[valid_idx] = sampled_pred
                        _num, _ = self.calculate_preserving_order_num(cv_pred, test_y)
                        order_preseving_nums.append(_num)
                    max_id = np.argmax(order_preseving_nums)
                    min_probability_array[max_id] += 1
                new_weights = np.array(min_probability_array) / sample_num
                t3 = time.time()
                if t3 - t1 > 1:
                    self.logger.info('update weight (rank loss prob) cost time: %.2fs, %.2fs'
                                     % (t2 - t1, t3 - t2))
            else:
                raise ValueError('Invalid weight method: %s!' % self.weight_method)
        else:
            new_weights = np.array(old_weights)

        # non decreasing full observation weight
        old_weights = np.asarray(old_weights)
        new_weights = np.asarray(new_weights)
        self.hist_weights_unadjusted.append(new_weights)
        if self.non_decreasing_weight:
            old_last_weight = old_weights[-1]
            new_last_weight = new_weights[-1]
            if new_last_weight < old_last_weight:
                old_remain_weight = 1.0 - old_last_weight
                new_remain_weight = 1.0 - new_last_weight
                if new_remain_weight <= 1e-8:
                    adjusted_new_weights = np.array([0.] * self.s_max + [1.], dtype=np.float64)
                else:
                    adjusted_new_weights = np.append(new_weights[:-1] / new_remain_weight * old_remain_weight,
                                                     old_last_weight)
                self.logger.info('[%s] %d-th. non_decreasing_weight: old_weights=%s, new_weights=%s, '
                                 'adjusted_new_weights=%s.' % (self.weight_method, self.weight_changed_cnt,
                                                               old_weights, new_weights, adjusted_new_weights))
                new_weights = adjusted_new_weights
        elif self.increasing_weight and len(test_y) >= 10:
            s = 10
            k = 0.025
            a = 0.5
            new_last_weight = a / (a + np.e ** (-(len(test_y) - s) * k))
            new_remain_weight = 1.0 - new_last_weight
            remain_weight = 1.0 - new_weights[-1]
            if remain_weight <= 1e-8:
                adjusted_new_weights = np.array([0.] * self.s_max + [1.], dtype=np.float64)
            else:
                adjusted_new_weights = np.append(new_weights[:-1] / remain_weight * new_remain_weight,
                                                 new_last_weight)
            self.logger.info('[%s] %d-th. increasing_weight: new_weights=%s, adjusted_new_weights=%s.'
                             % (self.weight_method, self.weight_changed_cnt, new_weights, adjusted_new_weights))
            new_weights = adjusted_new_weights

        self.logger.info('[%s] %d-th Updating weights: %s' % (
            self.weight_method, self.weight_changed_cnt, str(new_weights)))

        # Assign the weight to each basic surrogate.
        if not self.test_bohb:
            for i, r in enumerate(r_list):
                self.surrogate.surrogate_weight[r] = new_weights[i]
        self.weight_changed_cnt += 1
        # Save the weight data.
        self.hist_weights.append(new_weights)
        dir_path = os.path.join(self.data_directory, 'saved_weights')
        file_name = 'mfes_weights_%s.npy' % (self.method_name,)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.save(os.path.join(dir_path, file_name), np.asarray(self.hist_weights))
        self.logger.info('update_weight() cost %.2fs. new weights are saved to %s'
                         % (time.time() - start_time, os.path.join(dir_path, file_name)))

    def get_weights(self):
        return self.hist_weights
