import time
import numpy as np
import traceback
import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import scipy.stats as sps
import statsmodels.api as sm

from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.util_funcs import get_types
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.history_container import HistoryContainer
from openbox.core.base import Observation
from openbox.utils.constants import MAXINT, SUCCESS

from tuner.mq_hb import mqHyperband
from tuner.utils import sample_configurations, expand_configurations


class mqBOHB_v2(mqHyperband):
    """ The implementation of BOHB.
        The paper can be found in https://arxiv.org/abs/1807.01774 .
        v2: using TPE
    """

    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 num_iter=10000,
                 rand_prob=0.3,
                 min_points_in_model=None,
                 top_n_percent=15,
                 bandwidth_factor=3,
                 min_bandwidth=1e-3,
                 num_samples=5000,
                 random_state=1,
                 method_id='mqBOHB',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',):
        super().__init__(objective_func, config_space, R, eta=eta, num_iter=num_iter,
                         random_state=random_state, method_id=method_id,
                         restart_needed=restart_needed, time_limit_per_trial=time_limit_per_trial,
                         runtime_limit=runtime_limit,
                         ip=ip, port=port, authkey=authkey)

        self.random_fraction = rand_prob
        self.rng = np.random.RandomState(self.seed)

        self.history_container = HistoryContainer(task_id=method_id)

        self.num_samples = num_samples
        self.top_n_percent = top_n_percent
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth

        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.config_space.get_hyperparameters()) + 1

        if self.min_points_in_model < len(self.config_space.get_hyperparameters()) + 1:
            self.min_points_in_model = len(self.config_space.get_hyperparameters()) + 1

        hps = self.config_space.get_hyperparameters()

        self.kde_vartypes = ""
        self.vartypes = []

        for h in hps:
            if hasattr(h, 'choices'):
                self.kde_vartypes += 'u'
                self.vartypes += [len(h.choices)]
            else:
                self.kde_vartypes += 'c'
                self.vartypes += [0]

        self.vartypes = np.array(self.vartypes, dtype=int)

        # store precomputed probs for the categorical parameters
        self.cat_probs = []

        self.good_config_rankings = dict()
        self.kde_models = dict()

    def choose_next(self, num_config):
        start_time = time.time()
        # Sample n configurations according to BOHB strategy.
        self.logger.info('Sample %d configs in choose_next. rand_prob is %f.' % (num_config, self.random_fraction))

        # fit
        self.fit_kde_models(self.history_container)

        # If no model is available, sample random configs
        if len(self.kde_models.keys()) == 0:
            self.logger.info('len(self.incumbent_configs) = %d. Return all random configs.'
                             % (len(self.incumbent_configs),))
            return sample_configurations(self.config_space, num_config, excluded_configs=self.incumbent_configs)

        l = self.kde_models['good'].pdf
        g = self.kde_models['bad'].pdf

        minimize_me = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)

        kde_good = self.kde_models['good']
        kde_bad = self.kde_models['bad']

        vector_acq = []
        for i in range(self.num_samples):
            try:
                idx = self.rng.randint(0, len(kde_good.data))
                datum = kde_good.data[idx]
                vector = []

                for m, bw, t in zip(datum, kde_good.bw, self.vartypes):

                    bw = max(bw, self.min_bandwidth)
                    if t == 0:
                        bw = self.bw_factor * bw
                        try:
                            vector.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
                        except:
                            self.logger.warning(
                                "Truncated Normal failed for:\ndatum=%s\nbandwidth=%s\nfor entry with value %s" % (
                                    datum, kde_good.bw, m))
                            self.logger.warning("data in the KDE:\n%s" % kde_good.data)
                    else:

                        if self.rng.rand() < (1 - bw):
                            vector.append(int(m))
                        else:
                            vector.append(self.rng.randint(t))
                val = minimize_me(vector)

                if not np.isfinite(val):
                    self.logger.warning('sampled vector: %s has EI value %s' % (vector, val))
                    self.logger.warning("data in the KDEs:\n%s\n%s" % (kde_good.data, kde_bad.data))
                    self.logger.warning("bandwidth of the KDEs:\n%s\n%s" % (kde_good.bw, kde_bad.bw))
                    self.logger.warning("l(x) = %s" % (l(vector)))
                    self.logger.warning("g(x) = %s" % (g(vector)))

                    # right now, this happens because a KDE does not contain all values for a categorical parameter
                    # this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this one
                    # if the good_kde has a finite value, i.e. there is no config with that value in the bad kde, so it shouldn't be terrible.
                    # if np.isfinite(l(vector)):
                    #     best_vector = vector
                    #     break
                    continue
            except:
                continue
            else:
                vector_acq.append((vector, val))

        vector_acq.sort(key=lambda x: x[1])

        num_bo_config = num_config - int(num_config * self.random_fraction)
        config_candidates = []
        for vector, val in vector_acq:
            if len(config_candidates) >= num_bo_config:
                break
            # convert vector to config
            for i, hp_value in enumerate(vector):
                if isinstance(
                        self.config_space.get_hyperparameter(
                            self.config_space.get_hyperparameter_by_idx(i)
                        ),
                        ConfigSpace.hyperparameters.CategoricalHyperparameter
                ):
                    vector[i] = int(np.rint(vector[i]))
            try:
                config = ConfigSpace.Configuration(self.config_space, vector=vector)
            except Exception as e:
                self.logger.warning(("=" * 50 + "\n") * 3 +
                                    "Error converting configuration:\n%s" % vector +
                                    "\n here is a traceback:" +
                                    traceback.format_exc())
                continue
            config_candidates.append(config)
        self.logger.info('len bo configs = %d.' % len(config_candidates))

        # sample random configs
        config_candidates = expand_configurations(config_candidates, self.config_space, num_config,
                                                  excluded_configs=self.incumbent_configs)
        self.logger.info('len total configs = %d.' % len(config_candidates))
        assert len(config_candidates) == num_config
        self.logger.info('choose_next cost %.2fs.' % (time.time() - start_time))
        return config_candidates

    def update_incumbent_before_reduce(self, T, val_losses, n_iteration):
        if int(n_iteration) < self.R:
            return
        self.incumbent_configs.extend(T)
        self.incumbent_perfs.extend(val_losses)
        for config, perf in zip(T, val_losses):
            observation = Observation(
                config=config, objs=[perf], constraints=None,
                trial_state=SUCCESS, elapsed_time=None,
            )
            self.history_container.update_observation(observation)
        self.logger.info('%d observations updated. %d incumbent configs total.' % (len(T), len(self.incumbent_configs)))

    def update_incumbent_after_reduce(self, T, incumbent_loss):
        return

    def fit_kde_models(self, history_container):
        num_config_successful = len(history_container.successful_perfs)
        if num_config_successful <= self.min_points_in_model - 1:
            self.logger.debug("Only %i run(s) available, need more than %s -> can't build model!" % (
                num_config_successful, self.min_points_in_model + 1))
            return

        train_configs = convert_configurations_to_array(history_container.configurations)
        train_losses = history_container.get_transformed_perfs()

        n_good = max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0]) // 100)
        # n_bad = min(max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100), 10)
        n_bad = max(self.min_points_in_model, ((100 - self.top_n_percent) * train_configs.shape[0]) // 100)

        # Refit KDE for the current budget
        idx = np.argsort(train_losses)

        train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
        train_data_bad = self.impute_conditional_data(train_configs[idx[n_good:n_good + n_bad]])

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return

        # more expensive crossvalidation method
        # bw_estimation = 'cv_ls'

        # quick rule of thumb
        bw_estimation = 'normal_reference'

        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=self.kde_vartypes,
                                                   bw=bw_estimation)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde_vartypes,
                                                    bw=bw_estimation)

        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        self.kde_models = {
            'good': good_kde,
            'bad': bad_kde
        }

        # update probs for the categorical parameters for later sampling
        self.logger.debug(
            'done building a new model based on %i/%i split\nBest loss for this budget:%f\n\n\n\n\n' % (
                n_good, n_bad, np.min(train_losses)))

    def impute_conditional_data(self, array):

        return_array = np.empty_like(array)

        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()

            while np.any(nan_indices):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = self.rng.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = self.rng.rand()
                    else:
                        datum[nan_idx] = self.rng.randint(t)

                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return return_array
