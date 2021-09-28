import numpy as np
from math import log
from typing import List
from openbox.utils.config_space import Configuration, ConfigurationSpace
from openbox.utils.config_space.util import impute_default_values
from openbox.surrogate.base.gp import GaussianProcess
from openbox.surrogate.base.gp_mcmc import GaussianProcessMCMC
from openbox.surrogate.base.gp_base_prior import HorseshoePrior, LognormalPrior
from openbox.surrogate.base.gp_kernels import ConstantKernel, Matern, HammingKernel, WhiteKernel, RBF


def convert_configurations_to_resource_array(configs: List[Configuration], resources: List[int],
                                             max_resource: int) -> np.ndarray:
    """Impute inactive hyperparameters in configurations with their default. Add a feature for amount of resources

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configs : List[Configuration]
        List of configuration objects.
    resources: List[int]
        List of configuration resources.
    max_resource: int
        The maximum amount of resources.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    configs_array = np.array([config.get_array() for config in configs],
                             dtype=np.float64)
    configuration_space = configs[0].configuration_space
    config_features = impute_default_values(configuration_space, configs_array)
    resource_features = np.array([[log(resource) / log(max_resource)] for resource in resources])
    result = np.hstack([config_features, resource_features])
    return result


def create_resource_gp_model(model_type, config_space, types, bounds, rng):
    """
        Construct the Gaussian process surrogate that is capable of dealing with categorical hyperparameters.
    """
    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(np.exp(-10), np.exp(2)),
        prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
    )

    # resource feature
    types = np.hstack((types, [0])).astype(int)
    bounds = np.vstack((bounds, [[0.0, 1.0]])).astype(float)

    cont_dims = np.nonzero(types == 0)[0].astype(np.int)
    cat_dims = np.nonzero(types != 0)[0].astype(np.int)

    if len(cont_dims) > 0:
        exp_kernel = Matern(
            np.ones([len(cont_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cont_dims))],
            nu=2.5,
            operate_on=cont_dims,
        )

    if len(cat_dims) > 0:
        ham_kernel = HammingKernel(
            np.ones([len(cat_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cat_dims))],
            operate_on=cat_dims,
        )

    noise_kernel = WhiteKernel(
        noise_level=1e-8,
        noise_level_bounds=(np.exp(-25), np.exp(2)),
        prior=HorseshoePrior(scale=0.1, rng=rng),
    )

    if len(cont_dims) > 0 and len(cat_dims) > 0:
        # both
        kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
    elif len(cont_dims) > 0 and len(cat_dims) == 0:
        # only cont
        kernel = cov_amp * exp_kernel + noise_kernel
    elif len(cont_dims) == 0 and len(cat_dims) > 0:
        # only cont
        kernel = cov_amp * ham_kernel + noise_kernel
    else:
        raise ValueError()

    # seed = rng.randint(0, 2 ** 20)
    if model_type == 'gp_mcmc':
        n_mcmc_walkers = 3 * len(kernel.theta)
        if n_mcmc_walkers % 2 == 1:
            n_mcmc_walkers += 1
        model = GaussianProcessMCMC(
            configspace=config_space,
            types=types,
            bounds=bounds,
            kernel=kernel,
            n_mcmc_walkers=n_mcmc_walkers,
            chain_length=250,
            burnin_steps=250,
            normalize_y=True,
            seed=rng.randint(low=0, high=10000),
        )
    elif model_type == 'gp':
        model = GaussianProcess(
            configspace=config_space,
            types=types,
            bounds=bounds,
            kernel=kernel,
            normalize_y=True,
            seed=rng.randint(low=0, high=10000),
        )
    else:
        raise ValueError("Invalid surrogate str %s!" % model_type)
    return model

