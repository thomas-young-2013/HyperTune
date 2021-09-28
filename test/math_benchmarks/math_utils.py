import numpy as np
from functools import partial
from test.math_benchmarks.so_benchmark_function import get_problem


def get_math_obj_func_cs(total_resource, eta, problem_str, noise_alpha, rng, problem_kwargs):
    problem = get_problem(problem_str, **problem_kwargs)
    assert hasattr(problem, 'evaluate_config')
    if problem_str == 'hartmann':
        noise_scale = 0.38
    else:
        noise_scale = 1.0
    if problem_str.startswith('counting'):
        obj_func = partial(mf_objective_func_math, total_resource=total_resource, eta=eta,
                           problem=problem, continue_training=False)
    else:
        obj_func = partial(mf_objective_func_math_noise, total_resource=total_resource, eta=eta,
                           problem=problem, noise_scale=noise_scale, noise_alpha=noise_alpha, rng=rng)
    cs = problem.get_configspace()
    return obj_func, cs


def mf_objective_func_math_noise(
        config, n_resource, extra_conf,
        total_resource, eta, problem, noise_scale, noise_alpha, rng: np.random.RandomState):
    print('objective extra conf:', extra_conf)

    # noise_level = np.log(total_resource / n_resource) / np.log(eta)
    noise_level = 1 / n_resource - 1 / total_resource
    # noise_level = (1 / (1 + np.e ** (-x)) - 0.5) * 2

    original_perf = problem.evaluate_config(config)
    noise = rng.normal(scale=noise_level * noise_scale * noise_alpha)
    perf = original_perf + noise
    print('config: %s, resource: %f/%f, noise_level: %f. perf=%f+%f=%f'
          % (config, n_resource, total_resource, noise_level, original_perf, noise, perf))

    eval_time = 27 * n_resource / total_resource
    if not extra_conf['initial_run']:
        eval_time -= 27 * n_resource / eta / total_resource

    result = dict(
        objective_value=perf,  # minimize
        elapsed_time=eval_time,
    )
    return result


def mf_objective_func_math(config, n_resource, extra_conf, total_resource, eta, problem, continue_training):
    print('objective extra conf:', extra_conf)

    fidelity = n_resource / total_resource
    perf = problem.evaluate_config(config, fidelity=fidelity)
    print('config: %s, resource: %f/%f, perf=%f'
          % (config, n_resource, total_resource, perf))

    eval_time = 270 * n_resource / total_resource
    if continue_training and not extra_conf['initial_run']:
        eval_time -= 270 * n_resource / eta / total_resource

    result = dict(
        objective_value=perf,  # minimize
        elapsed_time=eval_time,
    )
    return result
