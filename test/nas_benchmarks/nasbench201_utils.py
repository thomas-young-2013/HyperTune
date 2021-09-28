import time
import numpy as np
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space import CategoricalHyperparameter
from nas_201_api import NASBench201API as API


OP_NUM = 6
MAX_IEPOCH = 200


def load_nasbench201(path='../nas_data/NAS-Bench-201-v1_1-096897.pth'):
    s = time.time()
    api = API(path, verbose=False)
    print('nas-bench-201 load time: %.2fs' % (time.time() - s))
    return api


def get_nasbench201_configspace():
    op_list = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    cs = ConfigurationSpace()
    for i in range(OP_NUM):
        cs.add_hyperparameter(CategoricalHyperparameter('op%d' % i, choices=op_list, default_value='none'))
    return cs


def objective_func(config, n_resource, extra_conf, total_resource, eta, api, dataset):
    assert dataset in ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
    print('objective extra conf:', extra_conf)
    iepoch = int(MAX_IEPOCH * n_resource / total_resource) - 1

    # convert config to arch
    arch = '|%s~0|+|%s~0|%s~1|+|%s~0|%s~1|%s~2|' % (config['op0'],
                                                    config['op1'], config['op2'],
                                                    config['op3'], config['op4'], config['op5'])

    # query
    info = api.get_more_info(arch, dataset, iepoch=iepoch, hp='200', is_random=False)
    train_time = info['train-all-time']
    if dataset == 'cifar10-valid':
        val_perf = info['valid-accuracy']
        test_perf = info.get('test-accuracy', None)
    elif dataset == 'cifar10':
        val_perf = info['test-accuracy']
        test_perf = None
    elif dataset == 'cifar100':
        val_perf = info['valtest-accuracy']
        test_perf = info.get('test-accuracy', None)
    elif dataset == 'ImageNet16-120':
        val_perf = info['valtest-accuracy']
        test_perf = info.get('test-accuracy', None)
    else:
        raise ValueError

    # Get checkpoint info
    if extra_conf['initial_run']:
        last_train_time = 0.0
    else:
        last_iepoch = int(MAX_IEPOCH * (n_resource / eta) / total_resource) - 1
        last_info = api.get_more_info(arch, dataset, iepoch=last_iepoch, hp='200', is_random=False)
        last_train_time = last_info['train-all-time']

    # restore from checkpoint
    train_time = train_time - last_train_time

    result = dict(
        objective_value=-val_perf,  # minimize
        test_perf=-test_perf if test_perf is not None else None,  # minimize
        elapsed_time=train_time,
    )
    return result


if __name__ == '__main__':
    cs = get_nasbench201_configspace()
    for i in range(3):
        conf = cs.sample_configuration()
        print(conf)

    test_load = False
    if test_load:
        api = load_nasbench201('../nas_data/NAS-Bench-201-v1_1-096897.pth')
        conf = cs.sample_configuration()
        print(conf)
        dataset = 'cifar10-valid'
        extra_conf = dict(initial_run=True)
        result = objective_func(conf, 3, extra_conf, total_resource=27, eta=3, api=api, dataset=dataset)
        print(result)
