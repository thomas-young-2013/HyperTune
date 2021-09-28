import time
import os
import warnings
import torch
from math import ceil, log
import numpy as np
try:
    from sklearn.metrics.scorer import accuracy_scorer
except ModuleNotFoundError:
    from sklearn.metrics._scorer import accuracy_scorer
    print('from sklearn.metrics._scorer import accuracy_scorer')
from resnet_model import get_estimator
from resnet_util import get_path_by_config, get_transforms
from resnet_dataset import ImageDataset

from openbox.utils.constants import MAXINT

# Constant
max_epoch = 200
scorer = accuracy_scorer
image_size = 32
data_dir = './datasets/img_datasets/cifar10/'
image_data = ImageDataset(data_path=data_dir, train_val_split=True, image_size=image_size)

from resnet_model import ResNet32Classifier

test_config = ResNet32Classifier.get_hyperparameter_search_space().get_default_configuration()


def mf_objective_func_gpu(config, n_resource, extra_conf, device, total_resource, run_test=False,
                          model_dir='./data/resnet_save_models/unnamed_trial', eta=3):    # device='cuda' 'cuda:0'
    print('extra_conf:', extra_conf)
    initial_run = extra_conf['initial_run']
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except FileExistsError:
        pass

    data_transforms = get_transforms(image_size=image_size)
    image_data.load_data(data_transforms['train'], data_transforms['val'])
    start_time = time.time()

    config_dict = config.get_dictionary().copy()

    estimator = get_estimator(config_dict, max_epoch, device=device)

    epoch_ratio = float(n_resource) / float(total_resource)

    config_model_path = os.path.join(model_dir,
                                     'tmp_' + get_path_by_config(config) + '_%d' % int(n_resource / eta) + '.pt')
    save_path = os.path.join(model_dir,
                             'tmp_' + get_path_by_config(config) + '_%d' % int(n_resource) + '.pt')

    # Continue training if initial_run=False
    if not initial_run:
        if not os.path.exists(config_model_path):
            raise ValueError('not initial_run but config_model_path not exists. check if exists duplicated configs '
                             'and saved model were removed.')
        estimator.epoch_num = ceil(estimator.max_epoch * epoch_ratio) - ceil(
            estimator.max_epoch * epoch_ratio / eta)
        estimator.load_path = config_model_path
        print(estimator.epoch_num)
    else:
        estimator.epoch_num = ceil(estimator.max_epoch * epoch_ratio)

    try:
        score = dl_holdout_validation(estimator, scorer, image_data, random_state=1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        score = -MAXINT
    print('Evaluation | Score: %.4f | Time cost: %.2f seconds' %
          (scorer._sign * score,
           time.time() - start_time))
    print(str(config))

    # Save low-resource models
    if np.isfinite(score) and epoch_ratio != 1.0:
        state = {'model': estimator.model.state_dict(),
                 'optimizer': estimator.optimizer_.state_dict(),
                 'scheduler': estimator.scheduler.state_dict(),
                 'cur_epoch_num': estimator.cur_epoch_num}
        torch.save(state, save_path)

    try:
        if epoch_ratio == 1:
            s_max = int(log(total_resource) / log(eta))
            for i in range(0, s_max + 1):
                if os.path.exists(os.path.join(model_dir,
                                               'tmp_' + get_path_by_config(config) + '_%d' % int(eta ** i) + '.pt')):
                    os.remove(os.path.join(model_dir,
                                           'tmp_' + get_path_by_config(config) + '_%d' % int(eta ** i) + '.pt'))
    except Exception as e:
        print('unexpected exception!')
        import traceback
        traceback.print_exc()

    # if np.isfinite(score):
    #     save_flag, model_path, delete_flag, model_path_deleted = self.topk_model_saver.add(config, score)
    #     if save_flag is True:
    #         state = {'model': estimator.model.state_dict(),
    #                  'optimizer': estimator.optimizer_.state_dict(),
    #                  'scheduler': estimator.scheduler.state_dict(),
    #                  'cur_epoch_num': estimator.cur_epoch_num,
    #                  'early_stop': estimator.early_stop}
    #         torch.save(state, model_path)
    #         print("Model saved to %s" % model_path)
    #
    #     # In case of double-deletion
    #     try:
    #         if delete_flag and os.path.exists(model_path_deleted):
    #             os.remove(model_path_deleted)
    #             print("Model deleted from %s" % model_path)
    #     except:
    #         pass

    # Turn it into a minimization problem.
    result = dict(
        objective_value=-score,
    )
    return result


def dl_holdout_validation(estimator, scorer, dataset, random_state=1, run_test=False, **kwargs):
    start_time = time.time()
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        estimator.fit(dataset, **kwargs)
        if 'profile_epoch' in kwargs or 'profile_iter' in kwargs:
            return time.time() - start_time
        else:
            return scorer._sign * estimator.score(dataset, scorer._score_func, run_test=run_test)


if __name__ == '__main__':
    extra_conf = dict(initial_run=True)
    mf_objective_func_gpu(config=test_config, n_resource=27, extra_conf=extra_conf, device='cuda', total_resource=81)
