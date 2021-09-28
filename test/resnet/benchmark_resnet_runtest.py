"""
example cmdline:

python test/resnet/benchmark_resnet_runtest.py --mth hyperband-n4 --rep 1 --start_id 0

"""
import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl

sys.path.insert(0, ".")
sys.path.insert(1, "../open-box")    # for dependency
from test.utils import seeds

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--mth', type=str, default='hyperband-n4')
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--runtime_limit', type=int, default=172800)

args = parser.parse_args()
dataset = args.dataset
mth = args.mth
rep = args.rep
start_id = args.start_id
runtime_limit = args.runtime_limit
model = 'resnet'

try:
    from sklearn.metrics.scorer import accuracy_scorer
except ModuleNotFoundError:
    from sklearn.metrics._scorer import accuracy_scorer
    print('from sklearn.metrics._scorer import accuracy_scorer')
from resnet_model import get_estimator
from resnet_util import get_transforms
from resnet_dataset import ImageDataset
from resnet_obj import dl_holdout_validation

from openbox.utils.constants import MAXINT

# Constant
max_epoch = 200
scorer = accuracy_scorer
image_size = 32
data_dir = './datasets/img_datasets/cifar10/'
image_data = ImageDataset(data_path=data_dir, train_val_split=True, image_size=image_size)


def test_func(config, device='cuda'):    # device='cuda' 'cuda:0'

    data_transforms = get_transforms(image_size=image_size)
    image_data.load_data(data_transforms['train'], data_transforms['val'])
    # load test
    image_data.set_test_path(data_dir)
    image_data.load_test_data(data_transforms['val'])
    start_time = time.time()

    config_dict = config.get_dictionary().copy()

    estimator = get_estimator(config_dict, max_epoch, device=device)

    estimator.epoch_num = estimator.max_epoch

    try:
        score = dl_holdout_validation(estimator, scorer, image_data, random_state=1, run_test=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        score = -MAXINT
    print('Evaluation | Score: %.4f | Time cost: %.2f seconds' %
          (scorer._sign * score,
           time.time() - start_time))
    print(str(config))

    # Turn it into a minimization problem.
    return -score


print('===== start test %s %s: rep=%d' % (mth, dataset, rep))
for i in range(start_id, start_id + rep):
    seed = seeds[i]

    dir_path = 'data/benchmark_%s/%s-%d/%s/' % (model, dataset, runtime_limit, mth)
    for file in os.listdir(dir_path):
        if file.startswith('incumbent_new_record_%s-%s-%d-' % (mth, dataset, seed)) \
                and file.endswith('.pkl'):
            # load config
            with open(os.path.join(dir_path, file), 'rb') as f:
                record = pkl.load(f)
            print(model, dataset, mth, seed, 'loaded!', record, flush=True)

            # run test
            config = record['configuration']
            perf = test_func(config, device='cuda')
            print(model, dataset, mth, seed, 'perf =', perf)

            # save perf
            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            method_id = mth + '-%s-%d-%s' % (dataset, seed, timestamp)
            perf_file_name = 'incumbent_test_perf_%s.pkl' % (method_id,)
            with open(os.path.join(dir_path, perf_file_name), 'wb') as f:
                pkl.dump(perf, f)
            print(dir_path, perf_file_name, 'saved!', flush=True)
