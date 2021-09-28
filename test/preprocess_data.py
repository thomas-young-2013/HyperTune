import os
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split


def load_dataset(dataset, data_dir):
    """
    no label encoding
    """
    data_path = os.path.join(data_dir, "%s.csv" % dataset)

    # Load train data.
    if dataset in ['higgs', 'amazon_employee', 'spectf', 'usps', 'vehicle_sensIT', 'codrna', 'HIGGS']:
        label_col = 0
    elif dataset in ['rmftsa_sleepdata(1)']:
        label_col = 1
    else:
        label_col = -1

    if dataset in ['spambase', 'messidor_features', 'covtype', 'HIGGS']:
        header = None
    else:
        header = 'infer'

    if dataset in ['winequality_white', 'winequality_red']:
        sep = ';'
    else:
        sep = ','

    na_values = ["n/a", "na", "--", "-", "?"]
    keep_default_na = True
    df = pd.read_csv(data_path, keep_default_na=keep_default_na,
                     na_values=na_values, header=header, sep=sep)

    # Drop the row with all NaNs.
    df.dropna(how='all')

    # Clean the data where the label columns have nans.
    columns_missed = df.columns[df.isnull().any()].tolist()

    label_colname = df.columns[label_col]

    if label_colname in columns_missed:
        labels = df[label_colname].values
        row_idx = [idx for idx, val in enumerate(labels) if np.isnan(val)]
        # Delete the row with NaN label.
        df.drop(df.index[row_idx], inplace=True)

    train_y = df[label_colname].values

    # Delete the label column.
    df.drop(label_colname, axis=1, inplace=True)

    train_X = df
    return train_X, train_y


data_dir = './datasets'
datasets = ['HIGGS', 'covtype', 'pokerhand']

new_data_dir = 'datasets'
if not os.path.exists(new_data_dir):
    os.makedirs(new_data_dir)

for dataset in datasets:
    x, y = load_dataset(dataset, data_dir)
    print(dataset, 'loaded', x.shape, y.shape)

    # split. train : validate : test = 6 : 2 : 2
    xx, x_test, yy, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(xx, yy, test_size=0.25, stratify=yy, random_state=1)
    print(dataset, 'split', x_train.shape[0], x_val.shape[0], x_test.shape[0])

    # save
    if dataset == 'codrna':
        name = dataset + '.pkl'
        obj = (x_train, x_val, x_test, y_train, y_val, y_test)
        with open(os.path.join(new_data_dir, name), 'wb') as f:
            pkl.dump(obj, f)
    else:
        name_x_train = dataset + '-x_train.npy'
        name_x_val = dataset + '-x_val.npy'
        name_x_test = dataset + '-x_test.npy'
        name_y_train = dataset + '-y_train.npy'
        name_y_val = dataset + '-y_val.npy'
        name_y_test = dataset + '-y_test.npy'
        np.save(os.path.join(new_data_dir, name_x_train), x_train)
        np.save(os.path.join(new_data_dir, name_x_val), x_val)
        np.save(os.path.join(new_data_dir, name_x_test), x_test)
        np.save(os.path.join(new_data_dir, name_y_train), y_train)
        np.save(os.path.join(new_data_dir, name_y_val), y_val)
        np.save(os.path.join(new_data_dir, name_y_test), y_test)
    print(dataset, 'finished')
