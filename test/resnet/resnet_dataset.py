import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torchvision import datasets


def get_folder_dataset(folder_path, udf_transforms=None, grayscale=False):
    return datasets.ImageFolder(folder_path, transform=udf_transforms)


class SubsetSequentialampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BaseDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.test_data_path = None

    def load_data(self):
        raise NotImplementedError()

    def load_test_data(self):
        raise NotImplementedError()

    def set_test_path(self, test_data_path):
        self.test_data_path = test_data_path


class DLDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.train_sampler, self.val_sampler = None, None
        self.subset_sampler_used = False
        self.train_indices, self.val_indices = None, None

    def create_train_val_split(self, dataset: Dataset, train_val_split=0.2, shuffle=True):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        test_split = int(np.floor(train_val_split * dataset_size))

        if shuffle:
            np.random.seed(1)
            np.random.shuffle(indices)

        self.val_indices, self.train_indices = indices[:test_split], indices[test_split:]

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetSequentialampler(self.val_indices)
        self.subset_sampler_used = True

    def get_train_samples_num(self):
        raise NotImplementedError()

    def get_train_val_indices(self):
        return self.train_indices, self.val_indices

    def get_loader_labels(self, loader: DataLoader):
        labels = list()
        for i, data in enumerate(loader):
            if len(data) != 2:
                raise ValueError('No labels found!')
            labels.extend(list(data[1]))
        return np.asarray(labels)

    def get_labels(self, mode='val'):
        if mode == 'val':
            if self.subset_sampler_used:
                loader = DataLoader(dataset=self.train_dataset, batch_size=32,
                                    sampler=self.val_sampler, num_workers=4)
                return self.get_loader_labels(loader)
            else:
                loader = DataLoader(dataset=self.val_dataset, batch_size=32, shuffle=False,
                                    sampler=None, num_workers=4)
                return self.get_loader_labels(loader)
        elif mode == 'train':
            if self.subset_sampler_used:
                loader = DataLoader(dataset=self.train_dataset, batch_size=32,
                                    sampler=self.train_sampler, num_workers=4)
                return self.get_loader_labels(loader)
            else:
                loader = DataLoader(dataset=self.train_dataset, batch_size=32, shuffle=False,
                                    sampler=None, num_workers=4)
                return self.get_loader_labels(loader)
        else:
            loader = DataLoader(dataset=self.test_dataset, batch_size=32, shuffle=False,
                                num_workers=4)
            return self.get_loader_labels(loader)


class ImageDataset(DLDataset):
    def __init__(self, data_path: str,
                 data_transforms: dict = None,
                 grayscale: bool = False,
                 train_val_split: bool = False,
                 image_size=32,
                 val_split_size: float = 0.2):
        super().__init__()
        self.train_val_split = train_val_split
        self.val_split_size = val_split_size
        self.data_path = data_path

        self.udf_transforms = data_transforms
        self.grayscale = grayscale
        self.image_size = image_size

        default_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'))
        self.classes = default_dataset.classes

    def load_data(self, train_transforms, val_transforms):
        # self.means, self.var = self.get_mean_and_var()
        self.train_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'),
                                                udf_transforms=train_transforms,
                                                grayscale=self.grayscale)
        if not self.train_val_split:
            self.val_dataset = get_folder_dataset(os.path.join(self.data_path, 'val'),
                                                  udf_transforms=val_transforms,
                                                  grayscale=self.grayscale)
        else:
            self.train_for_val_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'),
                                                            udf_transforms=val_transforms,
                                                            grayscale=self.grayscale)
            self.create_train_val_split(self.train_dataset, train_val_split=self.val_split_size, shuffle=True)

    def load_test_data(self, transforms):
        self.test_dataset = get_folder_dataset(os.path.join(self.test_data_path, 'test'),
                                               udf_transforms=transforms,
                                               grayscale=self.grayscale)
        self.test_dataset.classes = self.classes

    def get_train_samples_num(self):
        if self.train_dataset is None:
            _train_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'),
                                                udf_transforms=None,
                                                grayscale=self.grayscale)
            _train_size = len(_train_dataset)
        else:
            _train_size = len(self.train_dataset)
        if self.subset_sampler_used:
            return _train_size * (1 - self.val_split_size)
        else:
            return _train_size

    def get_mean_and_var(self):
        basic_transforms = transforms.Compose([
            transforms.ToTensor()])
        _train_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'),
                                            udf_transforms=basic_transforms)

        dataloader = torch.utils.data.DataLoader(_train_dataset, batch_size=1, shuffle=True, num_workers=2)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        print('==> Computing mean and std..')
        for inputs, targets in dataloader:
            for i in range(3):
                mean[i] += inputs[:, i, :, :].mean()
                std[i] += inputs[:, i, :, :].std()
        mean.div_(len(_train_dataset))
        std.div_(len(_train_dataset))
        mean = mean.numpy()
        std = std.numpy()
        return mean, std
