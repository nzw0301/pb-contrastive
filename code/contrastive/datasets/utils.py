import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split as sk_train_val_split


def get_class2samples(dataset: torch.utils.data.Dataset, num_classes: int) -> list:
    """
    Create list such that each index is corresponding to the class_id, and each element is a tensor of data.

    :param dataset: CIFAR100/AUSLAN dataset's instance.
    :param num_classes: The number of classes.

    :return: list contains tensors per class.
    """
    label2samples = [[] for _ in range(num_classes)]

    for sample, label in dataset:
        label2samples[label].append(sample)

    for label, sample in enumerate(label2samples):
        label2samples[label] = torch.stack(sample)

    return label2samples


def _train_val_split(
        rnd: np.random.RandomState,
        train_dataset: torch.utils.data.Dataset,
        validation_ratio=0.05
) -> tuple:
    """
    Apply sklearn's `train_val_split` function to PyTorch's dataset instance.

    :param rnd: `np.random.RandomState` instance.
    :param train_dataset: Training set. This is an instance of PyTorch's dataset.
    :param validation_ratio: The ratio of validation data.

    :return: Tuple of training set and validation set.
    """

    x_train, x_val, y_train, y_val = sk_train_val_split(
        train_dataset.data, train_dataset.targets, test_size=validation_ratio,
        random_state=rnd, stratify=train_dataset.targets
    )

    val_dataset = copy.deepcopy(train_dataset)

    train_dataset.data = x_train
    train_dataset.targets = y_train

    val_dataset.data = x_val
    val_dataset.targets = y_val

    return train_dataset, val_dataset
