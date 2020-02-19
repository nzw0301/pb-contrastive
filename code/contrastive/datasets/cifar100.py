import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from .utils import _train_val_split


def get_train_val_test_datasets(
        rnd: np.random.RandomState,
        root='~/data',
        validation_ratio=0.05,
) -> tuple:
    """
    Create CIFAR-100 train/val/test data loaders

    :param rnd: `np.random.RandomState` instance.
    :param validation_ratio: The ratio of validation data. If this value is `0.`, returned `val_set` is `None`.
    :param root: Path to save data.

    :return: Tuple of (train, val, test) or (train, test).
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    train_set = CIFAR100(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    # create validation split
    if validation_ratio > 0.:
        train_set, val_set = _train_val_split(rnd=rnd, train_dataset=train_set, validation_ratio=validation_ratio)

    # create a transform to do pre-processing
    train_loader = DataLoader(
        train_set,
        batch_size=len(train_set),
        shuffle=False,
    )

    data = iter(train_loader).next()
    dim = [0, 2, 3]
    mean = data[0].mean(dim=dim).numpy()
    std = data[0].std(dim=dim).numpy()
    # end of creating a transform to do pre-processing

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean, std
            ),
        ])

    train_set.transform = transform

    if validation_ratio > 0.:
        val_set.transform = transform
    else:
        val_set = None

    test_set = CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    return train_set, val_set, test_set


def get_shape_for_contrastive_learning(mini_batch_size: int, block_size: int, neg_size: int, dim_h: int) -> tuple:
    """
    Return mini-batch shapes for contrastive algorithms. It is especially useful when you use batch norm.

    :param mini_batch_size: The size of mini-batch.
    :param block_size: The size of block.
    :param neg_size: The number of negative samples.
    :param dim_h: The dimensionality of representation.

    :return: Four tuples.
    """

    batch2input_shape_pos = [mini_batch_size * block_size, 3, 32, 32]
    batch2input_shape_neg = [mini_batch_size * neg_size * block_size, 3, 32, 32]
    output2emb_shape_pos = [mini_batch_size, block_size, dim_h]
    output2emb_shape_neg = [mini_batch_size, neg_size, block_size, dim_h]

    return batch2input_shape_pos, batch2input_shape_neg, output2emb_shape_pos, output2emb_shape_neg
