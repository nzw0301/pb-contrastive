import copy

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.datasets import VisionDataset
from .utils import get_label2samples


class ContrastiveCIFAR100(VisionDataset):
    """`Contrastive CIFAR100 Dataset.
    """
    num_classes = 100

    def __init__(
            self,
            cifar100: CIFAR100,
            block_size=2,
            neg_size=4,
            num_blocks_per_class=500
    ):
        """
        This class does not use `transform` because samples of `cifar100Dataset` have already been preprocessed
        by calling `CIFAR100Dataset`.

        :param cifar100: Instance of torchvision.datasets.CIFAR100
        :param block_size: The size of blocks in Arora et al. 2019
        :param neg_size: The size of negative samples per positive samples in Arora et al. 2019
        :param num_blocks_per_class: the number of blocks per class
        """

        super(ContrastiveCIFAR100, self).__init__(root='')  # dummy

        class_id2images = get_label2samples(cifar100, num_classes=100)

        self.images = []
        self.positives = []
        self.negatives = []
        for class_id in range(self.num_classes):
            num_images_in_target_class = len(class_id2images[class_id])
            for _ in range(num_blocks_per_class):
                self.images.append(
                    class_id2images[class_id][torch.randint(low=0, high=num_images_in_target_class, size=(1,))])

                self.positives.append(
                    class_id2images[class_id][torch.randint(
                        low=0, high=num_images_in_target_class, size=(block_size,)
                    )])

                negative_block_samples = []
                for _ in range(neg_size):
                    noise_class_id = torch.randint(low=0, high=self.num_classes, size=(1,))
                    negative_block_samples.append(
                        class_id2images[noise_class_id][
                            torch.randint(low=0, high=num_images_in_target_class, size=(block_size,))
                        ]
                    )
                self.negatives.append(torch.stack(negative_block_samples))

        self.images = torch.cat(self.images, dim=0)
        self.positives = torch.stack(self.positives)
        self.negatives = torch.stack(self.negatives)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, positive-block, negative block)
        """
        return self.images[index], self.positives[index], self.negatives[index]

    def __len__(self):
        return len(self.images)


def get_train_val_test_datasets(
        rnd=np.random.RandomState(7), validation_ratio=0.02,
        root='~/data', supervised=False
):
    """
    A function to create CIFAR-100 train/val/test data loader

    :param rnd:
    :param validation_ratio: the ratio of validation data
    :param root: path to save data.
    :param supervised: Boolean flag. If supervised is True, training transform contains data-augmentation.
    :return:
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
        x_train, x_val, y_train, y_val = \
            train_test_split(train_set.data, train_set.targets, test_size=validation_ratio, random_state=rnd,
                             stratify=train_set.targets)

        val_set = copy.deepcopy(train_set)

        train_set.data = x_train
        train_set.targets = y_train
        val_set.data = x_val
        val_set.targets = y_val

    # create a transform to do pre-processing
    train_loader = DataLoader(
        train_set,
        batch_size=len(train_set),
        shuffle=False,
    )

    data = iter(train_loader).next()
    mean = data[0].mean(dim=[0, 2, 3]).numpy()
    std = data[0].std(dim=[0, 2, 3]).numpy()
    # end of creating a transform to do pre-processing

    if supervised:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean, std
                ),
            ])
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean, std
                ),
            ])

    train_set.transform = train_transform

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean, std
            ),
        ])

    if validation_ratio > 0.:
        val_set.transform = val_transform
    else:
        val_set = None

    test_set = CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=val_transform
    )

    return train_set, val_set, test_set


def get_contrastive_cifar100_data_loaders(
        rnd: np.random.RandomState(7),
        validation_ratio=0.02, mini_batch_size=100, num_blocks_per_class=500, block_size=10, neg_size=2, num_workers=2,
        root='~/data', include_test=False
):
    """
    :param rnd:
    :param validation_ratio: if this value is `0.`, `val_set` is `None`.
    :param mini_batch_size:
    :param num_blocks_per_class:
    :param block_size:
    :param neg_size:
    :param num_workers:
    :param root:
    :return:
    """

    train_set, val_set, test_set = get_train_val_test_datasets(rnd, validation_ratio, root=root)

    if val_set is None:
        train_set = ContrastiveCIFAR100(
            train_set,
            block_size=block_size,
            neg_size=neg_size,
            num_blocks_per_class=num_blocks_per_class
        )
        val_loader = None
    else:
        train_set = ContrastiveCIFAR100(
            train_set,
            block_size=block_size,
            neg_size=neg_size,
            num_blocks_per_class=int(num_blocks_per_class * (1. - validation_ratio))
        )

        val_set = ContrastiveCIFAR100(
            val_set,
            block_size=block_size,
            neg_size=neg_size,
            num_blocks_per_class=int(num_blocks_per_class * validation_ratio)
        )

        val_loader = DataLoader(
            val_set,
            batch_size=mini_batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    train_loader = DataLoader(
        train_set,
        batch_size=mini_batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    if include_test:
        test_set = ContrastiveCIFAR100(
            test_set,
            block_size=block_size,
            neg_size=neg_size,
            num_blocks_per_class=100
        )

        test_loader = DataLoader(
            test_set,
            batch_size=mini_batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def get_shape_for_contrastive_learning(mini_batch_size, block_size, neg_size, dim_h):
    batch2input_shape_pos = [mini_batch_size * block_size, 3, 32, 32]
    batch2input_shape_neg = [mini_batch_size * neg_size * block_size, 3, 32, 32]
    output2emb_shape_pos = [mini_batch_size, block_size, dim_h]
    output2emb_shape_neg = [mini_batch_size, neg_size, block_size, dim_h]

    return batch2input_shape_pos, batch2input_shape_neg, output2emb_shape_pos, output2emb_shape_neg
