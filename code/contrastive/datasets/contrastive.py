import numpy as np
import torch
from torch.utils.data import DataLoader

from .auslan import get_train_val_test_datasets as auslan_get_train_val_test_datasets
from .cifar100 import get_train_val_test_datasets as cifar100_get_train_val_test_datasets
from .utils import get_class2samples


class ContrastiveDataset(torch.utils.data.Dataset):
    """Contrastive Dataset.
    """

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            num_classes: int,
            block_size=2,
            neg_size=4,
            num_blocks_per_class=500,
            iid=True
    ):
        """
        This class does not use `transform` because samples of `cifar100Dataset` and `AUSLAN` have already
        been preprocessed by calling `CIFAR100Dataset` or `AUSLAN` via train/val/test split function.

        :param dataset: Instance of torchvision.datasets.CIFAR100/AUSLAN.
        :param block_size: The size of blocks.
        :param neg_size: The size of negative samples per positive samples.
        :param num_blocks_per_class: The number of blocks per class. It is only used for iid case.
        :param iid: Boolean flag to create time-dependent data.
        """

        self.num_classes = num_classes

        self.data = []
        self.positives = []
        self.negatives = []

        if iid:
            self._create_iid_data(dataset, num_classes, num_blocks_per_class, block_size, neg_size)
            self.data = torch.cat(self.data, dim=0)
        else:
            assert len(dataset.data.shape) == 3
            self._create_non_iid_data(dataset, num_classes, block_size, neg_size)
            self.data = torch.stack(self.data)

        self.positives = torch.stack(self.positives)
        self.negatives = torch.stack(self.negatives)

    def _create_iid_data(self, dataset, num_classes, num_blocks_per_class, block_size, neg_size):
        class_id2samples = get_class2samples(dataset, num_classes=num_classes)
        for class_id in range(self.num_classes):
            num_samples_in_target_class = len(class_id2samples[class_id])
            for _ in range(num_blocks_per_class):
                self.data.append(
                    class_id2samples[class_id][torch.randint(low=0, high=num_samples_in_target_class, size=(1,))])

                self.positives.append(
                    class_id2samples[class_id][torch.randint(
                        low=0, high=num_samples_in_target_class, size=(block_size,)
                    )])

                negative_block_samples = []
                for _ in range(neg_size):
                    noise_class_id = torch.randint(low=0, high=self.num_classes, size=(1,))
                    negative_block_samples.append(
                        class_id2samples[noise_class_id][
                            torch.randint(low=0, high=num_samples_in_target_class, size=(block_size,))
                        ]
                    )
                self.negatives.append(torch.stack(negative_block_samples))

    def _create_non_iid_data(self, dataset, num_classes, block_size, neg_size):
        # create tensor to sample negative sample
        class_id2samples = get_class2samples(dataset, num_classes=num_classes)
        # convert 3d to 2d.
        for class_id, tensor_3d in enumerate(class_id2samples):
            shape = tensor_3d.shape
            class_id2samples[class_id] = tensor_3d.view(shape[0] * shape[1], shape[2])

        for sequence, _ in dataset:
            for t, x in enumerate(sequence[:-block_size]):
                self.data.append(x)

                self.positives.append(sequence[t + 1:(t + 1 + block_size)])

                negative_block_samples = []
                for _ in range(neg_size):
                    noise_class_id = torch.randint(low=0, high=self.num_classes, size=(1,))
                    num_samples_in_target_class = len(class_id2samples[noise_class_id])
                    negative_block_samples.append(
                        class_id2samples[noise_class_id][
                            torch.randint(low=0, high=num_samples_in_target_class, size=(block_size,))
                        ]
                    )
                self.negatives.append(torch.stack(negative_block_samples))

    def __getitem__(self, index) -> tuple:
        return self.data[index], self.positives[index], self.negatives[index]

    def __len__(self) -> int:
        return len(self.data)


def get_contrastive_data_loaders(
        rnd: np.random.RandomState,
        data_name='cifar',
        validation_ratio=0.05,
        mini_batch_size=100,
        num_blocks_per_class=500,
        block_size=10,
        neg_size=2,
        num_workers=2,
        root='~/data',
        include_test=False,
        iid=True
) -> tuple:
    """
    :param rnd: `np.random.RandomState` instance.
    :param data_name: Data name. valid value is either `cifar` or `auslan`.
    :param validation_ratio: If this value is `0.`, `val_set` is `None`.
    :param mini_batch_size: The size of mini-batch.
    :param num_blocks_per_class: The number of block size.
    :param block_size: The size of block.
    :param neg_size: The number of negative samples.
    :param num_workers: The number of workers for data loader.
    :param root: Data path to store data.
    :param include_test: Bool flag whether return test data or not. Default: False.
    :param iid: Bool flag whether creating contrastive data by iid way or non-iid (time-dependent way). Default: True.
    
    :return: Tuple of data loaders.
    """
    data_name = data_name.lower()
    assert data_name in ['cifar', 'auslan']

    if data_name == 'cifar':
        assert iid
        num_classes = 100
        train_set, val_set, test_set = cifar100_get_train_val_test_datasets(
            rnd=rnd, validation_ratio=validation_ratio, root=root
        )
    else:
        num_classes = 95
        train_set, val_set, test_set = auslan_get_train_val_test_datasets(
            rnd=rnd, validation_ratio=validation_ratio, root=root, squash_time=iid
        )

    if val_set is None:
        train_set = ContrastiveDataset(
            train_set,
            num_classes=num_classes,
            block_size=block_size,
            neg_size=neg_size,
            num_blocks_per_class=num_blocks_per_class,
            iid=iid
        )
        val_loader = None
    else:
        train_set = ContrastiveDataset(
            train_set,
            num_classes=num_classes,
            block_size=block_size,
            neg_size=neg_size,
            num_blocks_per_class=int(num_blocks_per_class * (1. - validation_ratio)),
            iid=iid
        )

        val_set = ContrastiveDataset(
            val_set,
            num_classes=num_classes,
            block_size=block_size,
            neg_size=neg_size,
            num_blocks_per_class=int(num_blocks_per_class * validation_ratio),
            iid=iid
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

    if not include_test:
        return train_loader, val_loader
    else:
        test_set = ContrastiveDataset(
            test_set,
            num_classes=num_classes,
            block_size=block_size,
            neg_size=neg_size,
            num_blocks_per_class=len(test_set) // num_classes,
            iid=iid
        )

        test_loader = DataLoader(
            test_set,
            batch_size=mini_batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        return train_loader, val_loader, test_loader
