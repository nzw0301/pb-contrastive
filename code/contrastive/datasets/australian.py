import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_and_extract_archive

from .utils import get_label2samples


class Australian(torch.utils.data.Dataset):
    num_classes = 95
    base_folder = 'tctodd'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auslan2-mld/tctodd.tar.gz'
    filename = 'tctodd.tar.gz'
    classname2label = {
        'God': 0, 'I': 1, 'Norway': 2, 'alive': 3, 'all': 4, 'answer': 5, 'boy': 6, 'building': 7,
        'buy': 8, 'change_mind_': 9, 'cold': 10, 'come': 11, 'computer_PC_': 12, 'cost': 13, 'crazy': 14,
        'danger': 15, 'deaf': 16, 'different': 17, 'draw': 18, 'drink': 19, 'eat': 20, 'exit': 21,
        'flash': 22, 'forget': 23, 'girl': 24, 'give': 25, 'glove': 26, 'go': 27, 'happy': 28,
        'head': 29, 'hear': 30, 'hello': 31, 'hot': 32, 'how': 33, 'hurry': 34, 'hurt': 35,
        'innocent': 36, 'is_true_': 37, 'joke': 38, 'juice': 39, 'know': 40, 'later': 41, 'lose': 42,
        'love': 43, 'make': 44, 'man': 45, 'maybe': 46, 'mine': 47, 'money': 48, 'more': 49, 'name': 50,
        'no': 51, 'not': 52, 'paper': 53, 'pen': 54, 'please': 55, 'polite': 56, 'question': 57,
        'read': 58, 'ready': 59, 'research': 60, 'responsible': 61, 'right': 62, 'sad': 63, 'same': 64,
        'science': 65, 'share': 66, 'shop': 67, 'soon': 68, 'sorry': 69, 'spend': 70, 'stubborn': 71,
        'surprise': 72, 'take': 73, 'temper': 74, 'thank': 75, 'think': 76, 'tray': 77, 'us': 78,
        'voluntary': 79, 'wait_notyet_': 80, 'what': 81, 'when': 82, 'where': 83, 'which': 84, 'who': 85,
        'why': 86, 'wild': 87, 'will': 88, 'write': 89, 'wrong': 90, 'yes': 91, 'you': 92, 'zero': 93,
        'his_hers': 94
    }

    train_dir_ids = tuple(range(1, 9))
    test_dir_ids = (9,)
    max_length = 45

    def __init__(
            self, root='/tmp/tctodd', train=True, download=True, train_dir_ids=None, test_dir_ids=None,
            to_tensor=False
    ):
        """
        :param root:
        :param train:
        :param download:
        :param train_dir_ids:
        :param test_dir_ids:
        :param to_tensor: If True, training sample shape will be (len(train_dir_ids) * 95, 45, 22).
            False: training sample shape will be (len(train_dir_ids) * 95 * 45, 22).
        """
        self.train = train
        self.to_tensor = to_tensor

        if train_dir_ids is not None and train:
            self.train_dir_ids = train_dir_ids
        if test_dir_ids is not None and not train:
            self.test_dir_ids = test_dir_ids

        if download:
            download_and_extract_archive(self.url, download_root=root, extract_root=None, filename=self.filename)

        if train:
            target_dir_ids = self.train_dir_ids
        else:
            target_dir_ids = self.test_dir_ids

        x, y = [], []
        for i in target_dir_ids:
            for fname in sorted(glob.glob('{}/tctodd{}/*'.format(root, i))):
                x_in_file = []
                with open(fname) as f:
                    class_name = fname.split('/')[-1].split('-')[0]

                    # typo?
                    # `her` only exists in `tctodd1`, and `his_hers` only does not exists in `tctodd1`.
                    if '/her-' in fname:
                        class_name = 'his_hers'

                    label = self.classname2label[class_name]
                    for t, l in enumerate(f, start=1):
                        vec = np.array(list(map(float, l.split())))
                        x_in_file.append(vec)
                        if t == self.max_length:
                            break
                if to_tensor:
                    x.append(x_in_file)
                    y.append(label)
                else:
                    x += x_in_file
                    y += [label] * self.max_length

        self.data, self.targets = np.array(x), np.array(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.from_numpy(self.data[index]).float()
        target = self.targets[index]
        return sample, target


class ContrastiveAustralianDataset(torch.utils.data.Dataset):
    """
    Contrastive Australian Dataset.
    """
    def __init__(
            self,
            australian_dataset: Australian,
            block_size=1,
            neg_size=4
):
        """
        This class does not use `transform` because samples of `cifar100Dataset` have already been preprocessed
        by calling `CIFAR100Dataset`.

        :param australian_dataset: Instance of torchvision.datasets.CIFAR100
        :param block_size: The size of blocks in Arora et al. 2019
        :param neg_size: The size of negative samples per positive samples in Arora et al. 2019
        """
        assert australian_dataset.to_tensor

        self.samples = []
        self.positives = []
        self.negatives = []
        class_id2samples = get_label2samples(australian_dataset, num_classes=australian_dataset.num_classes)
        T = australian_dataset.max_length
        num_classes = australian_dataset.num_classes
        for time_steps, _ in australian_dataset:
            for t, time_step_sample in enumerate(time_steps[:(T-block_size)]):
                self.samples.append(time_step_sample)
                self.positives.append(time_steps[(t+1):(t+1+block_size)])

                negative_block_samples = []
                for _ in range(neg_size):
                    noise_class_id = torch.randint(low=0, high=num_classes, size=(1,))
                    s = torch.randint(low=0, high=len(class_id2samples[noise_class_id]), size=(1,))
                    t = torch.randint(low=0, high=T-block_size, size=(1, ))
                    negative_block_samples.append(
                        class_id2samples[noise_class_id][s, t:(t+block_size)].view(block_size, 22)
                    )

                self.negatives.append(torch.stack(negative_block_samples))

        self.samples = torch.stack(self.samples, dim=0)
        self.positives = torch.stack(self.positives)
        self.negatives = torch.stack(self.negatives)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, positive-block, negative block)
        """
        return self.samples[index], self.positives[index], self.negatives[index]

    def __len__(self):
        return self.samples.shape[0]


def get_train_val_test_datasets(
        root='~/data/tctodd',
        train_ids=None,
        validation_ids=None,
        test_ids=None,
        to_tensor=False
):
    if train_ids == validation_ids == test_ids is None:
        train_ids = tuple(range(1, 8))
        validation_ids = (8,)
        test_ids = (9,)

    train_set = Australian(
        root=root,
        train=True,
        download=True,
        train_dir_ids=train_ids,
        to_tensor=to_tensor
    )

    # create validation split
    if validation_ids is not None:
        val_set = Australian(
            root=root,
            train=False,
            download=True,
            test_dir_ids=validation_ids,
            to_tensor=to_tensor
        )

    # create a transform to do pre-processing
    train_loader = DataLoader(
        train_set,
        batch_size=len(train_set),
        shuffle=False,
    )

    data = iter(train_loader).next()
    if to_tensor:
        dim = [0, 1]
    else:
        dim = 0
    mean = data[0].mean(dim=dim).numpy()
    std = data[0].std(dim=dim).numpy()
    # end of creating a transform to do pre-processing
    train_set.data = (train_set.data - mean) / std

    if validation_ids is not None:
        val_set.data = (val_set.data - mean) / std
    else:
        val_set = None

    test_set = Australian(
        root=root,
        train=False,
        download=False,
        test_dir_ids=test_ids,
        to_tensor=to_tensor
    )
    test_set.data = (test_set.data - mean) / std

    return train_set, val_set, test_set


def get_contrastive_australian_data_loaders(
        train_ids, validation_ids, test_ids,
        mini_batch_size=100, block_size=10, neg_size=2, num_workers=2,
        root='~/data', include_test=False,
):

    train_set, val_set, test_set = get_train_val_test_datasets(
        root, train_ids, validation_ids, test_ids, to_tensor=True
    )

    train_set = ContrastiveAustralianDataset(
        train_set,
        block_size=block_size,
        neg_size=neg_size,
    )
    val_loader = None
    if val_set is not None:
        val_set = ContrastiveAustralianDataset(
            val_set,
            block_size=block_size,
            neg_size=neg_size,
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
        test_set = ContrastiveAustralianDataset(
            test_set,
            block_size=block_size,
            neg_size=neg_size
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
    batch2input_shape_pos = [mini_batch_size * block_size, 22]
    batch2input_shape_neg = [mini_batch_size * neg_size * block_size, 22]
    output2emb_shape_pos = [mini_batch_size, block_size, dim_h]
    output2emb_shape_neg = [mini_batch_size, neg_size, block_size, dim_h]

    return batch2input_shape_pos, batch2input_shape_neg, output2emb_shape_pos, output2emb_shape_neg
