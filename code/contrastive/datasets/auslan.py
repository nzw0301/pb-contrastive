import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_and_extract_archive

from .utils import _train_val_split


class AUSLAN(torch.utils.data.Dataset):
    num_classes = 95
    num_dir = 9
    max_length = 45
    base_folder = 'tctodd'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auslan2-mld/tctodd.tar.gz'
    filename = 'tctodd.tar.gz'

    classname2class_id = {
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

    def __len__(self) -> int:
        return len(self.data)

    def __init__(
            self,
            root='/tmp/tctodd',
            train=True,
            download=True,
            train_ids=None,
            test_ids=None,
    ):
        """
        :param root: absolute root path to store data.
        :param train: Boolean flag. If true, return train data, and else, return test data.
        :param download: Download's boolean flag.
        :param train_ids: tuple of int ids. Valid range is 1 -- 9.
        :param test_ids: tuple of int ids. Valid range is the same to the `train_ids`.
        """

        self.train = train

        if download:
            download_and_extract_archive(self.url, download_root=root, extract_root=None, filename=self.filename)

        if train:
            target_dir_ids = train_ids
        else:
            target_dir_ids = test_ids

        # x's shape: (num_samples, self._max_length, num_features), and y's one: (num_samples).
        x, y = [], []
        for i in target_dir_ids:
            for fname in sorted(glob.glob('{}/tctodd{}/*'.format(root, i))):
                x_in_file = []
                with open(fname) as f:
                    class_name = fname.split('/')[-1].split('-')[0]

                    # Merge `her` into `his_hers`
                    # it may be a typo.
                    # because `her` only exists in `tctodd1`, and `his_hers` does not exists in `tctodd1`.
                    if '/her-' in fname:
                        class_name = 'his_hers'

                    label = self.classname2class_id[class_name]
                    for line in f:
                        vec = np.array(list(map(float, line.split())))
                        x_in_file.append(vec)

                x.append(x_in_file[:self.max_length])
                y.append(label)

        self.data, self.targets = np.array(x), np.array(y)

    def __getitem__(self, index) -> tuple:
        sample = torch.from_numpy(self.data[index]).float()
        target = self.targets[index]
        return sample, target


def _squash_2nd_dim(dataset):
    shape = dataset.data.shape
    dataset.data = dataset.data.reshape(shape[0] * shape[1], shape[2])
    dataset.targets = np.repeat(dataset.targets, shape[1])
    return dataset


def get_train_val_test_datasets(
        rnd: np.random.RandomState,
        root='~/data/tctodd',
        validation_ratio=0.05,
        train_ids=None,
        test_ids=None,
        squash_time=False
) -> tuple:
    """
    fetch train/val/test data sets.

    :param rnd: `np.random.RandomState` instance.
    :param root: Root path to store data. Path must be absolute path.
    :param validation_ratio: The ratio of training data to create validation data. .
    :param train_ids: tuple of int ids. Valid range is 1 -- 9.
    :param test_ids: tuple of int ids. Valid range is the same to the `train_ids`.
    :param squash_time: boolean. If true, return each dataset's shape is (num_samples x 45, 22).
    if not, its shape is (num_samples, 45, 22)

    :return: Tuple of data sets.
    """
    if train_ids == test_ids is None:
        train_ids = tuple(range(1, 9))
        test_ids = (9,)

    if test_ids is None:
        test_ids = []

    assert len(set(test_ids) & set(train_ids)) == 0

    train_set = AUSLAN(
        root=root,
        train=True,
        download=True,
        train_ids=train_ids,
    )

    test_set = AUSLAN(
        root=root,
        train=False,
        download=False,
        test_ids=test_ids,
    )

    # create validation split
    if validation_ratio > 0.:
        train_set, val_set = _train_val_split(rnd=rnd, train_dataset=train_set, validation_ratio=validation_ratio)

    # create normalization parameters
    train_loader = DataLoader(
        train_set,
        batch_size=len(train_set),
        shuffle=False
    )

    data = iter(train_loader).next()

    dim = (0, 1)
    mean = data[0].mean(dim=dim).numpy()
    std = data[0].std(dim=dim).numpy()
    # end of making normalization parameters

    # apply normalization
    train_set.data = (train_set.data - mean) / std

    if validation_ratio > 0.:
        val_set.data = (val_set.data - mean) / std
    else:
        val_set = None

    test_set.data = (test_set.data - mean) / std

    if squash_time:
        train_set = _squash_2nd_dim(train_set)
        test_set = _squash_2nd_dim(test_set)
        if validation_ratio > 0.:
            val_set = _squash_2nd_dim(val_set)

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

    batch2input_shape_pos = [mini_batch_size * block_size, 22]
    batch2input_shape_neg = [mini_batch_size * neg_size * block_size, 22]
    output2emb_shape_pos = [mini_batch_size, block_size, dim_h]
    output2emb_shape_neg = [mini_batch_size, neg_size, block_size, dim_h]

    return batch2input_shape_pos, batch2input_shape_neg, output2emb_shape_pos, output2emb_shape_neg
