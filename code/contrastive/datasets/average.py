from torch.utils.data import Dataset
import torch


class AverageDataset(Dataset):
    """Average Dataset.
    """

    def __init__(
            self,
            fx_test,
            raw_test_targets,
            target_ids
    ):
        """
        This class does not use `transform` because samples of `cifar100/Australian`
        have already been preprocessed by calling `CIFAR100Dataset/AustralianDataset`.

        :param fx_test: Tensor contains feature representations on test data shape is (num-test data, dim-features)
        :param raw_test_targets: original CIFAR100/Australian's test label. shape is  (num-test data, )
        :param target_ids: CIFAR100/Australian's id list contains int numbers that are from 0 to 99.
        """

        label2average_id = {target_id: new_id for new_id, target_id in enumerate(target_ids)}
        self.data = []
        self.targets = []
        self.target_original_ids = target_ids

        for fx, label in zip(fx_test, raw_test_targets):
            if label in target_ids:
                new_label = label2average_id[label]
                self.targets.append(new_label)
                self.data.append(fx)

        self.data = torch.stack(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, positive-block, negative block)
        """
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)
