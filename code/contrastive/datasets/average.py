import torch
from torch.utils.data import Dataset


class AverageDataset(Dataset):
    """Average Dataset.
    """

    def __init__(
            self,
            fx_test: torch.FloatTensor,
            raw_test_targets,
            target_ids: tuple
    ):
        """
        This class does not use `transform` because samples of `cifar100/Auslan`
        have already been preprocessed by calling `CIFAR100/Auslan`.

        :param fx_test: Tensor contains feature representations on test data shape is (num_test_data, dim-features)
        :param raw_test_targets: original CIFAR100/Auslan's test label. shape is  (num_test_data, )
        :param target_ids: CIFAR100/Auslan's id tuple contains int numbers that are from 0 to 99 for CIFAR100
            and from 0 to 94 for AUSLAN.
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

    def __getitem__(self, index) -> tuple:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)
