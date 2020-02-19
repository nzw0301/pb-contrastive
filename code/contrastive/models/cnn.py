import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm


class CNN(nn.Module):
    def __init__(self, rnd: np.random.RandomState, num_last_units=100, init_weights=True, supervised=False):
        """
        :param rnd: `np.random.RandomState` instance.
        :param num_last_units: The dimensionality of representation and
            the number of supervised classes when `supervised=True`.
        :param init_weights: Boolean flag to initialize the weights.
        :param supervised: If this value is `True`, the model is used by the supervised way.
        """
        super(CNN, self).__init__()
        self.num_last_units = num_last_units

        self.features = self.create_features()
        self.f_last = nn.Linear(1600, num_last_units)

        if supervised:
            self.classifier = nn.Linear(self.num_last_units, self.num_last_units)

        if init_weights:
            self._initialize_weights(rnd, supervised)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """
        Feature extractor.

        :param inputs: CIFAR-100's input data.

        :return: Feature representation Shape is (mini_batch, num_last_units).
        """
        x = self.features(inputs)
        x = torch.flatten(x, 1)
        x = self.f_last(x)
        return x

    def g(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """
        Supervised feed-forwarding function.

        :param inputs: CIFAR-100's input data.

        :return: model's output. Shape is (mini_batch, num_last_units).
        """

        return self.classifier(self.forward(inputs))

    @staticmethod
    def create_features() -> torch.nn.modules.container.Sequential:
        """
        Initializes CNN components

        :return: PyTorch's sequential instance.
        """
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def _initialize_weights(self, rnd: np.random.RandomState, supervised=False) -> None:
        """
        Initialize the model's weights.

        Initialization is based on the following repo
        https://github.com/gkdziugaite/pacbayes-opt/blob/58beae1f63ce0efaf749757a7c98eac2c8414238/snn/core/cnn_fn.py#L101

        :param rnd: `np.random.RandomState` instance.
        :param supervised: Supervised mode flag.

        :return: None
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = torch.from_numpy(
                    truncnorm.rvs(
                        a=-2 * 5e-2, b=2 * 5e-2, size=tuple(m.weight.size()), random_state=rnd
                    ).astype(np.float32)
                )
                nn.init.constant_(m.bias, 0.)
            if isinstance(m, nn.Linear):
                m.weight.data = torch.from_numpy(
                    truncnorm.rvs(
                        a=-1. / 800., b=1. / 800., size=(self.num_last_units, 1600), random_state=rnd
                    ).astype(np.float32)
                )
                nn.init.constant_(m.bias, 0.)

        if supervised:
            self.classifier.weight.data = torch.from_numpy(
                truncnorm.rvs(
                    a=-1. / 50., b=1. / 50., size=(self.num_last_units, self.num_last_units), random_state=rnd
                ).astype(np.float32)
            )
            nn.init.constant_(self.classifier.bias, 0.)
