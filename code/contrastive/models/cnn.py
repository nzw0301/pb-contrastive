import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm


class CNN(nn.Module):
    def __init__(self, rnd: np.random.RandomState, num_last_units=100, init_weights=True, supervised=False):
        super(CNN, self).__init__()
        self.num_last_units = num_last_units
        self.features = self.create_features()
        self.f_last = nn.Linear(1600, num_last_units)

        if supervised:
            self.classifier = nn.Linear(num_last_units, num_last_units)

        if init_weights:
            self._initialize_weights(rnd, supervised)

    def forward(self, inputs):
        """
        Supervised feature extractor.
        :param inputs: CIFAR-100's input data
        :return: Feature representation Shape is (mini_batch, num_last_units)
        """
        x = self.features(inputs)
        x = torch.flatten(x, 1)
        x = self.f_last(x)
        return x

    def g(self, inputs):
        """
        Supervised feed-forwarding function.

        :param inputs: CIFAR-100's input data
        :return: model's output. Shape is (mini_batch, num_last_units)
        """

        return self.classifier(self.forward(inputs))

    @staticmethod
    def create_features():
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def _initialize_weights(self, rnd: np.random.RandomState, supervised=False):
        # Initialisation is based on the following repo
        # https://github.com/gkdziugaite/pacbayes-opt/blob/58beae1f63ce0efaf749757a7c98eac2c8414238/snn/core/cnn_fn.py#L101

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
