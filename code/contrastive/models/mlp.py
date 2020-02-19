import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm


class MLP(nn.Module):
    def __init__(
            self,
            rnd: np.random.RandomState,
            num_input_units=22,
            num_hidden_units=50,
            num_last_units=50,
            init_weights=True,
            supervised=False
    ):
        """
        :param rnd: `np.random.RandomState` instance.
        :param num_input_units: The number of input units.
        :param num_hidden_units: The number of hidden units.
        :param num_last_units: The dimensionality of representation / the number of last layer's units.
        :param init_weights: Initialization flag.
        :param supervised: Supervised flag.
        """

        super(MLP, self).__init__()
        self.num_input_units = num_input_units
        self.num_last_units = num_last_units
        self.num_hidden_units = num_hidden_units

        self.features = nn.Sequential(
            nn.Linear(self.num_input_units, self.num_hidden_units),
            nn.ReLU(inplace=True),
        )
        self.f_last = nn.Linear(self.num_hidden_units, num_last_units)

        if supervised:
            self.classifier = nn.Linear(num_last_units, 95)

        if init_weights:
            self._initialize_weights(rnd, supervised)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """
        Feature extractor.

        :param inputs: AUSLAN's input data.

        :return: Feature representation Shape is (mini_batch, num_last_units).
        """
        return self.f_last(self.features(inputs))

    def g(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """
        Supervised feed-forwarding function.

        :param inputs: AUSLAN's input data.

        :return: model's output. Shape is (mini_batch, num_last_units).
        """
        return self.classifier(self.forward(inputs))

    def _initialize_weights(self, rnd: np.random.RandomState, supervised=False) -> None:
        """
        Initialize the model's weights.

        Initialization is based on the following repo
        https://github.com/gkdziugaite/pacbayes-opt/blob/58beae1f63ce0efaf749757a7c98eac2c8414238/snn/core/cnn_fn.py#L101

        :param rnd: np.random.RandomState instance.
        :param supervised: Supervised mode flag.

        :return: None
        """

        self.features[0].weight.data = torch.from_numpy(
            truncnorm.rvs(
                a=-2. / self.num_input_units, b=2. / self.num_input_units,
                size=(self.num_hidden_units, self.num_input_units), random_state=rnd
            ).astype(np.float32)
        )
        nn.init.constant_(self.features[0].bias, 0.)

        self.f_last.weight.data = torch.from_numpy(
            truncnorm.rvs(
                a=-2. / self.num_hidden_units, b=2. / self.num_hidden_units,
                size=(self.num_last_units, self.num_hidden_units), random_state=rnd
            ).astype(np.float32)
        )
        nn.init.constant_(self.f_last.bias, 0.)

        if supervised:
            self.classifier.weight.data = torch.from_numpy(
                truncnorm.rvs(
                    a=-2. / self.num_last_units, b=2. / self.num_last_units,
                    size=(95, self.num_last_units), random_state=rnd
                ).astype(np.float32)
            )
            nn.init.constant_(self.classifier.bias, 0.)
