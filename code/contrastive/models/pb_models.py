import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import pi
from scipy.stats import truncnorm
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class StochasticConv2D(torch.nn.modules.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        """
        StochasticConv2D class.
        Parameters are same to PyTorch's Conv2d.
        """

        super(StochasticConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups,
            bias, padding_mode
        )

        self.weight_log_std = Parameter(torch.Tensor(self.weight.size()))
        self.bias_log_std = Parameter(torch.Tensor(self.bias.size()))
        self.weight_prior = Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
        self.bias_prior = Parameter(torch.Tensor(self.bias.size()), requires_grad=False)

        self.weight_noise = Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
        self.bias_noise = Parameter(torch.Tensor(self.bias.size()), requires_grad=False)

    def sample_noise(self):
        """
        Sample weights from posterior.
        :return: None
        """
        self.realised_weight = self.weight + self.weight_noise.normal_() * torch.exp(self.weight_log_std)
        self.realised_bias = self.bias + self.bias_noise.normal_() * torch.exp(self.bias_log_std)

    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            self.realised_weight, self.realised_bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, self.realised_weight, self.realised_bias, self.stride,
                        self.padding, self.dilation, self.groups)


class StochasticLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features):
        """
        StochasticLinear class.
        Parameters are same to PyTorch's Linear class.
        """
        super(StochasticLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_std = Parameter(torch.Tensor(out_features, in_features))
        self.weight_prior = Parameter(torch.Tensor(self.weight.size()), requires_grad=False)

        self.bias = Parameter(torch.Tensor(out_features))
        self.bias_log_std = Parameter(torch.Tensor(out_features))
        self.bias_prior = Parameter(torch.Tensor(self.bias.size()), requires_grad=False)

        self.weight_noise = Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
        self.bias_noise = Parameter(torch.Tensor(self.bias.size()), requires_grad=False)

    def forward(self, input):
        return F.linear(input, self.realised_weight, self.realised_bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def sample_noise(self):
        """
        Sample weights and bias from posterior.
        :return: None
        """
        self.realised_weight = self.weight + self.weight_noise.normal_() * torch.exp(self.weight_log_std)
        self.realised_bias = self.bias + self.bias_noise.normal_() * torch.exp(self.bias_log_std)


class StochasticCNN(nn.Module):
    def __init__(
            self, num_training_samples,
            rnd: np.random.RandomState, num_last_units=100,
            trained_deterministic_model=None, prior_log_std=-3., catoni_lambda=1.,
            delta=0.05, b=100, c=0.1, init_weights=True
    ):
        """
        :param num_training_samples: the number of training data
        :param rnd: numpy.random.RandomState instance for reproducesability
        :param num_last_units: The size of unis of the last linear layer
        :param trained_deterministic_model: The
        :param prior_log_std: initial value of prior's log std value.
        :param catoni_lambda: Catoni's Lambda Parameter. It must be positive.
        :param delta: Confidence parameter
        :param b: Prior's prevision paramettetr
        :param c: Prior's upper bound
        :param init_weights: If true, weights are initialized by truncated Gaussian. Note this value must be called to
            calculate KL divergence.
        """
        upper_log_std = 0.5 * np.log(np.float32(c))
        assert upper_log_std > prior_log_std, 'c is the upper bound of the prior\'s variance.'

        super(StochasticCNN, self).__init__()

        self.features = self.create_features()
        self.f_last = StochasticLinear(1600, num_last_units)

        self.num_weights = 0
        self.prior_log_std = Parameter(torch.Tensor([prior_log_std]))
        self.previous_prior_log_std = torch.Tensor([prior_log_std])  # for constraints

        if init_weights:
            if trained_deterministic_model is not None:
                self._initialize_weights_from_deterministic_model(trained_deterministic_model)
            else:
                self._initialize_weights(rnd)

        self.num_training_samples = Parameter(torch.Tensor([num_training_samples]), requires_grad=False)

        self.delta = Parameter(torch.Tensor([delta]), requires_grad=False)
        self.b = Parameter(torch.Tensor([b]), requires_grad=False)
        self.c = Parameter(torch.Tensor([c]), requires_grad=False)
        self.catoni_lambda = Parameter(torch.Tensor([catoni_lambda]), requires_grad=False)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.f_last(x)
        return x

    def sample_noise(self):
        """
        Sample weights and biases from the posterior.
        :return: None
        """
        for m in self.modules():
            if isinstance(m, (StochasticLinear, StochasticConv2D)):
                m.sample_noise()

    @staticmethod
    def create_features():
        return nn.Sequential(
            StochasticConv2D(3, 64, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            StochasticConv2D(64, 64, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def _initialize_weights_from_deterministic_model(self, deterministic_model):
        for m, premodel_module in zip(self.modules(), deterministic_model.modules()):
            if isinstance(m, (StochasticLinear, StochasticConv2D)):
                m.weight.data.copy_(premodel_module.weight.data)
                m.weight_prior.data.copy_(premodel_module.weight.data)

                m.bias.data.copy_(premodel_module.bias.data)
                m.bias_prior.data.copy_(premodel_module.bias.data)

                nn.init.constant_(m.weight_log_std, self.prior_log_std.item())
                nn.init.constant_(m.bias_log_std, self.prior_log_std.item())

                self.num_weights += np.prod(m.weight.size())
                self.num_weights += np.prod(m.bias.size())

    def _initialize_weights(self, rnd=np.random.RandomState(7)):
        conv_upper = 2 * 5e-2
        for m in self.modules():
            if isinstance(m, StochasticLinear):
                m.weight_prior.data = torch.from_numpy(
                    truncnorm.rvs(
                        a=-1. / 800., b=1. / 800., size=(100, 1600), random_state=rnd
                    ).astype(np.float32)
                )

            elif isinstance(m, StochasticConv2D):
                m.weight_prior.data = torch.from_numpy(
                    truncnorm.rvs(
                        a=-conv_upper, b=conv_upper, size=tuple(m.weight.size()), random_state=rnd
                    ).astype(np.float32)
                )

            if isinstance(m, (StochasticLinear, StochasticConv2D)):
                m.weight.data.copy_(m.weight_prior.data)

                nn.init.constant_(m.bias_prior, 0.)
                nn.init.constant_(m.bias, 0.)

                nn.init.constant_(m.weight_log_std, self.prior_log_std.item())
                nn.init.constant_(m.bias_log_std, self.prior_log_std.item())

                self.num_weights += np.prod(m.weight.size())
                self.num_weights += np.prod(m.bias.size())

    def kl(self, prior_log_std):
        num_weights = self.num_weights
        assert num_weights > 0
        mean_norm_list = []
        log_std_sum_list = []
        variance_l1_norm_list = []
        prior_variance = torch.exp(2. * prior_log_std)
        prior_log_variance = 2. * prior_log_std

        for m in self.modules():
            if isinstance(m, (StochasticLinear, StochasticConv2D)):
                mean_norm_list.append(torch.sum((m.weight - m.weight_prior) ** 2))
                mean_norm_list.append(torch.sum((m.bias - m.bias_prior) ** 2))

                log_std_sum_list.append(torch.sum(m.weight_log_std))
                log_std_sum_list.append(torch.sum(m.bias_log_std))

                # negative `prior` term provides more accurate a part of term in KL than
                # `torch.exp(2. * m.bias_log_std)` then is divided by `prior_variance`
                variance_l1_norm_list.append(torch.sum(torch.exp(2. * m.weight_log_std - prior_log_variance)))
                variance_l1_norm_list.append(torch.sum(torch.exp(2. * m.bias_log_std - prior_log_variance)))

        norm_weights = torch.sum(torch.stack(mean_norm_list))
        mean_part = norm_weights / prior_variance

        norm_log_std = 2. * torch.sum(torch.stack(log_std_sum_list))
        sum_variance_l1_norm = torch.sum(torch.stack(variance_l1_norm_list))

        std_part = sum_variance_l1_norm - norm_log_std + 2. * num_weights * prior_log_std

        kl = 0.5 * (mean_part + std_part - num_weights)
        return kl

    def union_bound(self, prior_log_std):
        """
        Calculate union bound value related to prior.
        :param prior_log_std: Float paramter contains prior's log std.
        :return: FloatTensor
        """
        return 2. * torch.log(self.b) + 2. * torch.log(torch.log(self.c) - 2. * prior_log_std) \
               + torch.log(pi ** 2 / (6. * self.delta))

    def pac_bayes_objective(self, contrastive_loss):
        """
        Catoni's PAC-Bayes bound with union bound

        :param contrastive_loss: empirical risk: FloatTensor
        :return: PAC-Bayes upper bound, KL, and complexity term; FloatTensors
        """
        kl = self.kl(prior_log_std=self.prior_log_std)

        # union bound term
        union_bound = self.union_bound(prior_log_std=self.prior_log_std)

        # KL term easily becomes large, so it is divided by Catoni's lambda
        objective = contrastive_loss + (kl + union_bound) / self.catoni_lambda

        return objective, kl, union_bound

    def compute_complexity_terms_with_discretized_prior_variance(self):
        """
        Compute kl divergence and union bound terms by using discretized_prior_variance.
        :return: tuple of kl divergence and union bound term
        """
        # discretize prior's variance parameter
        # https://github.com/gkdziugaite/pacbayes-opt/blob/master/snn/core/network.py#L398
        discretized_j = (self.b * (torch.log(self.c) - 2. * self.prior_log_std))

        discretized_j_up = torch.ceil(discretized_j)
        discretized_j_down = torch.floor(discretized_j)

        constant_in_log_delta = torch.log(np.pi ** 2 / (6 * self.delta))
        union_up = (constant_in_log_delta + 2 * torch.log(discretized_j_up)).item()
        union_down = (constant_in_log_delta + 2 * torch.log(discretized_j_down)).item()

        prior_log_std_up = (torch.log(self.c) - discretized_j_up / self.b) / 2.
        prior_log_std_down = (torch.log(self.c) - discretized_j_down / self.b) / 2.

        kl_up = self.kl(prior_log_std_up).item()
        kl_down = self.kl(prior_log_std_down).item()
        up_complexity = kl_up + union_up
        down_complexity = kl_down + union_down

        if up_complexity < down_complexity or np.isinf(down_complexity):
            return kl_up, union_up
        else:
            return kl_down, union_down

    def deterministic(self):
        """
        Only Use mean values for feed forwarding
        :return: None
        """
        for m in self.modules():
            if isinstance(m, (StochasticLinear, StochasticConv2D)):
                m.realised_weight = m.weight.detach()
                m.realised_bias = m.bias.detach()

    def constraints(self):
        if (torch.log(self.c) - 2. * self.prior_log_std).item() > 0.:
            self.previous_prior_log_std.data.copy_(self.prior_log_std.data)
        else:
            self.prior_log_std.data.copy_(self.previous_prior_log_std.data)


class StochasticMLP(StochasticCNN):
    def __init__(
            self, num_training_samples,
            rnd: np.random.RandomState, num_last_units=50,
            num_hidden_units=50,
            trained_deterministic_model=None, prior_log_std=-3., catoni_lambda=1.,
            delta=0.05, b=100, c=0.1, init_weights=True
    ):
        self.num_hidden_units = num_hidden_units
        self.num_last_units = num_last_units

        super(StochasticMLP, self).__init__(
            num_training_samples,
            rnd, num_last_units,
            trained_deterministic_model, prior_log_std, catoni_lambda,
            delta, b, c, init_weights=False
        )

        self.num_weights = 0
        self.features = nn.Sequential(
            StochasticLinear(22, num_hidden_units),
            nn.ReLU(inplace=True),
        )
        self.f_last = StochasticLinear(num_hidden_units, num_last_units)

        if init_weights:
            self._initialize_weights(rnd)

        self.num_training_samples = Parameter(torch.Tensor([num_training_samples]), requires_grad=False)

    def forward(self, inputs):
        x = self.features(inputs)
        x = self.f_last(x)
        return x

    def _initialize_weights(self, rnd=np.random.RandomState(7)):
        # Initialisation is based on the following repo
        # https://github.com/gkdziugaite/pacbayes-opt/blob/58beae1f63ce0efaf749757a7c98eac2c8414238/snn/core/cnn_fn.py#L101

        self.features[0].weight_prior.data = torch.from_numpy(
            truncnorm.rvs(
                a=-1. / 11., b=1. / 11., size=(self.num_hidden_units, 22), random_state=rnd
            ).astype(np.float32)
        )

        # feature layer
        self.features[0].weight.data.copy_(self.features[0].weight_prior.data)
        nn.init.constant_(self.features[0].bias_prior, 0.)
        nn.init.constant_(self.features[0].bias, 0.)
        nn.init.constant_(self.features[0].weight_log_std, self.prior_log_std.data[0])
        nn.init.constant_(self.features[0].bias_log_std, self.prior_log_std.data[0])
        self.num_weights += np.prod(self.features[0].weight.size())
        self.num_weights += np.prod(self.features[0].bias.size())

        # last layer
        self.f_last.weight_prior.data = torch.from_numpy(
            truncnorm.rvs(
                a=-2. / self.num_hidden_units, b=2. / self.num_hidden_units,
                size=(self.num_last_units, self.num_hidden_units), random_state=rnd
            ).astype(np.float32)
        )
        self.f_last.weight.data.copy_(self.f_last.weight_prior.data)
        nn.init.constant_(self.f_last.bias_prior, 0.)
        nn.init.constant_(self.f_last.bias, 0.)
        nn.init.constant_(self.f_last.weight_log_std, self.prior_log_std.data[0])
        nn.init.constant_(self.f_last.bias_log_std, self.prior_log_std.data[0])
        self.num_weights += np.prod(self.f_last.weight.size())
        self.num_weights += np.prod(self.f_last.bias.size())
