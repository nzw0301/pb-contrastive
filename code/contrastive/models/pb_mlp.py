# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.stats import truncnorm
# from torch.nn import Parameter
#
#
# class StochasticGaussianLinear(nn.Module):
#     __constants__ = ['bias', 'in_features', 'out_features']
#
#     def __init__(self, in_features, out_features):
#         super(StochasticGaussianLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         self.weight_log_std = Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
#         self.weight_prior = Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
#
#         self.bias = Parameter(torch.Tensor(out_features))
#         self.bias_log_std = Parameter(torch.Tensor(out_features), requires_grad=False)
#         self.bias_prior = Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
#
#         self.weight_noise = Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
#         self.bias_noise = Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
#
#     def forward(self, input):
#         return F.linear(input, self.realised_weight, self.realised_bias)
#
#     def extra_repr(self):
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )
#
#     def sample_noise(self):
#         self.realised_weight = self.weight + self.weight_noise.normal_() * torch.exp(self.weight_log_std)
#         self.realised_bias = self.bias + self.bias_noise.normal_() * torch.exp(self.bias_log_std)
#
#
# class StochasticCauchyLinear(nn.Module):
#     __constants__ = ['bias', 'in_features', 'out_features']
#
#     def __init__(self, in_features, out_features):
#         super(StochasticCauchyLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         self.log_scale = Parameter(torch.Tensor(out_features, in_features))
#         self.weight_prior = Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
#
#         self.bias = Parameter(torch.Tensor(out_features))
#         self.log_bias_scale = Parameter(torch.Tensor(out_features))
#         self.bias_prior = Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
#
#         self.weight_noise = Parameter(torch.Tensor(self.weight.size()), requires_grad=False)
#         self.bias_noise = Parameter(torch.Tensor(self.bias.size()), requires_grad=False)
#
#     def forward(self, input):
#         return F.linear(input, self.realised_weight, self.realised_bias)
#
#     def extra_repr(self):
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )
#
#     def sample_noise(self):
#         self.realised_weight = self.weight + self.weight_noise.cauchy_() * torch.exp(self.log_scale)
#         self.realised_bias = self.bias + self.bias_noise.cauchy_() * torch.exp(self.log_bias_scale)
#
#
# class StochasticCauchyMLP(nn.Module):
#     def __init__(self, rnd: np.random.RandomState, num_hidden_units=300, num_last_units=100, init_weights=True,
#                  log_prior_scale=0., delta=0.05, l_max=1.):
#         """
#         :param rnd:
#         :param num_hidden_units:
#         :param num_last_units:
#         :param init_weights:
#         :param log_prior_scale: Prior's scale parameter and posterior's scale's initialized value
#         :param delta: confidence parameter
#         :param l_max: upper bound of the loss function.
#         """
#         assert 0. < delta < 1.
#         assert l_max > 0.
#
#         self.l_max = l_max
#         self.num_hidden_units = num_hidden_units
#         super(StochasticCauchyMLP, self).__init__()
#         self.num_last_units = num_last_units
#
#         self.features = nn.Sequential(
#             StochasticCauchyLinear(22, self.num_hidden_units),
#             nn.ReLU(inplace=True),
#         )
#         self.f_last = StochasticCauchyLinear(self.num_hidden_units, num_last_units)
#
#         self.delta = Parameter(torch.Tensor([delta]), requires_grad=False)
#         self.num_weights = 0
#         self.log_prior_scale = Parameter(torch.Tensor([log_prior_scale]), requires_grad=False)
#
#         if init_weights:
#             self._initialize_weights(rnd)
#
#     def forward(self, inputs):
#         x = self.features(inputs)
#         x = self.f_last(x)
#         return x
#
#     def sample_noise(self):
#         """
#         Sample weights and biases from the posterior.
#         :return: None
#         """
#         for m in self.modules():
#             if isinstance(m, StochasticCauchyLinear):
#                 m.sample_noise()
#
#     def chi_square_divergence(self, num_samples=1):
#         num_weights = self.num_weights
#         assert self.num_weights > 0
#
#         sum_posterior_log_scale = []
#         for m in self.modules():
#             if isinstance(m, StochasticCauchyLinear):
#                 sum_posterior_log_scale.append(
#                     torch.sum(
#                         torch.exp(m.log_scale)
#                     )
#                 )
#                 sum_posterior_log_scale.append(
#                     torch.sum(
#                         torch.exp(m.log_bias_scale)
#                     )
#                 )
#         sum_posterior_log_scale = torch.sum(torch.stack(sum_posterior_log_scale))
#         sum_prior_log_scale = num_weights * torch.exp(self.log_prior_scale)
#
#         prior_location_part = []
#         posterior_location_part = []
#         cauchy_dist = torch.distributions.Cauchy(loc=torch.tensor([0.]), scale=torch.tensor([1.]))
#
#         for m in self.modules():
#             if isinstance(m, StochasticCauchyLinear):
#                 h = m.weight_prior + torch.exp(self.log_prior_scale) * cauchy_dist.sample(
#                     sample_shape=m.log_scale.size()).view_as(m.log_scale)
#                 h_b = m.bias_prior + torch.exp(self.log_prior_scale) * cauchy_dist.sample(
#                     sample_shape=m.log_bias_scale.size()).view_as(m.log_bias_scale)
#
#                 posterior_location_part.append(
#                     torch.sum(
#                         (h - m.weight) ** 2 / torch.exp(m.log_scale)
#                     ) + torch.sum(
#                         (h_b - m.bias) ** 2 / torch.exp(m.log_bias_scale)
#                     )
#                 )
#
#                 prior_location_part.append(
#                     torch.sum(
#                         ((h - m.weight_prior) ** 2) / torch.exp(self.log_prior_scale)
#                     ) + torch.sum(
#                         ((h_b - m.bias_prior) ** 2) / torch.exp(self.log_prior_scale)
#                     )
#                 )
#
#         denom = torch.sum(torch.stack(posterior_location_part)) + 1.
#         num = torch.sum(torch.stack(prior_location_part)) + 1.
#
#         inner = -sum_posterior_log_scale - (1 + num_weights) * torch.log(denom) \
#                 + sum_prior_log_scale + (1 + num_weights) * torch.log(num)
#
#         return torch.exp(inner) - 1.
#
#     def pac_bayes_objective(self, contrastive_loss, num_samples=1):
#         complexity = torch.sqrt(self.l_max ** 2 / self.delta) + torch.sqrt(self.chi_square_divergence(num_samples) + 1.)
#         objective = contrastive_loss + complexity
#         return objective, complexity
#
#     def _initialize_weights(self, rnd: np.random.RandomState):
#         # Initialisation is based on the following repo
#         # https://github.com/gkdziugaite/pacbayes-opt/blob/58beae1f63ce0efaf749757a7c98eac2c8414238/snn/core/cnn_fn.py#L101
#
#         self.features[0].weight_prior.data = torch.from_numpy(
#             truncnorm.rvs(
#                 a=-1. / 11., b=1. / 11., size=(self.num_hidden_units, 22), random_state=rnd
#             ).astype(np.float32)
#         )
#
#         # feature layer
#         self.features[0].weight.data.copy_(self.features[0].weight_prior.data)
#         nn.init.constant_(self.features[0].bias_prior, 0.)
#         nn.init.constant_(self.features[0].bias, 0.)
#         nn.init.constant_(self.features[0].log_scale, self.log_prior_scale.data[0])
#         nn.init.constant_(self.features[0].log_bias_scale, self.log_prior_scale.data[0])
#         self.num_weights += np.prod(self.features[0].weight.size())
#         self.num_weights += np.prod(self.features[0].bias.size())
#
#         # last layer
#         self.f_last.weight_prior.data = torch.from_numpy(
#             truncnorm.rvs(
#                 a=-2. / self.num_hidden_units, b=2. / self.num_hidden_units,
#                 size=(self.num_last_units, self.num_hidden_units), random_state=rnd
#             ).astype(np.float32)
#         )
#         self.f_last.weight.data.copy_(self.f_last.weight_prior.data)
#         nn.init.constant_(self.f_last.bias_prior, 0.)
#         nn.init.constant_(self.f_last.bias, 0.)
#         nn.init.constant_(self.f_last.log_scale, self.log_prior_scale.data[0])
#         nn.init.constant_(self.f_last.log_bias_scale, self.log_prior_scale.data[0])
#         self.num_weights += np.prod(self.f_last.weight.size())
#         self.num_weights += np.prod(self.f_last.bias.size())
#
#     def deterministic(self):
#         """
#         Only Use median values for feed forwarding
#         :return: None
#         """
#         for m in self.modules():
#             if isinstance(m, StochasticCauchyLinear):
#                 m.realised_weight = m.weight.detach()
#                 m.realised_bias = m.bias.detach()
#
#
# class StochasticGaussianMLP(nn.Module):
#     def __init__(self, rnd: np.random.RandomState, num_hidden_units=50, num_last_units=50, init_weights=True,
#                  prior_log_std=0., delta=0.05, l_max=1.):
#         """
#         :param rnd:
#         :param num_hidden_units:
#         :param num_last_units:
#         :param init_weights:
#         :param prior_log_std: Prior's scale parameter and posterior's scale's initialized value
#         :param delta: confidence parameter
#         :param l_max: upper bound of the loss function.
#         """
#         assert 0. < delta < 1.
#         assert l_max > 0.
#
#         super(StochasticGaussianMLP, self).__init__()
#         self.l_max = l_max
#         self.num_hidden_units = num_hidden_units
#         self.num_last_units = num_last_units
#
#         self.features = nn.Sequential(
#             StochasticGaussianLinear(22, self.num_hidden_units),
#             nn.ReLU(inplace=True),
#         )
#         self.f_last = StochasticGaussianLinear(self.num_hidden_units, num_last_units)
#
#         self.delta = Parameter(torch.Tensor([delta]), requires_grad=False)
#         self.num_weights = 0
#         self.prior_log_std = Parameter(torch.Tensor([prior_log_std]), requires_grad=False)
#
#         if init_weights:
#             self._initialize_weights(rnd)
#
#     def forward(self, inputs):
#         x = self.features(inputs)
#         x = self.f_last(x)
#         return x
#
#     def sample_noise(self):
#         """
#         Sample weights and biases from the posterior.
#         :return: None
#         """
#         for m in self.modules():
#             if isinstance(m, StochasticGaussianLinear):
#                 m.sample_noise()
#
#     def chi_square_divergence(self):
#         variance = torch.exp(2. * self.prior_log_std)
#         mean_part = []
#
#         for m in self.modules():
#             if isinstance(m, StochasticGaussianLinear):
#                 mean_part.append(
#                     torch.sum(
#                         (m.weight - m.weight_prior) ** 2
#                     )
#                     + torch.sum(
#                         (m.bias - m.bias_prior) ** 2
#                     )
#                 )
#
#         inner = torch.sum(torch.stack(mean_part)) / variance
#         return torch.exp(inner) - 1.
#
#     def pac_bayes_objective(self, contrastive_loss, num_samples=1):
#         complexity = torch.sqrt(self.l_max ** 2 / self.delta) * torch.sqrt(self.chi_square_divergence() + 1.)
#         objective = contrastive_loss + complexity
#         return objective, complexity
#
#     def _initialize_weights(self, rnd: np.random.RandomState):
#         # Initialisation is based on the following repo
#         # https://github.com/gkdziugaite/pacbayes-opt/blob/58beae1f63ce0efaf749757a7c98eac2c8414238/snn/core/cnn_fn.py#L101
#
#         self.features[0].weight_prior.data = torch.from_numpy(
#             truncnorm.rvs(
#                 a=-1. / 11., b=1. / 11., size=(self.num_hidden_units, 22), random_state=rnd
#             ).astype(np.float32)
#         )
#
#         # feature layer
#         self.features[0].weight.data.copy_(self.features[0].weight_prior.data)
#         nn.init.constant_(self.features[0].bias_prior, 0.)
#         nn.init.constant_(self.features[0].bias, 0.)
#         nn.init.constant_(self.features[0].weight_log_std, self.prior_log_std.data[0])
#         nn.init.constant_(self.features[0].bias_log_std, self.prior_log_std.data[0])
#         self.num_weights += np.prod(self.features[0].weight.size())
#         self.num_weights += np.prod(self.features[0].bias.size())
#
#         # last layer
#         self.f_last.weight_prior.data = torch.from_numpy(
#             truncnorm.rvs(
#                 a=-2. / self.num_hidden_units, b=2. / self.num_hidden_units,
#                 size=(self.num_last_units, self.num_hidden_units), random_state=rnd
#             ).astype(np.float32)
#         )
#         self.f_last.weight.data.copy_(self.f_last.weight_prior.data)
#         nn.init.constant_(self.f_last.bias_prior, 0.)
#         nn.init.constant_(self.f_last.bias, 0.)
#         nn.init.constant_(self.f_last.weight_log_std, self.prior_log_std.data[0])
#         nn.init.constant_(self.f_last.bias_log_std, self.prior_log_std.data[0])
#         self.num_weights += np.prod(self.f_last.weight.size())
#         self.num_weights += np.prod(self.f_last.bias.size())
#
#     def deterministic(self):
#         """
#         Only Use median values for feed forwarding
#         :return: None
#         """
#         for m in self.modules():
#             if isinstance(m, StochasticGaussianLinear):
#                 m.realised_weight = m.weight.detach()
#                 m.realised_bias = m.bias.detach()
