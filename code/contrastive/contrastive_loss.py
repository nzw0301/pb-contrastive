import numpy as np
import torch
import torch.nn.functional as F


def logistic_loss(input, reduction='mean', hinge_tensor=None):
    """
    :param input: 2D tensor. shape: (mini-batch-size, num-negative)
    :param reduction: PyTorch loss's reduction parameter
    :param hinge_tensor: This tensor is ignored in this function.

    :return: FloatTensor of loss.
    """

    loss = torch.log2(1. + torch.sum(torch.exp(
        torch.clamp(-input, max=64.)  # avoid nan by taking exp
    ), dim=1))

    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(reduction + ' is not valid')


def hinge_loss(input, reduction='mean', hinge_tensor=None):
    """
    :param input: 2D tensor. shape: (mini-batch-size, num-negative). element value is f(x)^T [f(x^+) - f(x^-)]
    :param reduction: PyTorch loss's reduction parameter
    :param hinge_tensor: Hinge loss's tensor contains 0.

    :return: FloatTensor of loss.
    """

    loss = torch.max(
        hinge_tensor,
        1. + torch.max(-input, dim=1)[0],
    )

    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(reduction + ' is not valid')


def zero_one_loss(input, reduction='mean', hinge_tensor=None):
    """
    :param input: 2D tensor. shape: (mini-batch-size, num-negative). Each element value is f(x)^T [f(x^+) - f(x^-)].
    :param reduction: PyTorch loss's reduction parameter
    :param hinge_tensor: Hinge loss's tensor contains 0.

    :return: FloatTensor of loss.
    """

    loss = torch.sum(input <= 0., dim=1).float() / input.shape[1]

    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(reduction + ' is not valid')


class ContrastiveLoss(torch.nn.Module):
    _loss_names = ['hinge', 'logistic', 'zero-one']

    def __init__(self, loss_name, device='cpu'):
        """
        Contrastive loss class. This class supports Hinge loss and logistic loss.
        :param loss_name: loss function name either `hinge` or `logistic`.
        :param device: Device to store hinge loss's constant tensor.
        """

        assert loss_name in self._loss_names, \
            '{} is unsupported. loss_name must be `hinge`, `logistic`, or `zero-one`.'.format(loss_name)

        super(ContrastiveLoss, self).__init__()

        self.tensor_in_hinge = None
        if loss_name == 'hinge':
            self.loss = hinge_loss
            self.tensor_in_hinge = torch.zeros(1).to(device)
        elif loss_name == 'logistic':
            self.loss = logistic_loss
        elif loss_name == 'zero-one':
            self.loss = zero_one_loss

    def forward(self, feature, positive_feature, negative_features, reduction='mean'):
        """
        :param feature: shape: (mini-batch, num-features)
        :param positive_feature: shape: (mini-batch, size-blocks, num-features)
        :param negative_features: shape: (mini-batch, num-negatives, size-blocks, num-features)
        :param reduction: Same to PyTorch
        :return: Tensor of loss.
        """

        positive_feature = torch.mean(positive_feature, dim=1)  # (mini-batch, num-features)
        negative_features = torch.mean(negative_features, dim=2)  # (mini-batch, num-negatives, num-features)
        fxfx_pos = torch.sum(feature * positive_feature, dim=1, keepdim=True)  # (mini-batch, 1)
        fxfx_negs = torch.sum(
            torch.unsqueeze(feature, dim=1) * negative_features, dim=2
        )  # (mini-batch, negs)

        return self.loss(fxfx_pos - fxfx_negs, reduction=reduction, hinge_tensor=self.tensor_in_hinge)


class SupervisedLoss(torch.nn.Module):
    _loss_names = (
        'logistic',
        'hinge',
        'cross_entropy'
    )

    def __init__(self, loss, num_last_units=100, device='cpu'):
        """
        :param loss: loss name. Valid loss is in {logistic, hinge, cross_entropy}
        :param num_last_units: The number of classes
        :param device: PyTorch's device instance
        """

        super(SupervisedLoss, self).__init__()
        self.num_last_units = num_last_units
        self.device = device
        if loss == 'logistic':
            self.forward = self.logistic
        elif loss == 'hinge':
            self.forward = self.hinge
            self.tensor_in_hinge = torch.zeros(1).to(self.device)
        elif loss == 'cross_entropy':
            self.forward = self.cross_entropy
        else:
            raise ValueError

    @staticmethod
    def cross_entropy(model_output, targets, reduction='mean'):
        """
        :param model_output: Output tensor of supervised model. Shape is (mini_batch, num_classes)
        :param targets: Label long tensor. Shape is (mini_batch, num_classes)
        :param reduction: Same to PyTorch
        :return: FloatTensor contains loss value
        """

        output = F.log_softmax(model_output, dim=1)
        loss = F.nll_loss(output, targets, reduction=reduction)
        return loss

    def hinge(self, model_output, targets, reduction='mean'):
        """
        :param model_output: Output tensor of supervised model. Shape is (mini_batch, num_classes)
        :param targets: Label long tensor. Shape is (mini_batch, num_classes)
        :param reduction: Same to PyTorch
        :return: FloatTensor contains loss value
        """

        loss_input = self.make_loss_input(model_output, targets)
        return hinge_loss(loss_input, reduction=reduction, hinge_tensor=self.tensor_in_hinge)

    def logistic(self, model_output, targets, reduction='mean'):
        """
        :param model_output: Output tensor of supervised model. Shape is (mini_batch, num_classes)
        :param targets: Label long tensor. Shape is (mini_batch, num_classes)
        :param reduction: Same to PyTorch
        :return: FloatTensor contains loss value
        """

        loss_input = self.make_loss_input(model_output, targets)
        return logistic_loss(loss_input, reduction=reduction)

    def make_loss_input(self, model_output, targets):
        """
        Convert model's output tensor to calculate loss value.
        :param model_output: shape is (mini_batch, num_classes)
        :param targets: Label. Shape is (mini_batch, num_classes)
        :return: FloatTensor shape (mini-batch, num_classes)
        """

        num_mini_batch = len(targets)
        ids = torch.tensor(
            np.tile(np.arange(self.num_last_units), num_mini_batch).reshape(num_mini_batch, self.num_last_units)
        ).to(self.device)

        targets = targets.view(len(targets), 1)
        not_targets = torch.where(ids != targets)[1].view(num_mini_batch, self.num_last_units - 1)
        pos = torch.gather(model_output, dim=1, index=targets)
        neg = torch.gather(model_output, dim=1, index=not_targets)
        return pos - neg
