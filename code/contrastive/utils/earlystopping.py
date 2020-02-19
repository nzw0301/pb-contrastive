import numpy as np
import torch


class EarlyStopping(object):
    _valid_modes = ['min', 'max']

    def __init__(
            self,
            mode='min',
            min_delta=0.,
            patience=10
    ):
        """
        :param mode: min or max. if your metric is a value such that higher is better like val. accuracy,
            you should use `max`. On the other hand, if your metric is a value such that lower is better
            like validation loss / PAC-Bayes bound, you should use `min`.
        :param min_delta: If new metric is better only `min_delta`, early stop count increases.
        :param patience: How many times patience from the best value.
        """
        if mode not in self._valid_modes:
            valid_modes = ', '.join(self._valid_modes)
            raise ValueError(
                'mode {} is not supported. You must pass one of [{}] to `mode`.'.format(mode, valid_modes)
            )
        if patience <= 0:
            raise ValueError('`patient` must be positive.')

        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.num_bad_epochs = 0

        if mode == 'min':
            self._is_better = lambda a, best: a < best - self.min_delta
            self.best = np.finfo(np.float(0.)).max
        else:
            self._is_better = lambda a, best: a > best + self.min_delta
            self.best = np.finfo(np.float(0.)).min

    def is_stopped_and_save(self, metric, model, save_name):
        """
        If the new metric becomes the best, model parameters are saved.
        Else early-stop's count increases.

        If early-stop count reaches `patience`, return `True`.
        Else return `False`

        :param metric: Monitored value such as validation accuracy / loss.
        :param model: PyTorch's model
        :param save_name: File name to save `model`'s parameters.
        :return: Boolean
        """

        if np.isnan(metric):
            raise ValueError('The metric becomes `nan`. Stop training.')

        if self._is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
            torch.save(model.state_dict(), save_name)
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience
