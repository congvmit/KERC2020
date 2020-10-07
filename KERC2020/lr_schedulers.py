from torch.optim.lr_scheduler import _LRScheduler
import warnings


class BoundingExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch. 
    Learning rate set at min_lr when it is smaller than min_lr 
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, initial_lr=0.01, min_lr=0.001, last_epoch=-1):
        self.gamma = gamma
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        super().__init__(optimizer, last_epoch)

    def _compute_lr(self, base_lr):
        if base_lr * self.gamma <= self.min_lr:
            return self.min_lr
        else:
            return base_lr * self.gamma

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs

        return [self._compute_lr(group['lr']) for group in self.optimizer.param_groups]
