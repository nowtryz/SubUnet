from typing import Optional, Sequence

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from metrics import DiceScore, JaccardIndex, TverskyIndex


class DiceLoss(DiceScore):
    r"""Sørensen–Dice Loss

    The Dice coefficient, or Dice-Sørensen coefficient, is a common metric for pixel segmentation that can also be
    modified to act as a loss function:

    .. math::
        DSC(X, Y) = 1 - \frac{2 \left| X + Y \right|}{\left| X \right| + \left| Y \right|}

    The loss functions used the value of the probabilities rather than amount of member in sets as this calculation
    would not be derivable

    Inspired by https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """

    def __init__(self, n_classes, apply_softmax=True, skip_first_class=True, smooth=1e-7):
        super(DiceLoss, self).__init__(n_classes, True, apply_softmax, skip_first_class, smooth)

    def forward(self, inputs: torch.Tensor, truth: torch.Tensor):
        dice_coefficient = super().forward(inputs, truth)
        return 1 - dice_coefficient


class JaccardLoss(JaccardIndex):
    r"""The Jaccard loss

    Pytorch loss function based on the Jaccard Index.
    The Jaccard index, also known as the Jaccard similarity coefficient or Intersection Over Union

    .. math::
        DSC(X, Y) = 1 - \frac{2 \left| X + Y \right|}{\left| X \right| + \left| Y \right|}

    The loss functions used the value of the probabilities rather than amount of member in sets as this calculation
    would not be derivable

    Inspired by https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """

    def __init__(self, n_classes, apply_softmax=True, skip_first_class=True, smooth=1e-7):
        super(JaccardLoss, self).__init__(n_classes, True, apply_softmax, skip_first_class, smooth)

    def forward(self, inputs: torch.Tensor, truth: torch.Tensor):
        jaccard_index = super().forward(inputs, truth)
        return 1 - jaccard_index


class TverskyLoss(TverskyIndex):
    r"""Tversky Loss

    .. math::
        TL = 1 - TI

    The loss functions used the value of the probabilities rather than amount of member in sets as this calculation
    would not be derivable
    """

    def __init__(self, n_classes, alpha=0.1, beta=0.9, apply_softmax=True, skip_first_class=True, smooth=1e-7):
        super(TverskyLoss, self).__init__(n_classes, alpha, beta, True, apply_softmax, skip_first_class, smooth)

    def forward(self, inputs: torch.Tensor, truth: torch.Tensor):
        tversky_index = super().forward(inputs, truth)
        return 1 - tversky_index


class FocalTverskyLoss(TverskyIndex):
    r"""Focal Tversky Loss

    .. math::
        FTL = (1 - TI)^\gamma

    The loss functions used the value of the probabilities rather than amount of member in sets as this calculation
    would not be derivable
    """

    def __init__(self, n_classes, alpha=0.1, beta=0.9, gamma=2, apply_softmax=True, skip_first_class=True, smooth=1e-7):
        super(FocalTverskyLoss, self).__init__(n_classes, alpha, beta, True, apply_softmax, skip_first_class, smooth)
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, truth: torch.Tensor):
        tversky_index = super().forward(inputs, truth)
        return (1 - tversky_index) ** self.gamma


class CombinedLoss(nn.Module):
    def __init__(self, loss1, loss2, alpha=.5):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha

    def forward(self, *args):
        return (1 - self.alpha) * self.loss1(*args) + self.alpha * self.loss2(*args)


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
