from torch.nn import functional as F
from torch import nn
import torch


class _Scorer(nn.Module):
    def __init__(self, n_classes, soft=False, apply_softmax=True, skip_first_class=True, smooth=1e-7):
        super(_Scorer, self).__init__()
        self.register_buffer('eye', torch.eye(n_classes))
        self.soft = soft
        self.n_classes = n_classes
        self.apply_softmax = apply_softmax
        self.skip_first_class = skip_first_class
        self.smooth = smooth

    def one_hot(self, x):
        # squeeze channels and convert to one hot, then move classes to second dimension
        x = self.eye[x.long()].permute(0, 3, 1, 2)
        if self.skip_first_class:
            x = x[:, 1-self.n_classes:, :, :]  # skip background (class 0)

        return x

    def transform_inputs(self, inputs: torch.Tensor, truth: torch.Tensor):
        truth = self.one_hot(truth)

        if self.apply_softmax:
            inputs = F.softmax(inputs, dim=1)

        if not self.soft:
            inputs = torch.argmax(inputs, dim=1)
            inputs = self.one_hot(inputs)
        elif self.skip_first_class:
            inputs = inputs[:, 1:, :, :]  # skip background

        return inputs, truth


class DiceScore(_Scorer):
    r"""Sørensen–Dice Score

    The Dice coefficient, or Dice-Sørensen coefficient, is a common metric for pixel segmentation that can also be
    modified to act as a loss function:

    .. math::
        DSC(X, Y) = \frac{2 \left| X + Y \right|}{\left| X \right| + \left| Y \right|}

    Inspired by https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """

    def __init__(self, n_classes, soft=False, apply_softmax=True, skip_first_class=True, smooth=1e-7):
        super(DiceScore, self).__init__(n_classes, soft, apply_softmax, skip_first_class, smooth)

    def forward(self, inputs: torch.Tensor, truth: torch.Tensor):
        inputs, truth = self.transform_inputs(inputs, truth)

        intersection = torch.sum(inputs * truth, dim=(0, 2, 3))
        cardinality = torch.sum(inputs ** 2 + truth ** 2, dim=(0, 2, 3))
        dice_coefficient = 2. * intersection / (cardinality + self.smooth)
        return dice_coefficient.mean()


class TverskyIndex(_Scorer):
    r"""Tversky Index

    The Tversky Index (TI) is a asymmetric similarity measure that is a 
    generalisation of the dice coefficient and the Jaccard index.

    .. math::
        TI = \frac{TP}{TP + \alpha FN +  \beta FP}
    """

    def __init__(self, n_classes, alpha=0.5, beta=0.5, soft=False, apply_softmax=True, skip_first_class=True, smooth=1e-7):
        super(TverskyIndex, self).__init__(n_classes, soft, apply_softmax, skip_first_class, smooth)
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs: torch.Tensor, truth: torch.Tensor):
        inputs, truth = self.transform_inputs(inputs, truth)

        intersection = torch.sum(inputs * truth, dim=(0, 2, 3))
        fps = torch.sum(inputs * (1 - truth), dim=(0, 2, 3))
        fns = torch.sum((1 - inputs) * truth, dim=(0, 2, 3))
        return (intersection / (intersection + (self.alpha * fps) + (self.beta * fns) + self.smooth)).mean()
        

class JaccardIndex(_Scorer):
    r"""The Jaccard index, also known as the Jaccard similarity coefficient or Intersection Over Union

    .. math::
        J(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}.
    """
    def __init__(self, n_classes, soft=False, apply_softmax=True, skip_first_class=True, smooth=1e-7):
        super(JaccardIndex, self).__init__(n_classes, soft, apply_softmax, skip_first_class, smooth)

    def forward(self, inputs: torch.Tensor, truth: torch.Tensor):
        inputs, truth = self.transform_inputs(inputs, truth)

        intersection = torch.sum(inputs * truth, dim=(0, 2, 3))
        union = torch.sum(inputs + truth, dim=(0, 2, 3)) - intersection
        iou = (intersection / (union + self.smooth))
        return iou.mean()


class Metrics(nn.Module):
    def __init__(self, buffer_size, num_classes, loss, device=None):
        super(Metrics, self).__init__()
        self.register_buffer("_losses", torch.zeros(buffer_size, dtype=torch.float32, device=device))
        self.register_buffer("_scores_iou", torch.zeros(buffer_size, dtype=torch.float32, device=device))
        self.register_buffer("_scores_dice", torch.zeros(buffer_size, dtype=torch.float32, device=device))
        self.register_buffer("_scores_soft_dice", torch.zeros(buffer_size, dtype=torch.float32, device=device))
        # self.register_buffer("_scores_hausdorff", torch.zeros(buffer_size, dtype=torch.double, device=device))

        self._loss = loss
        self._dice = DiceScore(num_classes)
        self._soft_dice = DiceScore(num_classes, soft=True)
        self._iou = JaccardIndex(num_classes)
        # self._hausdorff = AveragedHausdorffLoss()
        pass

    def collect_metrics_only(self, batch_index, net_predictions, segmentation_classes):
        self._scores_iou[batch_index] = self._iou(net_predictions, segmentation_classes).detach()
        self._scores_dice[batch_index] = self._dice(net_predictions, segmentation_classes).detach()
        self._scores_soft_dice[batch_index] = self._soft_dice(net_predictions, segmentation_classes).detach()
        # self._scores_hausdorff[batch_index] = self._hausdorff(net_predictions, segmentation_classes).detach()

    def collect_and_get_loss(self, batch_index, net_predictions, segmentation_classes):
        self.collect_metrics_only(batch_index, net_predictions, segmentation_classes)

        loss_value = self._loss(net_predictions, segmentation_classes)
        self._losses[batch_index] = loss_value.detach()
        return loss_value

    def collect(self, batch_index, net_predictions, segmentation_classes):
        self.collect_metrics_only(batch_index, net_predictions, segmentation_classes)
        self._losses[batch_index] = self._loss(net_predictions, segmentation_classes).detach()

    def get_loss(self, net_predictions, segmentation_classes):
        return self._loss(net_predictions, segmentation_classes)

    @property
    def loss(self):
        return self._losses.mean().item()

    @property
    def iou(self):
        return self._scores_iou.mean().item()

    @property
    def dice(self):
        return self._scores_dice.mean().item()

    @property
    def soft_dice(self):
        return self._scores_soft_dice.mean().item()

    # @property
    # def hausdorff(self):
    #     return self._scores_hausdorff.mean().item()

    def get_metrics(self):
        return self.loss, self.iou, self.dice, self.soft_dice  # , self.hausdorff


if __name__ == '__main__':
    from torch import Tensor
    dc = DiceScore(3)
    gt = Tensor([[[1, 0], [0, 2]]])
    pred = Tensor([[
        [[.1, .8],
         [.8, .1]],
        [[.8, .1],
         [.1, .1]],
        [[.1, .1],
         [.1, .8]]
    ]])

    print(pred)

    pred = torch.argmax(pred, dim=1)
    pred = torch.eye(3)[pred.long()]
    pred = pred.permute(0, 3, 1, 2)  # move classes to second dimension

    print(pred)

    # print(dc(pred, gt))
