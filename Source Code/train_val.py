from torch import nn

from metrics import Metrics
from utils import to_var, getTargetSegmentation


def single_output_training(net: nn.Module, batch_idx, images, labels, epoch_metrics: Metrics):
    # -- The CNN makes its predictions (forward pass)
    net_predictions = net(images)

    # -- Compute the loss --#
    segmentation_classes = getTargetSegmentation(labels)
    loss_epoch = epoch_metrics.collect_and_get_loss(batch_idx, net_predictions, segmentation_classes)
    return loss_epoch

def single_lr_training(net: nn.Module, images, labels, epoch_metrics: Metrics):
    # -- The CNN makes its predictions (forward pass)
    net_predictions = net(images)

    # -- Compute the loss --#
    segmentation_classes = getTargetSegmentation(labels)
    loss_epoch = epoch_metrics.get_loss(net_predictions, segmentation_classes)
    return loss_epoch


def multi_output_training(net: nn.Module, batch_idx, images, labels, epoch_metrics: Metrics):
    # -- The CNN makes its predictions (forward pass)
    net_predictions, x1, x2, x3 = net(images)

    # -- Compute the loss --#
    segmentation_classes = getTargetSegmentation(labels)
    loss_epoch = epoch_metrics.collect_and_get_loss(batch_idx, net_predictions, segmentation_classes)
    loss_sum = (epoch_metrics.get_loss(x1, segmentation_classes)
                + epoch_metrics.get_loss(x2, segmentation_classes)
                + epoch_metrics.get_loss(x3, segmentation_classes)
                + loss_epoch)
    return loss_sum

def multi_lr_training(net: nn.Module, images, labels, epoch_metrics: Metrics):
    # -- The CNN makes its predictions (forward pass)
    net_predictions, x1, x2, x3 = net(images)

    # -- Compute the loss --#
    segmentation_classes = getTargetSegmentation(labels)
    loss_sum = (epoch_metrics.get_loss(net_predictions, segmentation_classes)
                + epoch_metrics.get_loss(x1, segmentation_classes)
                + epoch_metrics.get_loss(x2, segmentation_classes)
                + epoch_metrics.get_loss(x3, segmentation_classes))
    return loss_sum


def single_output_validation(net: nn.Module, loader, validation_metrics: Metrics):
    for batch_idx, data in enumerate(loader):
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        validation_metrics.collect(batch_idx, net(images), getTargetSegmentation(labels))


# def multi_output_validation(net, loader, validation_metrics):
#     for batch_idx, data in enumerate(loader):
#         images, labels, img_names = data

#         images = to_var(images)
#         labels = to_var(labels)

#         validation_metrics.collect(batch_idx, net(images), getTargetSegmentation(labels))
