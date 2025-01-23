import torch
import numpy as np
from torch import nn


def sum_tensor(input_tensor, axes, keepdim=False):
    """
    Sums a tensor over specified axes.
    Args:
        input_tensor (torch.Tensor): Input tensor.
        axes (list or tuple): Axes to sum over.
        keepdim (bool): Whether to keep dimensions after summing.
    Returns:
        torch.Tensor: Summed tensor.
    """
    axes = np.unique(axes).astype(int)
    for ax in sorted(axes, reverse=True):
        input_tensor = input_tensor.sum(dim=ax, keepdim=keepdim)
    return input_tensor


def mean_tensor(input_tensor, axes, keepdim=False):
    """
    Computes the mean of a tensor over specified axes.
    Args:
        input_tensor (torch.Tensor): Input tensor.
        axes (list or tuple): Axes to average over.
        keepdim (bool): Whether to keep dimensions after averaging.
    Returns:
        torch.Tensor: Mean tensor.
    """
    axes = np.unique(axes).astype(int)
    for ax in sorted(axes, reverse=True):
        input_tensor = input_tensor.mean(dim=ax, keepdim=keepdim)
    return input_tensor


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., apply_nonlin=None, batch_dice=False, do_bg=True, smooth_in_nom=True,
                 background_weight=1, rebalance_weights=None):
        """
        Soft Dice Loss for segmentation tasks.
        Args:
            smooth (float): Smoothing factor to prevent division by zero.
            apply_nonlin (callable): Non-linear activation to apply on predictions.
            batch_dice (bool): Whether to calculate Dice loss across the batch.
            do_bg (bool): Whether to include the background class in loss calculation.
            smooth_in_nom (bool): Whether to include smoothing in the numerator.
            background_weight (float): Weight for the background class.
            rebalance_weights (list or np.ndarray): Class-specific weights for rebalancing.
        """
        super(SoftDiceLoss, self).__init__()
        if not do_bg and background_weight != 1:
            raise ValueError("If `do_bg` is False, `background_weight` must be 1.")
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth_in_nom = smooth_in_nom
        self.background_weight = background_weight
        self.rebalance_weights = rebalance_weights

    def forward(self, x, y):
        """
        Computes the Soft Dice Loss.
        Args:
            x (torch.Tensor): Predicted logits of shape (B, C, X, Y, ...).
            y (torch.Tensor): Ground truth labels of shape (B, 1, X, Y, ...).
        Returns:
            torch.Tensor: Dice loss.
        """
        with torch.no_grad():
            y = y.long()

        if self.apply_nonlin:
            x = self.apply_nonlin(x)

        if len(x.shape) != len(y.shape):
            y = y.view((y.size(0), 1, *y.shape[1:]))

        # Convert ground truth to one-hot encoding
        y_onehot = torch.zeros_like(x)
        y_onehot.scatter_(1, y, 1)

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        if self.batch_dice:
            return soft_dice_per_batch_2(x, y_onehot, self.smooth, self.smooth_in_nom,
                                         background_weight=self.background_weight,
                                         rebalance_weights=self.rebalance_weights)
        else:
            if self.background_weight != 1 or self.rebalance_weights is not None:
                raise NotImplementedError("Per-sample Dice loss with custom weights is not implemented.")
            return soft_dice(x, y_onehot, self.smooth, self.smooth_in_nom)


def soft_dice(net_output, gt, smooth=1., smooth_in_nom=1.):
    """
    Computes the Soft Dice Loss per sample.
    """
    axes = tuple(range(2, len(net_output.shape)))
    intersection = sum_tensor(net_output * gt, axes)
    denominator = sum_tensor(net_output + gt, axes)
    dice_loss = -((2 * intersection + smooth_in_nom) / (denominator + smooth)).mean()
    return dice_loss


def soft_dice_per_batch_2(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1, rebalance_weights=None):
    """
    Computes the Soft Dice Loss across the batch.
    """
    axes = tuple([0] + list(range(2, len(net_output.shape))))
    intersection = sum_tensor(net_output * gt, axes)
    net_output_square = sum_tensor(net_output ** 2, axes)
    gt_square = sum_tensor(gt ** 2, axes)

    weights = torch.ones_like(intersection)
    weights[0] = background_weight

    if rebalance_weights is not None:
        rebalance_weights = torch.tensor(rebalance_weights, dtype=weights.dtype, device=weights.device)
        intersection = intersection * rebalance_weights

    dice_loss = (1 - (2 * intersection + smooth_in_nom) /
                 (net_output_square + gt_square + smooth)) * weights
    dice_loss = dice_loss[dice_loss > 0]  # Ignore missing classes
    return dice_loss.mean()


class MultipleOutputLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Applies a loss function to multiple outputs.
        Args:
            loss (callable): Loss function.
            weight_factors (list or None): Weights for each output.
        """
        super(MultipleOutputLoss, self).__init__()
        self.loss = loss
        self.weight_factors = weight_factors

    def forward(self, outputs, target):
        """
        Args:
            outputs (list[torch.Tensor]): List of outputs.
            target (torch.Tensor): Ground truth.
        Returns:
            torch.Tensor: Weighted sum of losses.
        """
        if not isinstance(outputs, (list, tuple)):
            raise TypeError("Outputs must be a list or tuple of tensors.")

        weights = self.weight_factors or [1] * len(outputs)
        total_loss = weights[0] * self.loss(outputs[0], target)

        for i in range(1, len(outputs)):
            total_loss += weights[i] * self.loss(outputs[i], target)

        return total_loss
