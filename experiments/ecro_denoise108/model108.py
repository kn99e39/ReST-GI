"""Matched extension of the frozen nonnegative R3 implementation."""
import torch
from torch import nn

import common108  # sets historical module paths
import model105 as base


def extend(conv):
    """Preserve the original seeded state and zero-initialize the new channel."""
    with torch.random.fork_rng(devices=[]):
        expanded = nn.Conv2d(conv.in_channels + 1, conv.out_channels, conv.kernel_size, padding=conv.padding)
    with torch.no_grad():
        expanded.weight[:, :-1].copy_(conv.weight)
        expanded.weight[:, -1].zero_()
        expanded.bias.copy_(conv.bias)
    return expanded


class Reconstructor(base.Reconstructor):
    def __init__(self):
        super().__init__("R3")
        self.net_low.enc[0][0] = extend(self.net_low.enc[0][0])
        self.net_high.enc[0][0] = extend(self.net_high.enc[0][0])


def loss(pred, parts, target, aux, boundary):
    low, high, total = base.features(target, aux)
    component_loss = base.rel_mse(parts["low"], low) + base.rel_mse(parts["high"], high)
    if not boundary:
        return component_loss
    cover = aux[:, 6:7]
    mask = (cover > 0) & (cover < 1)
    relative = (pred - total).square() / (pred.detach().square() + 0.01)
    return component_loss + 0.25 * (relative * mask).sum() / (3 * mask.sum()).clamp_min(1)
