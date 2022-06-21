from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CustomAct(nn.Module):
    def __init__(self, inplace: bool = False):
        super(CustomAct, self).__init__()

        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        out = torch.zeros(input.shape).to(input.device)
        out[:, 0] = torch.tanh(input[:, 0])
        if input.shape[1] == 2:
            out[:, 1] = torch.clamp(input[:, 1], 0, 1)

        return out


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)


def conv(conv_sizes: Tuple, dense_sizes: Tuple, activation: nn.Module):
    return nn.Sequential(
        nn.Conv2d(5, conv_sizes[0], 3, 1),
        activation(),
        nn.Conv2d(conv_sizes[0], conv_sizes[1], 3, 1),
        activation(),
        nn.Conv2d(conv_sizes[1], conv_sizes[2], 3, 1),
        activation(),
        nn.Flatten(),
        nn.Linear(18*18*conv_sizes[-1], dense_sizes[0]),
        activation(),
        nn.Linear(dense_sizes[0], dense_sizes[1]),
        activation()
    )


def conv_last(out_size: int, input_size: int, activation: nn.Module = CustomAct):
    return nn.Sequential(
        nn.Linear(input_size, input_size),
        nn.ReLU(),
        nn.Linear(input_size, input_size//2),
        nn.ReLU(),
        nn.Linear(input_size//2, out_size),
        activation(),
    )
