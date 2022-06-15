import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CustomAct(nn.Module):
    def __init__(self, inplace: bool = False):
        super(CustomAct, self).__init__()

        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        input[:, 0] = torch.tanh(input[:, 0])
        if input.shape[1] == 2:
            # input[:, 1] = F.threshold(input[:, 1], 0, 1)
            input[:, 1] = torch.tanh(input[:, 1])

        return input


def conv(conv_sizes=(32, 64, 64), dense_sizes=(512, 128), activation=nn.ReLU):
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


def conv_last(out_size=2, input_size=130, activation=CustomAct):
    return nn.Sequential(
        nn.Linear(input_size, out_size),
        activation(),
    )
