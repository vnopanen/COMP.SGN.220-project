#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import Tensor, flatten
from torch.nn import Module, Linear, Dropout2d, ReLU, BatchNorm2d, \
    MaxPool2d, Conv2d, Sequential

__docformat__ = 'reStructuredText'
__all__ = ['CNNSystem']


class CNNSystem(Module):

    def __init__(self,
                 num_channels: int,
                 in_features: int,
                 output_classes: int) -> None:
        """CNNSystem, using four CNN layers, each followed batch norm, ReLU and
        max pooling. Additionally two fully connected layers, where the first 
        one is using ReLU as activation.

        :param num_channels: Input channels of first CNN.
        :type num_channels: int
        :param in_features: Input features of first linear layer.
        :type in_features: int
        :param output_classes: Output classes of the last linear layer.
        :type output_classes: int
        """
        super().__init__()

        cnn_channels_out_1 = 8
        cnn_channels_out_2 = 16
        cnn_channels_out_3 = 32
        cnn_channels_out_4 = 64

        self.block_1 = Sequential(
            Conv2d(in_channels=num_channels,
                   out_channels=cnn_channels_out_1,
                   kernel_size=4,
                   stride=1,
                   padding=0),
            BatchNorm2d(cnn_channels_out_1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))

        self.block_2 = Sequential(
            Conv2d(in_channels=cnn_channels_out_1,
                   out_channels=cnn_channels_out_2,
                   kernel_size=3,
                   stride=1,
                   padding=0),
            BatchNorm2d(cnn_channels_out_2),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))

        self.block_3 = Sequential(
            Conv2d(in_channels=cnn_channels_out_2,
                   out_channels=cnn_channels_out_3,
                   kernel_size=2,
                   stride=1,
                   padding=0),
            BatchNorm2d(cnn_channels_out_3),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))

        self.block_4 = Sequential(
            Conv2d(in_channels=cnn_channels_out_3,
                   out_channels=cnn_channels_out_4,
                   kernel_size=2,
                   stride=1,
                   padding=0),
            BatchNorm2d(cnn_channels_out_4),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2))

        self.fc_1 =  Sequential(
            Linear(in_features=in_features,
                   out_features=500),
            ReLU(),
            Dropout2d(0.25))

        self.fc_2 =  Sequential(
            Linear(in_features=500,
                   out_features=output_classes),
            Dropout2d(0.5))


    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        :param x: Input features.
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        # Convolutional layers
        h = self.block_1(x)
        h = self.block_2(h)
        h = self.block_3(h)
        h = self.block_4(h)

        # Fully connected layers. Flatten the tensor first.
        h = h.permute(0, 2, 1, 3).contiguous().view(x.size()[0], -1)
        h = self.fc_1(h)
        h = self.fc_2(h)
        return h