#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Tuple
from torch import Tensor
from torch.nn import Module, Linear, Dropout2d, ReLU, BatchNorm2d, \
    MaxPool2d, Conv2d, Sequential, Softmax2d

__docformat__ = 'reStructuredText'
__all__ = ['CNNSystem']


class CNNSystem(Module):

    def __init__(self,
                 cnn_channels_in_1: int,
                 cnn_kernel_1: Union[Tuple[int], int],
                 cnn_stride_1: Union[Tuple[int], int],
                 cnn_padding_1: Union[Tuple[int], int],
                 pooling_kernel_1: Union[Tuple[int], int],
                 pooling_stride_1: Union[Tuple[int], int],
                 cnn_channels_out_2: int,
                 cnn_kernel_2: Union[Tuple[int], int],
                 cnn_stride_2: Union[Tuple[int], int],
                 cnn_padding_2: Union[Tuple[int], int],
                 pooling_kernel_2: Union[Tuple[int], int],
                 pooling_stride_2: Union[Tuple[int], int],
                 classifier_input_features: int,
                 output_classes: int) -> None:
        """CNNSystem, using four CNN layers, each followed batch norm, ReLU and
        max pooling. Additionally two fully connected layers, using ReLU as
        activation, L2 regularization and dropout.

        :param cnn_channels_out_1: Output channels of first CNN.
        :type cnn_channels_out_1: int
        :param cnn_kernel_1: Kernel shape of first CNN.
        :type cnn_kernel_1: int|Tuple[int, int]
        :param cnn_stride_1: Strides of first CNN.
        :type cnn_stride_1: int|Tuple[int, int]
        :param cnn_padding_1: Padding of first CNN.
        :type cnn_padding_1: int|Tuple[int, int]
        :param pooling_kernel_1: Kernel shape of first pooling.
        :type pooling_kernel_1: int|Tuple[int, int]
        :param pooling_stride_1: Strides of first pooling.
        :type pooling_stride_1: int|Tuple[int, int]
        :param cnn_channels_out_2: Output channels of second CNN.
        :type cnn_channels_out_2: int
        :param cnn_kernel_2: Kernel shape of second CNN.
        :type cnn_kernel_2: int|Tuple[int, int]
        :param cnn_stride_2: Strides of second CNN.
        :type cnn_stride_2: int|Tuple[int, int]
        :param cnn_padding_2: Padding of second CNN.
        :type cnn_padding_2: int|Tuple[int, int]
        :param pooling_kernel_2: Kernel shape of second pooling.
        :type pooling_kernel_2: int|Tuple[int, int]
        :param pooling_stride_2: Strides of second pooling.
        :type pooling_stride_2: int|Tuple[int, int]
        :param classifier_input_features: Input features to the classifier.
        :type classifier_input_features: int
        :param output_classes: Output classes.
        :type output_classes: int
        """
        super().__init__()

        self.block_1 = Sequential(
            Conv2d(in_channels=2,
                   out_channels=cnn_channels_out_1,
                   kernel_size=cnn_kernel_1,
                   stride=cnn_stride_1,
                   padding=cnn_padding_1),
            BatchNorm2d(cnn_channels_out_1),
            ReLU(),
            MaxPool2d(kernel_size=pooling_kernel_1,
                      stride=pooling_stride_1))

        self.block_2 = Sequential(
            Conv2d(in_channels=cnn_channels_out_1,
                   out_channels=cnn_channels_out_2,
                   kernel_size=cnn_kernel_2,
                   stride=cnn_stride_2,
                   padding=cnn_padding_2),
            BatchNorm2d(cnn_channels_out_2),
            ReLU(),
            MaxPool2d(kernel_size=pooling_kernel_2,
                      stride=pooling_stride_2))

        self.block_3 = Sequential(
            Conv2d(in_channels=cnn_channels_out_2,
                   out_channels=cnn_channels_out_3,
                   kernel_size=cnn_kernel_3,
                   stride=cnn_stride_3,
                   padding=cnn_padding_3),
            BatchNorm2d(cnn_channels_out_3),
            ReLU(),
            MaxPool2d(kernel_size=pooling_kernel_3,
                      stride=pooling_stride_3))

        self.block_4 = Sequential(
            Conv2d(in_channels=cnn_channels_out_3,
                   out_channels=cnn_channels_out_4,
                   kernel_size=cnn_kernel_4,
                   stride=cnn_stride_4,
                   padding=cnn_padding_4),
            BatchNorm2d(cnn_channels_out_4),
            ReLU(),
            MaxPool2d(kernel_size=pooling_kernel_4,
                      stride=pooling_stride_4))

        self.fc_1 =  Sequential(
            Linear(in_features=classifier_input_features, #?
                   out_features=500),
            ReLU(),
            Dropout2d(dropout=0.25))

        self.fc_2 =  Sequential(
            Linear(in_features=500,
                   out_features=output_classes),
            ReLU(),
            Dropout2d(dropout=0.5))


    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        :param x: Input features.
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        # Conv layers
        h = self.block_1(h)
        h = self.block_2(h)
        h = self.block_3(h)
        h = self.block_4(h)

        # Fit to fc layer? Flatten?
        #h = h.permute(0, 2, 1, 3).contiguous().view(x.size()[0], -1)

        # Fc layers
        h = self.fc_1(h)
        h = self.fc_2(h)
        return h