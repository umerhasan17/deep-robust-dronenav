import numpy as np
import torch.nn as nn


class BaseCNN(nn.Module):
    r""" Modified from SimpleCNN """

    def _conv_output_dim(
            self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                                (
                                        dimension[i]
                                        + 2 * padding[i]
                                        - dilation[i] * (kernel_size[i] - 1)
                                        - 1
                                )
                                / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        return NotImplemented()

    def forward(self, observations):
        return NotImplemented()
