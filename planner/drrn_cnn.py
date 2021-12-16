import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import Flatten

from config.config import BATCHSIZE, MAP_DIMENSIONS


class MapPlanner(nn.Module):

    def __init__(self, output_size):
        super().__init__()

        self.output_size = output_size

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=4,
                kernel_size=(5, 5),
                stride=(2, 2),
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(5, 5),
                stride=(1, 1),
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3, 3),
                stride=(1, 1),
            ),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(115200, 512),
            nn.ReLU(True),
            nn.Linear(512, self.output_size),
            nn.Tanh(),
        )

    def forward(self, new_map):
        """ Returns encoding for new map """
        return self.cnn(new_map)
