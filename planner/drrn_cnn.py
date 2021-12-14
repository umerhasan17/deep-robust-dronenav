import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import Flatten


class MapPlanner(nn.Module):

    def __init__(self):
        super().__init__()

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
            nn.Linear(400, 512),
            nn.ReLU(True),
        )


    def forward(self, observations):
        midlevel_observations = observations["midlevel"]
        midlevel_observations = midlevel_observations.permute(0, 3, 1, 2)
        midlevel_observations = midlevel_observations / 255.0  # TODO normalise mid level
        cnn_input = [midlevel_observations]
        cnn_input = torch.cat(cnn_input, dim=1)
        return self.cnn(cnn_input)
