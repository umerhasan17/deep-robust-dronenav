import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import Flatten
from planner.base_cnn import BaseCNN


class MidLevelCNN(BaseCNN):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(
            self,
            observation_space,
            output_size,
    ):
        super().__init__()

        assert "midlevel" in observation_space.spaces
        self._n_input_mid_level = observation_space.spaces["midlevel"].shape[2]

        cnn_dims = np.array(observation_space.spaces["midlevel"].shape[:2], dtype=np.uint8)

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            print('CNN DIMS, OUTPUT SIZE', cnn_dims, output_size)

            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_mid_level,
                    out_channels=16,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=16,
                    out_channels=8,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=8,
                    out_channels=4,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                ),
                nn.ReLU(True),
                Flatten(),
                nn.Linear(400, output_size),
                nn.ReLU(True),
            )

        self.layer_init()

    @property
    def is_blind(self):
        return self._n_input_mid_level == 0

    def forward(self, observations):
        midlevel_observations = observations["midlevel"]
        midlevel_observations = midlevel_observations.permute(0, 3, 1, 2)
        midlevel_observations = midlevel_observations / 255.0  # TODO normalise mid level
        cnn_input = [midlevel_observations]
        cnn_input = torch.cat(cnn_input, dim=1)
        return self.cnn(cnn_input)
