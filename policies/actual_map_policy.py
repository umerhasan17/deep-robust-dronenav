""" 
policies/midlevel_map.py
----------------------------------------------------------------------------
     Authors : Yongtao Wu, Umer Hasan Sayed, Titouan Renard
     Last Update : December 2021

     policy class for agent which uses actual map to train

----------------------------------------------------------------------------
"""

import torch

from habitat.tasks.nav.nav import (
    IntegratedPointGoalGPSAndCompassSensor,
)
from habitat_baselines.rl.models.map_cnn import MapCNN
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Policy, Net


class PointNavDRRNActualMapPolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size=512):
        super().__init__(
            PointNavDRRNActualMapNet(
                observation_space=observation_space, hidden_size=hidden_size
            ),
            action_space.n,
        )


class PointNavDRRNActualMapNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size):
        super().__init__()

        assert IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces
        self._n_input_goal = observation_space.spaces[IntegratedPointGoalGPSAndCompassSensor.cls_uuid].shape[0]

        self._hidden_size = hidden_size

        self.map_cnn = MapCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size) + self._n_input_goal,
            self._hidden_size,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.map_cnn.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = observations[IntegratedPointGoalGPSAndCompassSensor.cls_uuid]
        x = [target_encoding]
        perception_embed = self.map_cnn(observations)  # encode back to policy
        x = [perception_embed] + x
        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states
