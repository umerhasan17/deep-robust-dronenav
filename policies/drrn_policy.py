""" 
policies/midlevel_map.py
----------------------------------------------------------------------------
     Authors : Yongtao Wu, Umer Hasan Sayed, Titouan Renard
     Last Update : December 2021

     policy class for the full RL agent

----------------------------------------------------------------------------
"""

import torch

from config.config import RESIDUAL_LAYERS_PER_BLOCK, RESIDUAL_NEURON_CHANNEL, RESIDUAL_SIZE, \
    STRIDES, MAP_DIMENSIONS, DEBUG, BATCHSIZE, device
from habitat.tasks.nav.nav import (
    IntegratedPointGoalGPSAndCompassSensor,
)
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Policy, Net
from mapper.map import convert_midlevel_to_map
from mapper.mid_level.decoder import UpResNet
from mapper.mid_level.fc import FC
from mapper.transform import egomotion_transform
from mapper.update import update_map
from planner.drrn_cnn import MapPlanner


class PointNavDRRNPolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size=512):
        super().__init__(
            PointNavDRRNNet(
                observation_space=observation_space, hidden_size=hidden_size
            ),
            action_space.n,
        )


class PointNavDRRNNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size):
        super().__init__()

        assert IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces
        self._n_input_goal = observation_space.spaces[IntegratedPointGoalGPSAndCompassSensor.cls_uuid].shape[0]

        self._hidden_size = hidden_size

        self.visual_encoder = MapPlanner(output_size=self._hidden_size)

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
        return False

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):

        target_encoding = observations[IntegratedPointGoalGPSAndCompassSensor.cls_uuid]
        x = [target_encoding]
        new_map = observations['midlevel_map']

        if DEBUG:
            print(f'New map generated of shape: {new_map.shape}')

        perception_embed = self.visual_encoder(new_map)  # encode back to policy
        x = [perception_embed] + x
        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
