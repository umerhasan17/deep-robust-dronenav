"""
Replace RGB observations with mid level representations in PPO policy
"""
import torch

from config.config import RESIDUAL_LAYERS_PER_BLOCK, RESIDUAL_NEURON_CHANNEL, RESIDUAL_SIZE, \
    STRIDES
from habitat.tasks.nav.nav import (
    IntegratedPointGoalGPSAndCompassSensor,
)
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.rl.ppo.policy import Policy, Net
from mapper.map import convert_rgb_obs_to_map
from mapper.mid_level.decoder import UpResNet
from mapper.mid_level.fc import FC


class PointNavBaselineMidLevelPolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size=512):
        super().__init__(
            PointNavBaselineMidLevelNet(
                observation_space=observation_space, hidden_size=hidden_size
            ),
            action_space.n,
        )


class PointNavBaselineMidLevelNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size):
        super().__init__()

        assert IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces
        self._n_input_goal = observation_space.spaces[IntegratedPointGoalGPSAndCompassSensor.cls_uuid].shape[0]

        self._hidden_size = hidden_size

        self.fc = FC()

        self.upresnet = UpResNet(
            layers=RESIDUAL_LAYERS_PER_BLOCK,
            channels=RESIDUAL_NEURON_CHANNEL,
            sizes=RESIDUAL_SIZE,
            strides=STRIDES
        )

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

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
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = observations[IntegratedPointGoalGPSAndCompassSensor.cls_uuid]
        x = [target_encoding]

        if not self.is_blind:
            observations = convert_rgb_obs_to_map(observations, self.fc, self.upresnet)
            perception_embed = self.visual_encoder(observations)  # [batch,256,256,3]
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states
