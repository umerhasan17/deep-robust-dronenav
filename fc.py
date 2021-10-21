import pdb

import torch
import torch.nn as nn
from config import FC_NEURON_LISTS

# each fc for each representations ?
# use cnn and then fc ?
# (cognitive_mapping_and_planning-master\tfcode\cmp.py # " Add a layer to reduce dimensions for a fc layer.")
class FC(nn.Module):
    """
    3 layer perceptron class nn.Sequential
    output dimensions:
    """
    def __init__(self):
        super().__init__()
        self.neuron_lists=FC_NEURON_LISTS
        fc_layers = []
        for i in range(len(self.neuron_lists)-1):
            fc_layers.append(nn.Linear(self.neuron_lists[i],self.neuron_lists[i+1]))
            fc_layers.append(nn.BatchNorm1d(self.neuron_lists[i+1]))
            fc_layers.append(nn.ReLU())
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.fc(x)
        return  x





# def deconv(x, is_training, wt_decay, neurons, strides, layers_per_block,
#             kernel_size, conv_fn, name, offset=0):
#   """Generates a up sampling network with residual connections.
#   """
#   batch_norm_param = {'center': True, 'scale': True,
#                       'activation_fn': tf.nn.relu,
#                       'is_training': is_training}
#   outs = []
#   for i, (neuron, stride) in enumerate(zip(neurons, strides)):
#     for s in range(layers_per_block):
#       scope = '{:s}_{:d}_{:d}'.format(name, i+1+offset,s+1)
#       x = custom_residual_block(x, neuron, kernel_size, stride, scope,
#                                 is_training, wt_decay, use_residual=True,
#                                 residual_stride_conv=True, conv_fn=conv_fn,
#                                 batch_norm_param=batch_norm_param)
#       stride = 1
#     outs.append((x,True))
#   return x, outs

class Deconv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass





