from os import getcwd
from os.path import dirname, realpath, join, expanduser, normpath,isdir,split
import sys
PACKAGE_PARENT = '../'
SCRIPT_DIR = dirname(realpath(join(getcwd(), expanduser(__file__))))
sys.path.append(normpath(join(SCRIPT_DIR, PACKAGE_PARENT)))

from mapper.mid_level.encoder import mid_level_representations  # mid_level wrapper class
from mapper.mid_level.decoder import UpResNet  # upsampling resnet
from mapper.transform import egomotion_transform  # upsampling resnet
from mapper.update import update_map  # upsampling resnet
from mapper.mid_level.fc import FC  # fully connected fc layer
import torch
import torchvision.transforms.functional as TF
from config.config import REPRESENTATION_NAMES, BATCHSIZE, device, RESIDUAL_LAYERS_PER_BLOCK, RESIDUAL_NEURON_CHANNEL, \
    STRIDES, \
    RESIDUAL_SIZE, IMG_DIMENSIONS
import torch.nn as nn
import torch.nn.functional as F

class FC_UpResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = FC()
        self.decoder = UpResNet(layers=RESIDUAL_LAYERS_PER_BLOCK, channels=RESIDUAL_NEURON_CHANNEL, sizes=RESIDUAL_SIZE,
                           strides=STRIDES)
    def forward(self, activation):
        # ==========FC==========
        print("Passing fully connected layer...")
        activation = activation.view(activation.shape[0], 1, -1)  # flatten all dimensions except batch,
        # --> tensor of the form (BATCHSIZE x 2048*REPRESENTATION_NUMBER)
        activation = self.fc(activation)  # pass through dense layer --> (BATCHSIZE x 2048*REPRESENTATION_NUMBER) tensor
        activation = activation.view(activation.shape[0], 8 * len(REPRESENTATION_NAMES), 16,
                                     16)  # after fully connected layer, # (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

        # ==========Deconv==========
        print("Passing residual decoder...")
        map_update = self.decoder(activation)  # upsample to map object
        return map_update



# TODO this will be image batch
def create_map(image):
    """ Returns map for RL to train on """

    # ==========download image to debug==========

    image = torch.transpose(image, 1, 3)
    print(image.shape)
    activation = image
    # ==========Mid level encoder==========
    print("Passing mid level encoder...")
    activation = mid_level_representations(activation,REPRESENTATION_NAMES)  #  (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

    map_update = FC_UpResNet(activation)

    output = torch.transpose(map_update, 1, 3)
    print(f'Returning output tensor with shape: {output.shape}')

    return output




