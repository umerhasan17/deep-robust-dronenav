from collections import deque

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


# TODO this will be image batch
def create_map(image):
    """ Returns map for RL to train on """

    # ==========download image to debug==========

    image = torch.transpose(image, 1, 3)
    print(image.shape)
    activation = image
    # ==========Mid level encoder==========
    print("Passing mid level encoder...")
    activation = mid_level_representations(activation,
                                           REPRESENTATION_NAMES)  #  (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor
    # ==========FC==========
    print("Passing fully connected layer...")
    fc = FC().to(device)
    activation = activation.view(activation.shape[0], 1, -1)  # flatten all dimensions except batch,
    # --> tensor of the form (BATCHSIZE x 2048*REPRESENTATION_NUMBER)
    activation = fc(activation)  # pass through dense layer --> (BATCHSIZE x 2048*REPRESENTATION_NUMBER) tensor
    activation = activation.view(activation.shape[0], 8 * len(REPRESENTATION_NAMES), 16,
                                 16)  # after fully connected layer, # (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

    # ==========Deconv==========
    print("Passing residual decoder...")
    decoder = UpResNet(layers=RESIDUAL_LAYERS_PER_BLOCK, channels=RESIDUAL_NEURON_CHANNEL, sizes=RESIDUAL_SIZE,
                       strides=STRIDES).to(device)
    map_update = decoder(activation)  # upsample to map object

    output = torch.transpose(map_update, 1, 3)
    print(f'Returning output tensor with shape: {output.shape}')

    return output
