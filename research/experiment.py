"""

    Example use of subfunction of full network

"""
import pdb

"""
    Example use of subfunction of full network
"""
import pdb

from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import visualpriors
# from config import REPRESENTATION_NAMES,BATCHSIZE,DEVICE,RESIDUAL_LAYERS_PER_BLOCK,RESIDUAL_NEURON_LISTS,STRIDES
import subprocess
from networks.fc import FC
from networks.decoder_residual import ResNet
from networks.transform import egomotion_transform
import matplotlib.pyplot as plt
import torch


def mid_level():
    # ==========download image to debug==========
    print("download image to debug...")
    # subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)
    image = Image.open('Bedroom.jpg')

    feature_type = 'depth_euclidean'
    o_t = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
    o_t = o_t.unsqueeze_(0)

    # ==========Mid level encoder==========
    print("Passing mid level encoder...")
    depth_representation = visualpriors.representation_transform(o_t, 'depth_euclidean',
                                                                 device='cpu')  # phi(o_t) in the diagram below
    normal_representation = visualpriors.representation_transform(o_t, 'normal',
                                                                  device='cpu')  # phi(o_t) in the diagram below

    # Transform to normals feature and then visualize the readout
    depth = visualpriors.feature_readout(o_t, 'depth_euclidean', device='cpu')
    normal = visualpriors.feature_readout(o_t, 'normal', device='cpu')

    print("Done!")
    return image, depth, normal[0]


def test_geom_transform(x=0, y=0, t=0):
    print("download image to debug...")
    # subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)
    image = Image.open('Bedroom.jpg')

    feature_type = 'depth_euclidean'
    o_t = TF.to_tensor(TF.resize(image, 256))
    o_t = torch.unsqueeze(o_t,0)
    print(o_t.size())
    d_X = np.array([x, y, t])  # advance 1[m] forward, 1[m] right, turn + 0.2 [rad]

    transformed_map = egomotion_transform(o_t, d_X)

    print("Done!")

    return image, transformed_map


if __name__ == '__main__':
    cogmap, transformed_map = test_geom_transform(0,.2,2)
    plt.imshow(transformed_map[0, :, :].permute(1, 2, 0))
    plt.show()
