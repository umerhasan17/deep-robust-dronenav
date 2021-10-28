# =============== Processing graph =====================
#
#                      image
#                        |
#                        | variable name: img
#                        v
#           ---------------------------
#           |mid_level_representations| directly using the visualpriors pip install
#           ---------------------------
#                        |
#                        | variable name: mid_level
#                        v
#               ---------------------
#               |  fc ->  decoder   |  content : two functions, fully-connected layer "fc" and decoder resnet "decoder"
#               ---------------------
#                        |
#                        | variable name: map_update
#                        v
#                   -------------                   -----------------
#  TODO: Implement  |  combine  |   <-------------  |   transform   | <------------- previous_map
#                   -------------                   -----------------
#                        |
#                        | variable name: map
#                        v
#                 ---------------
#  TODO: Implement|   policy    |
#                 ---------------
#                        |
#                        |
#                        v
#                   velocities
#
 

import pdb

import subprocess

from networks.encoder_mid_level import mid_level_representations
from networks.decoder_residual import ResNet
from networks.fc import FC

from PIL import Image

import torch
import torchvision.transforms.functional as TF

from config import REPRESENTATION_NAMES,BATCHSIZE,DEVICE,RESIDUAL_LAYERS_PER_BLOCK,RESIDUAL_NEURON_LISTS,STRIDES



if __name__ == '__main__':
    #==========download image to debug==========
    print("download image to debug...")
    subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)
    image = Image.open('test.png')
    img = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
    img = img.unsqueeze(0) #(1,3,256,256)
    img = img.repeat(BATCHSIZE, 1, 1, 1)


    #==========Mid level encoder==========
    print("Passing mid level encoder...")
    mid_level=mid_level_representations(img,REPRESENTATION_NAMES)

    # ==========FC==========
    print("Passing fully connected layer...")
    fc=FC()
    mid_level = mid_level.view(BATCHSIZE,-1) #x = torch.flatten(x, start_dim=1)  # flatten all dimensions except batc
    mid_level = fc(mid_level)
    mid_level = mid_level.view(BATCHSIZE,8*len(REPRESENTATION_NAMES),16,16)

    # ==========Deconv==========
    print("Passing residual decoder...")
    decoder = ResNet(layers=RESIDUAL_LAYERS_PER_BLOCK,channels=RESIDUAL_NEURON_LISTS,strides=STRIDES).to(DEVICE)
    map_update = decoder(mid_level)

    print("Done!")





