""" 
main.py
----------------------------------------------------------------------------
     Authors : Yongtao Wu, Umer Hasan Sayed, Titouan Renard
     Last Update : Octobre 2021

     Main routine for our deep cognitive drone mapping agent,
     this file implements the architecture of our networks

----------------------------------------------------------------------------
Processing graph:
     


                     image
                       |
                       | variable name: img -- (BATCHSIZE x 3 x 256 x 256) tensor
                       v
          ------------done-----------
          |mid_level_representations| directly using the visualpriors pip install
          ---------------------------
                       |
                       | variable name: mid_level -- (BATCHSIZE x 3 x 256 x 256) tensor
                       v
          -----------done--------------
          |                           | 
          |  fc ->  UpSampleResNet    |  content : two functions, fully-connected layer "fc" and decoder resnet "decoder"
          |                           | 
          |                           |  (BATCHSIZE x 2048*REPRESENTATION_NUMBER) -(fc)->  (BATCHSIZE x 2*REPRESENTATION_NUMBER x 16 x 16) 
          -----------------------------  
                       |
                       | variable name: map_update  -- (BATCHSIZE x 2 x 256 x 256) tensor // encodes confidence and free space channels
                       v
                  -----todo----                   -------done------
 TODO: Implement  |  combine  |   <-------------  |   transform   | <------------- previous_map -- (BATCHSIZE x 2[c,f] x 256 x 256) tensor
                  -------------                   -----------------
                       |
                       | variable name: map 
                       v
                ------todo-----
 TODO: Implement|   policy    |
                ---------------
                       |
                       |
                       v
                  velocities

  """

import pdb
import subprocess

from networks.encoder_mid_level import mid_level_representations
from networks.decoder_residual import UpResNet
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
    img = img.unsqueeze(0)                                          # (1,3,256,256)
    activation = img.repeat(BATCHSIZE, 1, 1, 1)                            # (BATCHSIZE x 3 x 256 x 256) tensor


    #==========Mid level encoder==========
    print("Passing mid level encoder...")
    activation=mid_level_representations(activation,REPRESENTATION_NAMES)   # (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

    # ==========FC==========
    print("Passing fully connected layer...")
    fc=FC()
    activation = activation.view(BATCHSIZE,-1)                                # flatten all dimensions except batch, 
                                                                            # --> tensor of the form (BATCHSIZE x 2048*REPRESENTATION_NUMBER)
    activation = fc(activation)                                               # pass through dense layer --> (BATCHSIZE x 2048*REPRESENTATION_NUMBER) tensor
    activation = activation.view(BATCHSIZE,8*len(REPRESENTATION_NAMES),16,16) # after fully connected layer, # (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

    # ==========Deconv==========
    print("Passing residual decoder...")
    decoder = UpResNet(layers=RESIDUAL_LAYERS_PER_BLOCK,channels=RESIDUAL_NEURON_LISTS,strides=STRIDES).to(DEVICE)
    map_update = decoder(activation)

    print("Done!")





