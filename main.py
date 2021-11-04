""" 
main.py
----------------------------------------------------------------------------
     Authors : Yongtao Wu, Umer Hasan, Titouan Renard
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
                       | variable name: map_update  -- (BATCHSIZE x 2 x M x M) tensor // encodes confidence and free space channels
                       |
                       |                                  egomotion
                       |                                  | variable name : dx -- (3x1) numpy array, (x,y,theta)
                       v                                  v
                  -----todo----                   -------done------
 TODO: Implement  |  combine  |   <-------------  |   transform   | <------------- previous_map -- (BATCHSIZE x 2 x M x M) tensor
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

from networks.encoder_mid_level import mid_level_representations      # mid_level wrapper class
from networks.decoder_residual import UpResNet                        # upsampling resnet
from networks.transform import egomotion_transform                    # upsampling resnet
from networks.update import update_map                                # upsampling resnet
from networks.fc import FC                                            # fully connected fc layer
from utils.storage import GlobalRolloutStorage
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from config import REPRESENTATION_NAMES,BATCHSIZE,DEVICE,RESIDUAL_LAYERS_PER_BLOCK,RESIDUAL_NEURON_CHANNEL,STRIDES,RESIDUAL_SIZE


def forward(image,egomotion,prev_map,verbose = False):
     #==========download image to debug==========

    img = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
    img = img.unsqueeze(0)                                                 # (1,3,256,256)
    activation = img.repeat(BATCHSIZE, 1, 1, 1)                            # (BATCHSIZE x 3 x 256 x 256) tensor


    #==========Mid level encoder==========
    print("Passing mid level encoder...")
    activation=mid_level_representations(activation,REPRESENTATION_NAMES)   # (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

    # ==========FC==========
    print("Passing fully connected layer...")
    fc=FC()
    activation = activation.view(BATCHSIZE,-1)                                  # flatten all dimensions except batch,
                                                                                # --> tensor of the form (BATCHSIZE x 2048*REPRESENTATION_NUMBER)
    activation = fc(activation)                                                 # pass through dense layer --> (BATCHSIZE x 2048*REPRESENTATION_NUMBER) tensor
    activation = activation.view(BATCHSIZE,8*len(REPRESENTATION_NAMES),16,16)   # after fully connected layer, # (BATCHSIZE x REPRESENTATION_NUMBER*2 x 16 x 16) tensor

    # ==========Deconv==========
    print("Passing residual decoder...")
    decoder = UpResNet(layers=RESIDUAL_LAYERS_PER_BLOCK,channels=RESIDUAL_NEURON_CHANNEL, sizes=RESIDUAL_SIZE, strides=STRIDES).to(DEVICE)
    map_update = decoder(activation) #upsample to map object

     # ==========Transform and Update==========
    print("Passing transform and update steps...")
    prev_map = egomotion_transform(prev_map,egomotion)
    new_map = update_map(map_update,prev_map)
    print("Done!")
    return new_map


if __name__ == '__main__':
     print("download image to debug...")
     subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)
     image = Image.open('test.png') #example image valuec
     # prev_map_raw = Image.open('Bedroom.jpg') #example prevmap value
     prev_map = TF.to_tensor(TF.resize(image, 256))[0:2] * 2 - 1
     prev_map = prev_map.unsqueeze(0)                                                 # (1,3,256,256)
     prev_map = prev_map.repeat(BATCHSIZE, 1, 1, 1)                            # (BATCHSIZE x 3 x 256 x 256) tensor


     egomotion = np.array([.1,0.,1.4]) #example egomotion value
     new_map=forward(image,egomotion,prev_map,verbose=True)



     # Starting environments
     from env import make_vec_envs
     from utils.arguments import get_args
     torch.set_num_threads(1)
     args = get_args()
     envs = make_vec_envs(args)
     obs, infos = envs.reset()
     # Storage
     g_rollouts = GlobalRolloutStorage(num_global_steps=128,
                                       num_processes=1,
                                       g_observation_space=(3,256,256),
                                       g_action_space=2,
                                       rec_state_size=1,
                                       extras_size=1).to(DEVICE)
     #input of ppo
     global_input = prev_map
     g_rollouts.obs[0].copy_(global_input)









