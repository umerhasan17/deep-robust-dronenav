""" 
networks/update.py
----------------------------------------------------------------------------
     Authors : Yongtao Wu, Umer Hasan Sayed, Titouan Renard
     Last Update : Octobre 2021

     applies the update function to the computed map

----------------------------------------------------------------------------
Processing graph:

      map_update (BATCHSIZE x 2 x 256 x 256)
          |
          v
    -------------
    |  combine  |   <----------- previous_map_transformed (BATCHSIZE x 2 x 256 x 256)
    -------------
          |
          v
    updated map (BATCHSIZE x 2 x 256 x 256)
  """

import pdb
# from torchvision import datasets, transforms
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
from config.config import device, MAP_SIZE, RESIDUAL_SIZE

"""
:param input_image_tensor:  (batch_size, 3, 256, 256)
:param representation_names: list
:return: concatted image tensor to pass into FCN  (batch_size, 8*len(representation_names), 16, 16)
"""


def update_map(map_update, previous_map_transformed):
    prev_confmap = previous_map_transformed[:, 0, :, :]
    update_confmap = map_update[:, 0, :, :]
    total_conf = torch.add(prev_confmap, update_confmap)

    return map_update
