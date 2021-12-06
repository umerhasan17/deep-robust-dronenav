""" 
mapper/transform.py
----------------------------------------------------------------------------
     Authors : Yongtao Wu, Umer Hasan Sayed, Titouan Renard
     Last Update : Octobre 2021

     Geometric transform functions for out autonomous agent

----------------------------------------------------------------------------
Processing graph:
                                                dX : (3x1) numpy array, (x,y,theta) -- represents robot's motion
                                                |
                                                v
                                         -----------------
    previous_map_transformed  <- ------  |   transform   | <----- previous_map -- (BATCHSIZE x 2 x 256 x 256) tensor
    -> (BATCHSIZE x 2 x 256 x 256)       -----------------
"""

import pdb
# from torchvision import datasets, transforms
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
from matplotlib.transforms import Affine2D
from config.config import device, MAP_SIZE, MAP_DIMENSIONS, RESIDUAL_SIZE, BATCHSIZE

"""
:param input_image_tensor:  (batch_size, 3, 256, 256)
:param representation_names: list
:return: concatted image tensor to pass into FCN  (batch_size, 8*len(representation_names), 16, 16)
"""
def egomotion_transform(input_map_tensor, dX): 
    x = dX[0] * (MAP_DIMENSIONS[1]/MAP_SIZE[0])
    y = dX[1] * (MAP_DIMENSIONS[2]/MAP_SIZE[1])
    t = dX[2]

    affine_transform_vector = -np.array([x,y,t]) # compute map coorrection

    return tensor_transform(input_map_tensor,affine_transform_vector) # call affine transform function

"""
:param input_image_tensor:  (batch_size, 3, 256, 256)
:param representation_names: list
:return: concatted image tensor to pass into FCN  (batch_size, 8*len(representation_names), 16, 16)
"""
def tensor_transform(input_map_tensor, transform): 

    # Construct a 2d rotation and transformation matrix (using scipy functions)
    T = torch.tensor((Affine2D().rotate_around(width//2,height//2,transform[2]) + Affine2D().translate(tx = transform[0], ty = transform[1])).get_matrix())
    # unsqueezed_tensor = torch.unsqueeze(input_map_tensor,0)                        # (CxHxW) map tensor expanded to a (1xCxHxW) tensor

    # Compute grid transform tensor
    grid = F.affine_grid(T, input_map_tensor.size())          # a(1xHxWx2) full affine grid transform tensor
    # Resample input tensor according to grid transform, returns a rotated and translated tensor
    output_map_tensor = F.grid_sample(input_map_tensor, grid)      # (1xCxHxW) transformed map
    return output_map_tensor