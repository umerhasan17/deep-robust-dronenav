""" 
networks/transform.py
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
from config.config import device, MAP_SIZE, RESIDUAL_SIZE, BATCHSIZE

"""
:param input_image_tensor:  (batch_size, 3, 256, 256)
:param representation_names: list
:return: concatted image tensor to pass into FCN  (batch_size, 8*len(representation_names), 16, 16)
"""
def egomotion_transform(input_map_tensor, dX): 
    x = dX[0]
    y = dX[1]
    t = dX[2]

    scale = (map_resolution[0]/map_size[0],map_resolution[1]/map_size[1]) # scale according to the map size parameter

    affine_transform_vector = -np.array([x*scale[0],y*scale[1],t]) # compute map coorrection

    return tensor_transform(input_map_tensor,affine_transform_vector) # call affine transform function

"""
:param input_image_tensor:  (batch_size, 3, 256, 256)
:param representation_names: list
:return: concatted image tensor to pass into FCN  (batch_size, 8*len(representation_names), 16, 16)
"""
def tensor_transform(input_map_tensor, transform): 

    # Construct a 2d rotation and transformation matrix (using scipy functions)
    r = np.array([R.from_euler('z',transform[2]).as_matrix()[0:2,0:2]])            # (2x2) rotation matrix for "angle" radiants, stored in a (1x2x2 tensor)
    x = np.array([[[transform[0]],[transform[1]]]])                                # (2x1) translation tensor, stored in a (1x2x1) tensor…
    T = torch.tensor(np.concatenate([r,x],axis=2), dtype=torch.float32)
    # unsqueezed_tensor = torch.unsqueeze(input_map_tensor,0)                        # (CxHxW) map tensor expanded to a (1xCxHxW) tensor

    # Compute grid transform tensor
    grid = F.affine_grid(T, input_map_tensor.size())          # a(1xHxWx2) full affine grid transform tensor
    # Resample input tensor according to grid transform, returns a rotated and translated tensor
    output_map_tensor = F.grid_sample(input_map_tensor, grid)      # (1xCxHxW) transformed map
    return output_map_tensor