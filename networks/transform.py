import pdb
# from torchvision import datasets, transforms
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
from config import DEVICE

def egomotion_transform(input_map_tensor, egomotion):
    """
    :param input_image_tensor:  (batch_size, 3, 256, 256)
    :param representation_names: list
    :return: concatted image tensor to pass into FCN  (batch_size, 8*len(representation_names), 16, 16)
    """
    
    fwd = egomotion[0]
    left = egomotion[1]
    angle = egomotion[2]

    # Construct a 2d rotation and transformation matrix (using scipy functions)
    r = np.array([R.from_euler('z',angle).as_matrix()[0:2,0:2]])            # (2x2) rotation matrix for "angle" radiants, stored in a (1x2x2 tensor)
    x = np.array([[[fwd],[left]]])                                          # (2x1) translation tensor, stored in a (1x2x1) tensor…
    T = torch.tensor(np.concatenate([r,x],axis=2), dtype=torch.float32)     # (1x2x3) geometric tensor

    unsqueezed_tensor = torch.unsqueeze(input_map_tensor,0)         # (CxHxW) map tensor expanded to a (1xCxHxW) tensor
    grid = F.affine_grid(T, unsqueezed_tensor.size(),align_corners = False)               # (1xHxWx2) affine grid transform tensor

    output_map_tensor = F.grid_sample(unsqueezed_tensor, grid)      # (1xCxHxW) transformed map
    return output_map_tensor

