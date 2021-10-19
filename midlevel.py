import pdb
import visualpriors
import torch
from config import DEVICE



def mid_level_representations(input_image_tensor, representation_names):
    """
    :param input_image_tensor:  (batch_size, 3, 256, 256)
    :param representation_names: list
    :return: concatted image tensor to pass into FCN  (batch_size, 8*len(representation_names), 256, 256)
    """
    representations=[]
    for name in representation_names:
        # (batch_size, 3, 256, 256) ——>(batch_size, 8, 16, 16)
        representations.append(visualpriors.representation_transform(input_image_tensor, name, device=DEVICE))
    return torch.cat(representations,dim=1)
