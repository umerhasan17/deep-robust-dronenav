# ================== Processing graph =====================
#
#                      image
#                        |
#                        | variable name: img
#                        v
#           ---------------------------
#           |mid_level_representations| directly using the visualpriors pip install
#           ---------------------------
#                        |
#                        | variable name: representation1,representation2
#                        v
#               ---------------------
#               |  feature_readout  |  directly using the visualpriors pip install
#               ---------------------
#                        |
#                        | variable name: pred1, pred2
#                        v
#                   velocities
#


import pdb

from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
from config import REPRESENTATION_NAMES,BATCHSIZE,DEVICE,RESIDUAL_LAYERS_PER_BLOCK,RESIDUAL_NEURON_LISTS,STRIDES
import subprocess
from fc import FC
from decoder_residual import ResNet
import torch

if __name__ == '__main__':
    #==========download image to debug==========
    print("download image to debug...")
    # subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)
    image = Image.open('test.png')

    
    feature_type = 'depth_euclidean'
    o_t1 = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
    o_t1 = o_t1.unsqueeze_(0)
    o_t2 = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
    o_t2 = o_t2.unsqueeze_(0)


    #==========Mid level encoder==========
    print("Passing mid level encoder...")
    representation1 = visualpriors.representation_transform(o_t1, 'depth_euclidean', device='cpu') # phi(o_t) in the diagram below
    representation2 = visualpriors.representation_transform(o_t2, 'normal', device='cpu') # phi(o_t) in the diagram below

    # Transform to normals feature and then visualize the readout
    pred1 = visualpriors.feature_readout(o_t1, 'depth_euclidean', device='cpu')
    pred2 = visualpriors.feature_readout(o_t2, 'normal', device='cpu')

    # Save it

    image.show()
    TF.to_pil_image(pred1[0] / 2. + 0.5).show()
    TF.to_pil_image(pred2[0] / 2. + 0.5).show()

    print("Done!")





