import pdb

from PIL import Image
import torchvision.transforms.functional as TF
from encoder_mid_level import mid_level_representations
from config import REPRESENTATION_NAMES,BATCHSIZE,DEVICE,RESIDUAL_LAYERS_PER_BLOCK,RESIDUAL_NEURON_LISTS,STRIDES
import subprocess
from fc import FC
from decoder_residual import ResNet
import torch

if __name__ == '__main__':
    #==========download image to debug==========
    print("download image to debug...")
    subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)
    image = Image.open('test.png')
    x = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
    x = x.unsqueeze(0) #(1,3,256,256)
    x = x.repeat(BATCHSIZE, 1, 1, 1)


    #==========Mid level encoder==========
    print("Passing mid level encoder...")
    x=mid_level_representations(x,REPRESENTATION_NAMES)

    # ==========FC==========
    print("Passing fully connected layer...")
    fc=FC()
    x = x.view(BATCHSIZE,-1) #x = torch.flatten(x, start_dim=1)  # flatten all dimensions except batc
    x = fc(x)
    x = x.view(BATCHSIZE,8*len(REPRESENTATION_NAMES),16,16)

    # ==========Deconv==========
    print("Passing residual decoder...")
    decoder = ResNet(layers=RESIDUAL_LAYERS_PER_BLOCK,channels=RESIDUAL_NEURON_LISTS,strides=STRIDES).to(DEVICE)
    x = decoder(x)

    print("Done!")





