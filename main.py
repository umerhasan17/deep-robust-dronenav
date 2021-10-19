from PIL import Image
import torchvision.transforms.functional as TF
from midlevel import mid_level_representations
from config import REPRESENTATION_NAMES,BATCHSIZE
import subprocess
from mappernn import FC
import torch

if __name__ == '__main__':
    # # Download a test image
    subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)
    # Load image and rescale/resize to [-1,1] and 3x256x256
    image = Image.open('test.png')
    x = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
    x = x.unsqueeze(0) #(1,3,256,256)
    x = x.repeat(BATCHSIZE, 1, 1, 1)
    x=mid_level_representations(x,REPRESENTATION_NAMES)

    fc=FC()
    x = fc(x)