import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import REPRESENTATION_NAMES

"""
BasicBlock: simple convolutional module (can downsample)
    Constructor arguments : 
        - inplanes  : number of input channels
        - planes    : number of output channels
        - stride    : (default = 1)
        - downsample: (default = None)
"""
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:

            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,  layers, channels,strides,block=BasicBlock):
        super().__init__()
        self.inplanes = 8*len(REPRESENTATION_NAMES)

        self.layer0 = self._make_layer(block, channels[0], layers[0], strides[0])
        self.layer1 = self._make_layer(block, channels[1], layers[1], strides[1])
        self.layer2 = self._make_layer(block, channels[2], layers[2], strides[2])
        print("built net")


    def _make_layer(self, block, planes, number_of_layers, stride):

        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(self.inplanes, planes,stride, downsample))

        self.inplanes = planes
        for _ in range(1, number_of_layers):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


"""
UpSampleBlock: simple deconvolutional module (performs upsampling)
    Constructor arguments : 
        - inplanes  : number of input channels
        - planes    : number of output channels
        - size      : (default = (256,256))
        - stride    : (default = 1)
"""
class UpSampleBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, size = (256,256)):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.size = size
        self.planes = planes

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False) # Upsample with bilinear interpolation

        # identity = torch.split(x,self.planes,dim = 0)
        out = self.conv1(x)
        identity = out      # Barely a residual connection if you ask me :(
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

"""
UpSampleBlock: simple upsampling resnet, used to regress a map from the dense representation embedding coming out of fc
    Constructor arguments : 
        - layers        :  
        - channels      :
        - strides       :
        - block         :
"""
class UpResNet(nn.Module):
    def __init__(self,  layers, inchannels, outchannels,strides,block=UpSampleBlock):
        super().__init__()
        self.inplanes = 8*len(REPRESENTATION_NAMES)

        self.layer0 = self._make_layer(block, inchannels[0], outchannels[0], layers[0], strides[0],(32,32))
        self.layer1 = self._make_layer(block, inchannels[1], outchannels[1], layers[1], strides[1],(64,64))
        self.layer2 = self._make_layer(block, inchannels[2], outchannels[2], layers[2], strides[2],(128,128))
        self.layer3 = self._make_layer(block, inchannels[3], outchannels[3], layers[3], strides[2],(256,256))
        print("built net")


    def _make_layer(self, block, inplanes, outplanes, number_of_layers, stride, size):

        layers = []
        layers.append(block(self.inplanes, outplanes,stride, size))

        self.inplanes = outplanes
        for _ in range(1, number_of_layers):
            layers.append(block(self.inplanes, self.inplanes,stride, size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x