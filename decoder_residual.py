import pdb
import torch.nn as nn
from config import REPRESENTATION_NAMES

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