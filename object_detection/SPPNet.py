import torch
import torch.nn as nn
import torchvision
import math

from class_network.resnet import ResNet50


class SpatialPyramidPool2d(nn.Module):
    """
    3 layer, 4*4, 2*2, 1*1
    and concatenate
    """
    def __init__(self):
        super(SpatialPyramidPool2d, self).__init__()

    def forward(self, x):
        n, c, h, w = x.shape

        out = None

        for i in [1, 2, 4]:
            kernel_size = (math.ceil(h/i), math.ceil(w/i))
            stride = (math.ceil(h/i), math.ceil(w/i))
            maxpool = nn.MaxPool2d(kernel_size, stride)
            v = maxpool(x)
            v = v.view(n, -1)
            if out is None:
                out = v
            else:
                out = torch.cat((out, v), dim=1)

        return out


class SPPNet(nn.Module):
    """
    if the dim of feature map is 256, then the spp layer has a dimension of  256*(16+4+1)=5376
    """
    def __init__(self):
        super(SPPNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 256, 3, stride=2, padding=1, bias=False),  # size/2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

        )

        self.spp = SpatialPyramidPool2d()

    
    def forward(self, x):
        x = self.conv(x)    # any conv layer
        x = self.spp(x)     # spp layer
        
        return x


if __name__ == '__main__':
    model = SPPNet()
    x = torch.randn((2, 3, 224, 224))
    out = model(x)
    print(out.shape)
