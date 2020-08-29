"""
Why conv bias is all false?
"""

import torch
import torch.nn as nn
import torchvision


def resnet_conv1():
    return nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.conv3 = nn.Conv2d(out_chan, out_chan*expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan*expansion)
        self.relu = nn.ReLU(inplace=True)

        self.expansion = expansion
        self.downsampling = downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(in_chan, out_chan*expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_chan*expansion)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsampling:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.expansion = 4

        self.conv1 = resnet_conv1()
        self.layer1 = self._make_layer(in_chan=64, out_chan=64, block=3, stride=1)
        self.layer2 = self._make_layer(in_chan=256, out_chan=128, block=4, stride=2)
        self.layer3 = self._make_layer(in_chan=512, out_chan=256, block=6, stride=2)
        self.layer4 = self._make_layer(in_chan=1024, out_chan=512, block=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(2048, 1000)  # class number: 1000


    def _make_layer(self, in_chan, out_chan, block, stride):
        layers = []
        layers.append(Bottleneck(in_chan, out_chan, stride, downsampling=True))
        for _ in range(1, block):
            layers.append(Bottleneck(out_chan*self.expansion, out_chan))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    model = ResNet50()
    x = torch.randn((2, 3, 224, 224))
    rst = model(x)
    print(rst.shape)
