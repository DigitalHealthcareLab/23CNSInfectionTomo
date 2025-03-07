import torch
import torch.nn as nn
import torch.nn.functional as F 

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4 # number of channels after a block is always four times what it was when it entered
        self.conv1 = nn.Conv3d(
            in_channels, 
            intermediate_channels,
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias = False)
        self.bn1 = nn.BatchNorm3d(intermediate_channels)
        self.conv2 = nn.Conv3d(
            intermediate_channels, 
            intermediate_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm3d(intermediate_channels)
        self.conv3 = nn.Conv3d(
            intermediate_channels, 
            intermediate_channels*self.expansion, 
            kernel_size=1, 
            stride=1, 
            padding=0,
            bias = False)
        self.bn3 = nn.BatchNorm3d(intermediate_channels*self.expansion)      
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        identity = x.clone()

        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x 



class ResNet(nn.Module):  # [3,4,6,3]-> list
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(
            image_channels, 
            64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias = False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(
            kernel_size=3, 
            stride=2, 
            padding=1)

        # ResNet layers 
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128,stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512*4, num_classes)
        self.dropout = nn.Dropout(0.7)
     #   self.BatchNorm3 = nn.BatchNorm3d(512)


    

    def forward(self, x):
      #  print(x.shape)
        x = x.view([x.size(0), 1, 64, 128, 128])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

        # ResNet layers
    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    intermediate_channels*4, 
                    kernel_size=1, 
                    stride=stride,
                    bias = False),
                nn.BatchNorm3d(intermediate_channels*4))

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride)) # layer that change the number of channels 
        self.in_channels = intermediate_channels*4 

        for i in range(num_residual_blocks - 1):     # -1은 바로 위에 layers.append 여기 부분으로 layer하나가 이미 있기 때문
            layers.append(block(self.in_channels, intermediate_channels))  # input : 256, output : 64, 64*4 (256) again 

        return nn.Sequential(*layers)




def ResNet50(img_channel = 1, num_classes = 1000):
    return ResNet(block, [3,4,6,3], img_channel, num_classes)

def ResNet101(img_channel = 1, num_classes = 1000):
    return ResNet(block, [3,4,23,3], img_channel, num_classes)

def ResNet152(img_channel = 1, num_classes = 1000):
    return ResNet(block, [3,8, 36,3], img_channel, num_classes)

def create_regression_resnet(device):
    return ResNet152(1,1).to(device)

def create_classification_resnet(device):
    return ResNet152(1,2).to(device)