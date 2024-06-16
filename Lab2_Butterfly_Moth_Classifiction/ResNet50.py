import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torch.nn.functional as F

class conv_lay(nn.Module):
    """Left block + shortcut template"""
    def __init__(self, Input_Channel, Output_Channel, stride = 1, downsample=None):
        super(conv_lay, self).__init__()
        """torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, bias=True, 
            padding_mode='zeros', device=None, dtype=None)"""
        self.conv1 = nn.Sequential(
            nn.Conv2d(Input_Channel, Output_Channel, (1,1), 1, bias=False),
            nn.BatchNorm2d(Output_Channel),
            nn.ReLU(inplace = True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(Output_Channel, Output_Channel, (3,3), stride = stride, bias=False, padding=1),
            nn.BatchNorm2d(Output_Channel),
            nn.ReLU(inplace = True))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(Output_Channel, Output_Channel*4, (1,1), 1, bias=False),
            nn.BatchNorm2d(Output_Channel*4))
        
        self.ReLu = nn.ReLU(inplace = True)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        Output = self.conv1(x)
        Output = self.conv2(Output)
        Output = self.conv3(Output)

        # Apply shortcut
        Output += identity
        # Relu
        Output = self.ReLu(Output)

        return Output

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (7,7), stride = 2, bias = False, padding = 3),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = (3,3), stride = 2, padding=1),      # [64, 64, 64]
        )
        self.input_channel_size = 64

        self.conv2_x = self.layer_iteration(3, 64, 1)
        self.conv3_x = self.layer_iteration(4, 128, 2)
        self.conv4_x = self.layer_iteration(6, 256, 2)
        self.conv5_x = self.layer_iteration(3, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # Softmax Classification
        self.fullyconnect = nn.Linear(2048, 100)
        
    def layer_iteration(self, session, nodenum, stride = 1):
        downsample = None
        if stride != 1 or self.input_channel_size != nodenum*4:
            downsample = nn.Sequential(
            nn.Conv2d(self.input_channel_size, nodenum*4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(nodenum*4))

        ML = []    
        ML.append(conv_lay(self.input_channel_size, nodenum, stride = stride, downsample=downsample))
        self.input_channel_size = nodenum*4
        
        for i in range(1, session):
            ML.append(conv_lay(self.input_channel_size, nodenum))
            
        return nn.Sequential(*ML)
    
    def forward(self, x):
        # print(f"Input shape: {x.size()}")
        x = self.conv1(x)
        # print(f"con1 shape: {x.size()}")
        x = self.conv2_x(x)
        # print(f"con2 shape: {x.size()}")
        x = self.conv3_x(x)
        # print(f"con3 shape: {x.size()}")
        x = self.conv4_x(x)
        # print(f"con4 shape: {x.size()}")
        x = self.conv5_x(x)
        # print(f"con5 shape: {x.size()}")
        x = self.avgpool(x)
        # print(f"con6 shape: {x.size()}")
        x = x.reshape(x.shape[0], -1)
        x = self.fullyconnect(x)
        return x
    
# model = ResNet50()
# print(summary(model, input_size=(16, 3, 224, 224)))