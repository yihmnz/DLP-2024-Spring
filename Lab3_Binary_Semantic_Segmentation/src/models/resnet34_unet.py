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
            nn.BatchNorm2d(Output_Channel))
        
        self.ReLu = nn.ReLU(inplace = True)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        Output = self.conv1(x)
        Output = self.conv2(Output)

        # Apply shortcut
        Output += identity
        # Relu
        Output = self.ReLu(Output)
        del x, identity
        return Output
    
class ConvTwice(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ConvTwice = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, (3,3), padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace = True),
            nn.Conv2d(ch_out, ch_out, (3,3), padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace = True))

    def forward(self, x):
        return self.ConvTwice(x)
    
class U_ResNet34(nn.Module):
    def __init__(self):
        super(U_ResNet34, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (7,7), stride = 2, bias = False, padding = 3),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = (3,3), stride = 1, padding=1),      # [64, 64, 64]
        )
        self.input_channel_size = 64

        self.conv2_x = self.layer_iteration(3, 64, 1)
        self.conv3_x = self.layer_iteration(4, 128, 2)
        self.conv4_x = self.layer_iteration(6, 256, 2)
        self.conv5_x = self.layer_iteration(3, 512, 2)
        self.conv1_up1 = nn.ConvTranspose2d(512, 256, (2,2),stride = 2)
        self.conv1_up2 = ConvTwice(512, 256)
        self.conv2_up1 = nn.ConvTranspose2d(256, 128, (2,2),stride = 2)
        self.conv2_up2 = ConvTwice(256, 128)
        self.conv3_up1 = nn.ConvTranspose2d(128, 64, (2,2),stride = 2)
        self.conv3_up2 = ConvTwice(128, 64)

        self.fullyconnect = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (2,2),stride = 2),
            nn.Conv2d(64, 1, (1,1)))
        
    def layer_iteration(self, session, nodenum, stride = 1):
        downsample = None
        if stride != 1 or self.input_channel_size != nodenum:
            downsample = nn.Sequential(
            nn.Conv2d(self.input_channel_size, nodenum, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(nodenum))

        ML = []    
        ML.append(conv_lay(self.input_channel_size, nodenum, stride = stride, downsample=downsample))
        self.input_channel_size = nodenum
        
        for i in range(1, session):
            ML.append(conv_lay(self.input_channel_size, nodenum))
            
        return nn.Sequential(*ML)
    
    def forward(self, x):
        # print(f"Input shape: {x.size()}")
        x = self.conv1(x)
        # print(f"con1 shape: {x.size()}")
        x1 = self.conv2_x(x)
        # print(f"con2 shape: {x.size()}")
        x2 = self.conv3_x(x1)
        # print(f"con3 shape: {x.size()}")
        x3 = self.conv4_x(x2)
        # print(f"con4 shape: {x.size()}")
        x4 = self.conv5_x(x3)

        x = self.conv1_up1(x4)
        x = self.conv1_up2(torch.cat([x, x3], dim=1))
        
        x = self.conv2_up1(x)
        x = self.conv2_up2(torch.cat([x, x2], dim=1))
        
        x = self.conv3_up1(x)
        x = self.conv3_up2(torch.cat([x, x1], dim=1))
        
        pred = self.fullyconnect(x)
        del x, x1, x2, x3, x4
        return pred
 
# model = U_ResNet34()
# summary(model,(16,3,256,256))