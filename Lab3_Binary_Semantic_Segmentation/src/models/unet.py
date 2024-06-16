import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torch.nn.functional as F

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
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1_down = nn.Sequential(ConvTwice(3,64))
        self.conv2_down = nn.Sequential(
            nn.MaxPool2d((2,2),stride = 2),
            ConvTwice(64,128))
        self.conv3_down = nn.Sequential(
            nn.MaxPool2d((2,2),stride = 2),
            ConvTwice(128,256))
        self.conv4_down = nn.Sequential(
            nn.MaxPool2d((2,2),stride = 2),
            ConvTwice(256,512))
        self.conv1_up1 = nn.ConvTranspose2d(512, 256, (2,2),stride = 2)
        self.conv1_up2 = ConvTwice(512, 256)
        self.conv2_up1 = nn.ConvTranspose2d(256, 128, (2,2),stride = 2)
        self.conv2_up2 = ConvTwice(256, 128)
        self.conv3_up1 = nn.ConvTranspose2d(128, 64, (2,2),stride = 2)
        self.conv3_up2 = ConvTwice(128, 64)

        self.fullyconnect = nn.Conv2d(64, 1, (1,1))

    def forward(self, x):
        x1 = self.conv1_down(x)
        x2 = self.conv2_down(x1)
        x3 = self.conv3_down(x2)
        x4 = self.conv4_down(x3)
        
        x = self.conv1_up1(x4)
        x = self.conv1_up2(torch.cat([x, x3], dim=1))
        
        x = self.conv2_up1(x)
        x = self.conv2_up2(torch.cat([x, x2], dim=1))
        
        x = self.conv3_up1(x)
        x = self.conv3_up2(torch.cat([x, x1], dim=1))
        
        pred = self.fullyconnect(x)
        return pred

# model = Unet()
# summary(model, (16,3,256,256))