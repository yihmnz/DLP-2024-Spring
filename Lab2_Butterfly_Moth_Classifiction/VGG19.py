import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torch.nn.functional as F

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d((2, 2), stride = 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d((2, 2), stride = 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d((2, 2), stride = 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d((2, 2), stride = 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, (3,3), stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d((2, 2), stride = 2)
        )
        self.fullyconnect = nn.Sequential(
            nn.Linear(512*(7*7), 4096), # 224*224/(2^5) = 7*7
            nn.ReLU(inplace = True),
            nn.Dropout(0.5), # 
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 100)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1) #Flatten 
        x = self.fullyconnect(x)
        return x

# model = VGG19()
# print(summary(model, input_size=(1, 3, 224, 224)))