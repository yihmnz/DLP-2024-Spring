import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import numpy as np

def train(model, data, criterion, optimizer, device):
    # implement the training function here
    # Start training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images_dist in data:
        # Move data to the appropriate device (CPU or GPU)
        # sample["image"] = 
        # sample["mask"] = np.expand_dims(mask, 0)
        images = images_dist['image'].to(device)
        labels = images_dist['mask'].to(device)
        # zero grad optimizer
        optimizer.zero_grad()

        # Forward get model prediction
        outputs = model(images)
        # Calc loss
        # print(outputs.shape)
        labels = torch.squeeze(labels, 1)
        outputs = torch.squeeze(outputs, 1)

        loss = criterion(outputs, labels)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        # print(loss.item())
        del images, labels, outputs, loss 
        torch.cuda.empty_cache()