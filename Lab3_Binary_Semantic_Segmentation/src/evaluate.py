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
import gc

def dice_score(output, label):
    output = torch.sigmoid(output) 
    preds = output > 0.5
    label = label > 0.5
    intersection = (preds & label).float().sum()
    total_size = preds.float().sum() + label.float().sum()
    
    dice = (2. * intersection+1e-6) / (total_size+1e-6) 
    return dice.mean()

def evaluate(model, data, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    for images_dist in data:
        # Move data to the appropriate device (CPU or GPU)
        images = images_dist['image'].to(device)
        labels = images_dist['mask'].to(device)
        # get updated model prediction
        outputs = model(images)
        labels = torch.squeeze(labels, 1)
        outputs = torch.squeeze(outputs, 1)
        # Calc updated loss
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        dice_score_batch = dice_score(outputs, labels)
        dice_scores.append(dice_score_batch.item())

        del images, labels, outputs, loss, dice_score_batch
        torch.cuda.empty_cache()
    
    average_dice_score = sum(dice_scores) / len(dice_scores)
    return running_loss/len(data), average_dice_score