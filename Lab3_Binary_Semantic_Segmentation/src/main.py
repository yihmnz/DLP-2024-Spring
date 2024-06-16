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
from evaluate import evaluate
from inference import test
from train import train
from oxford_pet import OxfordPetDataset, SimpleOxfordPetDataset, load_dataset
from models.unet import Unet
from models.resnet34_unet import U_ResNet34

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True

    # model setting
    a = int(input('Which model r u going to train? (Unet = 0), (ResUnet = 1)'))
    if a == 0:
        Model1 = Unet().to(device)
    else:
        Model1 = U_ResNet34().to(device)
    # load data
    print('----Start Data load----')
    current_path =  os.getcwd()
    data_path = os.path.join(current_path,"dataset", "oxford-iiit-pet")
    batch_size = 32
    num_workers = 4
    TrainData = load_dataset(data_path, 'train', batch_size, num_workers)
    print('{} images are loaded for training'.format(len(TrainData)*batch_size))
    ValidData = load_dataset(data_path, "valid" , batch_size, num_workers)
    print('{} images are loaded for validation'.format(len(ValidData)*batch_size))
    TestData = load_dataset(data_path, "test" , 1, num_workers)
    print('{} images are loaded for testing'.format(len(TestData)))

    # training epochs
    num_epochs = 200
    # start training 
    loss_fn = nn.CrossEntropyLoss()
    lr = 1*1e-4
    opt_fn = optim.Adam(Model1.parameters(), lr=lr, weight_decay=1e-4)
    
    hist_UNet = dict(
        loss=np.zeros((num_epochs, )), val_loss=np.zeros((num_epochs, )),
        acc=np.zeros((num_epochs, )), val_acc=np.zeros((num_epochs, ))
    )
    
    print('-----start_training-----')
    if a == 0:
        savepath_unet = os.path.join('saved_models', 'Unet_0408')
        os.makedirs(savepath_unet, exist_ok=True)
    else:
        savepath_unet = os.path.join('saved_models', 'ResUnet_0408')
        os.makedirs(savepath_unet, exist_ok=True)
        
    for epoch in range(num_epochs):
        # train(model, data, criterion, optimizer, device):
        train(Model1, TrainData, loss_fn, opt_fn, device)
        # evaluate(model, data, criterion, device):
        loss, acc = evaluate(Model1, TrainData, loss_fn, device)
        val_loss, val_acc = evaluate(Model1, ValidData, loss_fn, device)
        print("Epoch {}: loss={:.4f}, acc={:.4f}, val_loss={:.4f}, val_acc={:.4f}".format(epoch, loss, acc, val_loss, val_acc))
        hist_UNet["loss"][epoch] = loss
        hist_UNet["acc"][epoch] = acc
        hist_UNet["val_loss"][epoch] = val_loss
        hist_UNet["val_acc"][epoch] = val_acc
        if True:
            checkpoint = {
                'epoch': epoch,
                'state_dict': Model1.state_dict(),
                'optimizer': opt_fn.state_dict(),
                'loss': loss,
                'acc': acc,
                'val_loss': val_loss,
                'val_acc': val_acc}
            torch.save(checkpoint, os.path.join(savepath_unet, f"Model-ep{epoch}.pth"))
    np.savez(os.path.join(savepath_unet, 'training_process.npz'), **hist_UNet)
    print('-----File Saved-----')
