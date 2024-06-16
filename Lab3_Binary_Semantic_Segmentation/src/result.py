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
from inference import test, best_epoch_test, test_on_pretrained
from train import train
from oxford_pet import OxfordPetDataset, SimpleOxfordPetDataset, load_dataset
from models.unet import Unet
from models.resnet34_unet import U_ResNet34
from utils import training_curve, draw_pic, draw_pic_on_pretrained

if __name__ == "__main__":
    print('Are you going to test on DL_Lab3_xxxx_412551030_黃羿寧.pth?')
    q1 = int(input('(Yes = 1), (No = 0 please make sure you train the model in main.py in advance) '))
    if q1 == 0:
        c = int(input('Do you want to show the training curve? (Yes = 1), (No = 0) '))
        if c == 1:
            print('----Training curve----')
            current_path =  os.path.join(os.getcwd(),"..")
            savepath_unet = os.path.join('saved_models', 'Unet_0408')
            hist_unet = np.load(os.path.join(savepath_unet, 'training_process.npz'))
            savepath_unet = os.path.join('saved_models', 'ResUnet_0408')
            hist_resunet = np.load(os.path.join(savepath_unet, 'training_process.npz'))
            training_curve(hist_unet, hist_resunet)
            print('----File saved----')
            
        # Prepare machine
        device = "cuda" if torch.cuda.is_available() else "cpu"
        current_path =  os.getcwd()

        cc = int(input('Do you want to test the accuracy on the two model? (Yes = 1), (No = 0) '))
        if cc == 1:
            # Load data
            data_path = os.path.join(current_path,"dataset", "oxford-iiit-pet")
            batch_size = 1
            num_workers = 4
            TestData = load_dataset(data_path, "test" , batch_size, num_workers)
            print('{} images are loaded for testing'.format(len(TestData)*batch_size))

            for a in range(2):
                if a == 0:
                    Model1 = Unet().to(device)
                    savepath_unet = os.path.join(current_path, 'saved_models', 'Unet_0408')
                    print('----Unet testing----')
                else:
                    Model1 = U_ResNet34().to(device)
                    savepath_unet = os.path.join(current_path, 'saved_models', 'ResUnet_0408')
                    print('----ResUnet testing----')
                hist_unet = np.load(os.path.join(savepath_unet, 'training_process.npz'))
                test(TestData, Model1, hist_unet, savepath_unet, device)
        
        ccc = int(input('Do you want to show the prediction results on testing dataset? (Yes = 1), (No = 0) '))
        if ccc == 1:
            m = 0
            # Load data
            data_path = os.path.join(current_path,"dataset", "oxford-iiit-pet")
            batch_size = 1
            num_workers = 4
            TestData = load_dataset(data_path, "test" , batch_size, num_workers)
            print('{} images are loaded for testing'.format(len(TestData)*batch_size))

            while True:
                a = int(input('Which model r u going to test on? (Unet = 0), (ResUnet = 1), (quit = 2)'))
                if a == 0:
                    Model1 = Unet().to(device)
                    savepath_unet = os.path.join(current_path, 'saved_models', 'Unet_0408')
                    print('----Unet testing----')
                elif a == 1:
                    Model1 = U_ResNet34().to(device)
                    savepath_unet = os.path.join(current_path, 'saved_models', 'ResUnet_0408')
                    print('----ResUnet testing----')
                else:
                    break
                hist_unet = np.load(os.path.join(savepath_unet, 'training_process.npz'))
                best_epoch = best_epoch_test(hist_unet)
                # best_epoch = int(input('Ep = '))
                fig_num = int(input('Which pic r u going to show? Number (0-3668): '))
                while fig_num > 3668:
                    print('Out of range')
                    fig_num = int(input('Which pic r u going to show? Number (0-3668): '))
                draw_pic(TestData, Model1, best_epoch, savepath_unet, fig_num, m, device, a)
                m += 1
    else:
        # Prepare machine
        device = "cuda" if torch.cuda.is_available() else "cpu"
        current_path =  os.getcwd()

        cc = int(input('Do you want to test the accuracy on the two model? (Yes = 1), (No = 0) '))
        if cc == 1:
            # Load data
            data_path = os.path.join(current_path,"dataset", "oxford-iiit-pet")
            batch_size = 1
            num_workers = 4
            TestData = load_dataset(data_path, "test" , batch_size, num_workers)
            batch_size = 32
            TrainData = load_dataset(data_path, 'train', batch_size, num_workers)
            ValidData = load_dataset(data_path, "valid" , batch_size, num_workers)

            for a in range(2):
                if a == 0:
                    Model1 = Unet().to(device)
                    test_model_path1 = 'DL_Lab3_UNet_412551030_黃羿寧.pth'
                    print('----Unet testing----')
                else:
                    Model1 = U_ResNet34().to(device)
                    test_model_path1 = 'DL_Lab3_ResNet34_UNet_412551030_黃羿寧.pth'
                    print('----ResUnet testing----')
                test_on_pretrained(TrainData, ValidData, TestData, Model1, device, test_model_path1)

        ccc = int(input('Do you want to show the prediction results on testing dataset? (Yes = 1), (No = 0) '))
        if ccc == 1:
            m = 0
            # Load data
            data_path = os.path.join(current_path,"dataset", "oxford-iiit-pet")
            batch_size = 1
            num_workers = 4
            TestData = load_dataset(data_path, "test" , batch_size, num_workers)
            print('{} images are loaded for testing'.format(len(TestData)*batch_size))

            while True:
                a = int(input('Which model r u going to test on? (Unet = 0), (ResUnet = 1), (quit = 2)'))
                if a == 0:
                    Model1 = Unet().to(device)
                    test_model_path1 = 'DL_Lab3_UNet_412551030_黃羿寧.pth'
                    print('----Unet testing----')
                elif a == 1:
                    Model1 = U_ResNet34().to(device)
                    test_model_path1 = 'DL_Lab3_ResNet34_UNet_412551030_黃羿寧.pth'
                    print('----ResUnet testing----')
                else:
                    break
                fig_num = int(input('Which pic r u going to show? Number (0-3668): '))
                while fig_num > 3668:
                    print('Out of range')
                    fig_num = int(input('Which pic r u going to show? Number (0-3668): '))
                draw_pic_on_pretrained(TestData, Model1, test_model_path1, fig_num, m, device, a)
                m += 1