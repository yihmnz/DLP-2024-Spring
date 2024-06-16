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

def best_epoch_test(hist_res):
    val_acc_array = np.array(hist_res["val_acc"])
    best_epoch = val_acc_array.argmax()
    return best_epoch

def test(TestData, Model1, hist_res, savepath_Res, device):
    best_epoch1 = best_epoch_test(hist_res)
    test_model_path1 = os.path.join(savepath_Res, "{}-ep{}.pth".format("Model", best_epoch1))
    checkpoint = torch.load(test_model_path1, map_location="cpu")
    Model1.load_state_dict(checkpoint["state_dict"])
    loss_fn = nn.CrossEntropyLoss()
    # evaluate(model, data, criterion, device)
    _, testacc = evaluate(Model1, TestData, loss_fn, device)
    print("(argmax valid ep = {}): Train dice ={:.4f}, Valid dice ={:.4f}, Test dice={:.4f}".format(best_epoch1, hist_res["acc"][best_epoch1],hist_res["val_acc"][best_epoch1], testacc))

def test_on_pretrained(TrainData, ValidData, TestData, Model1, device, test_model_path1):
    checkpoint = torch.load(test_model_path1, map_location="cpu")
    Model1.load_state_dict(checkpoint["state_dict"])
    loss_fn = nn.CrossEntropyLoss()
    # evaluate(model, data, criterion, device)
    _, trainacc = evaluate(Model1, TrainData, loss_fn, device)
    _, validacc = evaluate(Model1, ValidData, loss_fn, device)
    _, testacc = evaluate(Model1, TestData, loss_fn, device)
    print("Train dice ={:.4f}, Valid dice ={:.4f}, Test dice={:.4f}".format(trainacc, validacc, testacc))

