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
from ResNet50 import ResNet50
from VGG19 import VGG19
from dataloader import BufferflyMothLoader

def evaluate(model, data, criterion, optimizer, device):
    # Evaluate on updated data
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in data:
        # Move data to the appropriate device (CPU or GPU)
        images, labels = images.to(device), labels.to(device)
        # get updated model prediction
        outputs = model(images)
        # Calc updated loss
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss/len(data), correct/total

def test(TestData, hist_res, savepath_Res, hist_vgg, savepath_VGG, device):
    Model1 = ResNet50().to(device)
    val_acc_array = np.array(hist_res["val_acc"])
    best_epoch1 = val_acc_array.argmax()
    test_model_path1 = os.path.join(savepath_Res, "{}-ep{}.pth".format("ResNet50", best_epoch1))
    checkpoint = torch.load(test_model_path1, map_location="cpu")
    Model1.load_state_dict(checkpoint["state_dict"])
    
    loss_fn = nn.CrossEntropyLoss()
    lr = 1*1e-3
    opt_fn = optim.Adam(Model1.parameters(), lr=lr, weight_decay=1e-4)
    testloss, testacc = evaluate(Model1, TestData, loss_fn, opt_fn, device)
    print("ResNet50 (argmax valid ep = {}): Train acc ={:.4f}, Valid acc ={:.4f}, Test acc={:.4f}".format(best_epoch1, hist_res["acc"][best_epoch1],hist_res["val_acc"][best_epoch1], testacc))
    
    del Model1
    Model2 = VGG19().to(device)
    val_acc_array = np.array(hist_vgg["val_acc"])
    best_epoch2 = val_acc_array.argmax()
    test_model_path2 = os.path.join(savepath_VGG, "{}-ep{}.pth".format("VGG19", best_epoch2))
    checkpoint = torch.load(test_model_path2, map_location="cpu")
    Model2.load_state_dict(checkpoint["state_dict"])
    opt_fn1 = optim.Adam(Model2.parameters(), lr=lr, weight_decay=1e-4)
    testloss1, testacc1 = evaluate(Model2, TestData, loss_fn, opt_fn1, device)
    del Model2
    print("VGG19 (argmax valid ep = {}): Train acc ={:.4f}, Valid acc ={:.4f}, Test acc={:.4f}".format(best_epoch2, hist_vgg["acc"][best_epoch2],hist_vgg["val_acc"][best_epoch2], testacc1))

def train(model, data, criterion, optimizer, device):
    # Start training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in data:
        # Move data to the appropriate device (CPU or GPU)
        images, labels = images.to(device), labels.to(device)

        # zero grad optimizer
        optimizer.zero_grad()

        # Forward get model prediction
        outputs = model(images)
        # Calc loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    
    # env setting
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True

    # model setting
    Model1 = ResNet50().to(device)
    
    # Batch size
    batch_size = 16
    # load data
    print('----Start Training Data load----')
    TrainData = BufferflyMothLoader(os.getcwd(), 'train')
    TrainData = DataLoader(TrainData, batch_size, shuffle=True, num_workers=4)
    print('----Start Valid Data load----')
    ValidData = BufferflyMothLoader(os.getcwd(), 'valid')
    ValidData = DataLoader(ValidData, batch_size, shuffle=True, num_workers=4)
    print('----Start Test Data load----')
    TestData = BufferflyMothLoader(os.getcwd(), 'test')
    TestData = DataLoader(TestData, batch_size, shuffle=False, num_workers=4)
    print('----Data loaded----')
    # training epochs
    num_epochs = 150
    # start training 
    loss_fn = nn.CrossEntropyLoss()
    lr = 1*1e-3
    opt_fn = optim.Adam(Model1.parameters(), lr=lr, weight_decay=1e-4)
    
    hist_ResNet = dict(
        loss=np.zeros((num_epochs, )), val_loss=np.zeros((num_epochs, )),
        acc=np.zeros((num_epochs, )), val_acc=np.zeros((num_epochs, ))
    )
    
    print('-----start_ResNet-----')
    savepath_Res = "ResNet50"
    os.makedirs(savepath_Res, exist_ok=True)
    for epoch in range(num_epochs):
        train(Model1, TrainData, loss_fn, opt_fn, device)
        loss, acc = evaluate(Model1, TrainData, loss_fn, opt_fn, device)
        val_loss, val_acc = evaluate(Model1, ValidData, loss_fn, opt_fn, device)
        print("Epoch {}: loss={:.4f}, acc={:.4f}, val_loss={:.4f}, val_acc={:.4f}".format(epoch, loss, acc, val_loss, val_acc))
        hist_ResNet["loss"][epoch] = loss
        hist_ResNet["acc"][epoch] = acc
        hist_ResNet["val_loss"][epoch] = val_loss
        hist_ResNet["val_acc"][epoch] = val_acc
        if True:
            checkpoint = {
                'epoch': epoch,
                'state_dict': Model1.state_dict(),
                'optimizer': opt_fn.state_dict(),
                'loss': loss,
                'acc': acc,
                'val_loss': val_loss,
                'val_acc': val_acc}
            torch.save(checkpoint, os.path.join(savepath_Res, f"ResNet50-ep{epoch}.pth"))
    np.savez(os.path.join(savepath_Res, 'savepath_Res_training_process.npz'), **hist_ResNet)
    print('-----File Saved-----')
    del Model1

    Model2 = VGG19().to(device)
    opt_fn_VGG = optim.Adam(Model2.parameters(), lr=lr, weight_decay=1e-4)
    print('-----start VGG-----')
    hist_VGG19 = dict(
        loss=np.zeros((num_epochs, )), val_loss=np.zeros((num_epochs, )),
        acc=np.zeros((num_epochs, )), val_acc=np.zeros((num_epochs, ))
    )
    savepath_VGG = "VGG19"
    os.makedirs(savepath_VGG, exist_ok=True)
    for epoch in range(num_epochs):
        train(Model2, TrainData, loss_fn, opt_fn_VGG, device)
        loss, acc = evaluate(Model2, TrainData, loss_fn, opt_fn_VGG, device)
        val_loss, val_acc = evaluate(Model2, ValidData, loss_fn, opt_fn_VGG, device)
        print("Epoch {}: loss={:.4f}, acc={:.4f}, val_loss={:.4f}, val_acc={:.4f}".format(epoch, loss, acc, val_loss, val_acc))
        hist_VGG19["loss"][epoch] = loss
        hist_VGG19["acc"][epoch] = acc
        hist_VGG19["val_loss"][epoch] = val_loss
        hist_VGG19["val_acc"][epoch] = val_acc
        if True:
            checkpoint = {
                'epoch': epoch,
                'state_dict': Model2.state_dict(),
                'optimizer': opt_fn_VGG.state_dict(),
                'loss': loss,
                'acc': acc,
                'val_loss': val_loss,
                'val_acc': val_acc}
            torch.save(checkpoint, os.path.join(savepath_VGG, f"VGG19-ep{epoch}.pth"))
    np.savez(os.path.join(savepath_VGG, 'savepath_VGG_training_process.npz'), **hist_VGG19)
    del Model2
    test(TestData, hist_ResNet, savepath_Res, hist_VGG19, savepath_VGG, device)
