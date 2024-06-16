import numpy as np
import scipy.io as sio
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import time
import h5py
import torch.nn.functional as F
from CRNN import CRNN 
from tqdm import tqdm
from CrossDomainFeatureFusion import CombinedModel
# from CDFF import CombinedModel
# Model settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
epochs = 100
lr = 0.0001
adj = torch.eye(32).to(device)

model = CombinedModel(Nt=1, hidden_dim=160, lstm_layers=2, 
                      gcn_nfeat=5, gcn_nhid=32, gcn_nclass=5, 
                      gcn_dropout=0.5, output_features=5, num_heads=5)
model = model.to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), 
                    lr=lr,weight_decay=1e-4)
savepath = os.path.join('A_saved_models_CDA_MHtest')
histname = 'A_Training_CDA_MHtest.npz'
os.makedirs(savepath, exist_ok=True)
# Dataset Loading 
datasetpath = 'data_raw_A_NO'
path = os.path.join(datasetpath,'sub'+ str(1)+'.hdf')
dataset = h5py.File(path, 'r')
data = np.array(dataset['data'])
label = np.array(dataset['label'])
x_train = np.zeros((0, *data.shape[1:]), dtype=data.dtype)
y_train = np.zeros((0, ), dtype=label.dtype)
# load training
for sid in tqdm(range(1,33)):
    path = os.path.join(datasetpath, 'sub' + str(sid) + '.hdf')
    with h5py.File(path, 'r') as dataset:
        data = np.array(dataset['data'], dtype=np.float32)
        label = np.array(dataset['label'], dtype=np.int64)

        x_train = np.concatenate((x_train, data[:1110]), axis=0)
        y_train = np.concatenate((y_train, label[:1110]), axis=0)
# load validation
x_valid = np.zeros((0, *data.shape[1:]), dtype=data.dtype)
y_valid = np.zeros((0, ), dtype=label.dtype)
for sid in tqdm(range(1,33)):
    path = os.path.join(datasetpath, 'sub' + str(sid) + '.hdf')
    with h5py.File(path, 'r') as dataset:
        data = np.array(dataset['data'], dtype=np.float32)
        label = np.array(dataset['label'], dtype=np.int64)

        x_valid = np.concatenate((x_valid, data[1110:]), axis=0)
        y_valid = np.concatenate((y_valid, label[1110:]), axis=0)
# numpy array to tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).long()
x_valid = torch.from_numpy(x_valid)
y_valid = torch.from_numpy(y_valid).long()
# y_train = F.one_hot(y_train, 4)
trainset = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
                                          shuffle=True, num_workers=4)
validset = torch.utils.data.TensorDataset(x_valid, y_valid)
validloader = torch.utils.data.DataLoader(validset, batch_size=128, 
                                        shuffle=False, num_workers=4)
print("train: {}, {}".format(x_train.size(), y_train.size()))

# Train
hist = dict(
        loss=np.zeros((epochs, )), val_loss=np.zeros((epochs, )),
        acc=np.zeros((epochs, )), val_acc=np.zeros((epochs, ))
    )
for epoch in range(0,epochs):
    # Training 
    a, b = 0, 0  # hit sample, total sample
    epoch_loss = np.zeros((len(trainloader), ))
    for i, (x_batch, y_batch) in enumerate(tqdm(trainloader, desc=f'Epoch {epoch}/{epochs}')):
        model.train()
        x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.view(-1, 1).to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(x_batch, adj)
        loss = loss_fn(output, y_batch)
        loss.backward() 
        optimizer.step() 

        epoch_loss[i] = loss.item()
        probs = torch.sigmoid(output)
        predictions = (probs > 0.5).float()
        correct = (predictions == y_batch).float().sum().item()
        a += correct
        b += y_batch.size(0)

    hist["loss"][epoch] = epoch_loss.mean()
    hist["acc"][epoch] = a / b
    
    # evaluation
    a, b = 0, 0  # hit sample, total sample
    epoch_loss = np.zeros((len(validloader), ))
    for i, (x_batch, y_batch) in tqdm(enumerate(validloader)):
        model.eval()
        with torch.no_grad():
            x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.view(-1, 1).to(device, dtype=torch.float)
            optimizer.zero_grad()
            output = model(x_batch, adj)
            loss = loss_fn(output, y_batch)
            epoch_loss[i] = loss.item()
            probs = torch.sigmoid(output) 
            predictions = (probs > 0.5).float()
            correct = (predictions == y_batch).float().sum().item()
            a += correct
            b += y_batch.size(0)
    hist["val_loss"][epoch] = epoch_loss.mean()
    hist["val_acc"][epoch] = a / b
    print("Epoch {}: loss={:.4f}, acc={:.4f}, val_loss={:.4f}, val_acc={:.4f}".format(epoch, 
                                                                                      hist["loss"][epoch], 
                                                                                      hist["acc"][epoch], 
                                                                                      hist["val_loss"][epoch], 
                                                                                      hist["val_acc"][epoch]))
    if epoch % 1 == 0:
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
        torch.save(checkpoint, os.path.join(savepath, f"Model-ep{epoch}.pth"))
    np.savez(os.path.join(histname), **hist)