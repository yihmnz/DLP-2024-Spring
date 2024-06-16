import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=self.args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs(args.save_root, exist_ok=True)
        # os.makedirs("transformer_checkpoints_consine", exist_ok=True)

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        # Check the structure of the batch and adjust accordingly
        for batch_idx, data in enumerate(tqdm(train_loader, desc="Training Epoch")):
            if isinstance(data, tuple) and len(data) == 2:
                images, _ = data  # If data is a tuple and only has two elements
            elif isinstance(data, tuple) and len(data) > 2:
                images, targets = data[0], data[1]  # Adjust based on actual data structure
            else:
                images = data  # If data is directly the images

            images = images.to(self.args.device)
            self.optim.zero_grad()
            logits, target = self.model(images)  # Ensure 'target' handling matches your model's expectations
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss.backward()
            self.optim.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        return average_loss

    def eval_one_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            
            for images in tqdm(val_loader, desc="Validation Epoch"):
                images = images.to(self.args.device)
                logits, target = self.model(images)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                total_loss += loss.item()
        average_loss = total_loss/len(val_loader)
        return average_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.00005, help='Learning rate.')
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--MaskGitConfig', type=str, default='./Lab5_code/config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
#TODO2 step1-5:    
    hist = dict(train_loss=np.zeros((args.epochs, )), val_loss=np.zeros((args.epochs, )))
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader)
        print(f'Epoch {epoch}, Training Loss: {train_loss}')
        val_loss = train_transformer.eval_one_epoch(val_loader)
        print(f'Epoch {epoch}, Validation Loss: {val_loss}')
        hist["train_loss"][epoch-1] = train_loss
        hist["val_loss"][epoch-1] = val_loss

        if epoch % args.save_per_epoch == 0:
            torch.save(train_transformer.model.state_dict(), os.path.join(args.save_root, f"transformer_epoch_{epoch}.pt"))

        train_transformer.scheduler.step()
    np.savez(os.path.join(args.save_root, 'training_process.npz'), **hist)