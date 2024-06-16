import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

# evaluate the quality of a figure
def Generate_PSNR(imgs1, imgs2, data_range=1.): 
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

# calculate the KL Divergence in VAE
def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD

# control the KL weight (beta) using frange_cycle_linear to avoid optimize KL Divergence to early
# parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
# parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
# parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.args = args
        self.current_epoch = current_epoch

        if self.args.kl_anneal_type == 'None':
            self.schedule = np.ones(args.num_epoch)
        elif self.args.kl_anneal_type == 'Monotonic':
            self.schedule = self.frange_cycle_linear(args.num_epoch, n_cycle=1, ratio=args.kl_anneal_ratio)
        elif self.args.kl_anneal_type == 'Cyclical':
            self.schedule = self.frange_cycle_linear(args.num_epoch, n_cycle=args.kl_anneal_cycle, ratio=args.kl_anneal_ratio)
        
    def update(self):
        # TODO
        self.current_epoch += 1
    
    def get_beta(self):
        # TODO
        return self.schedule[self.current_epoch]

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        L = np.ones(n_iter)
        period = n_iter / n_cycle
        step = (stop-start)/((period-1) * ratio)
        
        for c in range(n_cycle):
            v , i = start , 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        hist = dict(
        loss=np.zeros((self.args.num_epoch, )), mse=np.zeros((self.args.num_epoch, )),
        KLD=np.zeros((self.args.num_epoch, )), psnr=np.zeros((self.args.num_epoch, 629))
    )
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            AVG_loss, AVG_mse, AVG_KLDi = 0, 0, 0
            for idx, (img, label) in enumerate((pbar := tqdm(train_loader, ncols=120))):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                train_loss, train_mse, train_KLD = self.training_one_step(img, label, adapt_TeacherForcing)
                AVG_loss += train_loss.item()
                AVG_mse += train_mse.item()
                AVG_KLDi += train_KLD.item()

                beta = self.kl_annealing.get_beta()
                show_dic = {'ELBO': float(train_loss.detach().cpu()), 'MSE': float(train_mse.detach().cpu()), 'KLD': float(train_KLD.detach().cpu())}
                
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TF: ON, {:.1f}], beta: {:.3f}'.format(self.tfr, beta), pbar, show_dic, lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.3f}'.format(self.tfr, beta), pbar, show_dic, lr=self.scheduler.get_last_lr()[0])
                
            AVG_loss = AVG_loss/len(train_loader)
            AVG_mse = AVG_mse/len(train_loader)
            AVG_KLDi = AVG_KLDi/len(train_loader)

            hist["loss"][i] = AVG_loss
            hist["mse"][i] = AVG_mse
            hist["KLD"][i] = AVG_KLDi

            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            self.eval()
            hist["psnr"][i,:] = self.psnr
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
        np.savez(os.path.join(self.args.save_root, 'training_process.npz'), **hist)    
            
    @torch.no_grad()
    def eval(self):
        avg_psnr =[]
        val_loader = self.val_dataloader()
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr = self.val_one_step(img, label)
            show_dic = {'loss': float(loss.detach().cpu()), 'psnr': psnr}
            self.tqdm_bar('val', pbar, show_dic, lr=self.scheduler.get_last_lr()[0])
            avg_psnr.append(psnr)
        self.psnr = avg_psnr
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        self.train()
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        
        avg_ELBO, avg_KLD, avg_MSE = 0.0, 0.0, 0.0
        pre_out = img[0]
        self.optim.zero_grad()
        for i in range(1, self.train_vi_len):
            # Encoder
            human_feat = self.frame_transformation(img[i])
            label_feat = self.label_transformation(label[i])
            z, mu, logvar = self.Gaussian_Predictor(human_feat, label_feat)
            KLD = kl_criterion(mu, logvar, self.batch_size)

            # Decoder
            out_feat = self.frame_transformation(pre_out)
            parm = self.Decoder_Fusion(out_feat, label_feat, z)
            out = self.Generator(parm)
            MSE = self.mse_criterion(out, img[i])

            #Update predictive frame
            pre_out = img[i] if adapt_TeacherForcing else out
            ELBO = MSE + (KLD * self.kl_annealing.get_beta())

            avg_ELBO += ELBO
            avg_KLD += KLD
            avg_MSE += MSE
            
        avg_ELBO = avg_ELBO/(self.train_vi_len - 1)
        avg_MSE = avg_MSE/(self.train_vi_len - 1)
        avg_KLD = avg_KLD/(self.train_vi_len - 1)

        avg_ELBO.backward()
        self.optimizer_step()

        return avg_ELBO, avg_MSE, avg_KLD
    
    def val_one_step(self, img, label):
        # TODO
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)

        loss, avg_psnr = 0.0, 0.0
        pre_out = img[0]
        for i in range(1, self.val_vi_len):
            # Decoder
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()
            out_feat = self.frame_transformation(pre_out)
            label_feat = self.label_transformation(label[i])
            parm = self.Decoder_Fusion(out_feat, label_feat, z)
            out = self.Generator(parm)
            MSE = self.mse_criterion(out, img[i])

            psnr = Generate_PSNR(out, img[i]).item()
            avg_psnr += psnr
            # if self.args.store_visualization and self.current_epoch == (self.args.num_epoch-1):
            #     self.save_history('psnr', psnr)

            pre_out = out
            loss += MSE
            
        loss = loss/(self.val_vi_len - 1)
        avg_psnr = avg_psnr/(self.val_vi_len - 1)
        # self.save_history('avg_psnr', avg_psnr)

        return loss, avg_psnr
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch >= self.tfr_sde:
            self.tfr = max(self.tfr - self.tfr_d_step, 0.0)
            
    def tqdm_bar(self, mode, pbar, show_dic, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(**show_dic, refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
