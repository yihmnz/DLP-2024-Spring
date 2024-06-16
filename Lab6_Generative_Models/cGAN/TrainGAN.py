import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from torch.utils.data import DataLoader
import numpy as np
import torchvision.models as models
from GAN.generator import Generator
from GAN.discriminator import Discriminator
import json
from ObjectDataloader import MultiLabelDataset 
from tqdm import tqdm

def main():
    # Set device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    savepath = os.path.join('saved_models_cGAN_3')
    os.makedirs(savepath, exist_ok=True)
    # Load training data
    def load_json_labels(json_file):
        with open(json_file, 'r') as f:
            labels = json.load(f)
        return labels
    label_mapping  = load_json_labels('./Lab6_code/objects.json')
    json_labels = load_json_labels('./Lab6_code/train.json')
    root = './Lab6_dataset'
    dataset = MultiLabelDataset(root, label_mapping, json_labels)
    train_dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

    # GAN model setting
    '''
        ADC GAN contains training of discriminator (real/false imgae), classifier (labels in fig) 
        and training of generator (synthesized images) 
    '''
    ### Weight initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    ## Generator
    Gen = Generator()
    Gen = Gen.to(device = device)
    Gen.apply(weights_init)

    # Discriminator
    Dis = Discriminator()
    Dis = Dis.to(device = device)
    Dis.apply(weights_init)

    ## Classifier (pre-trained by TA)
    '''
        ○ Images should be all generated images. E.g. (batch size, 3, 64, 64)
        ○ Images should be normalized with transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    '''
    checkpoint = torch.load('./Lab6_code/checkpoint.pth')
    Classif = models.resnet18(pretrained=False)
    Classif.fc = nn.Sequential(
        nn.Linear(512,24),
        nn.Sigmoid())
    Classif.load_state_dict(checkpoint['model'])
    Classif = Classif.to(device = device)
    # Freeze discriminator parametors
    for param in Classif.parameters(): 
        param.requires_grad = False

    # Parameter settings
    criterion = nn.BCELoss()  # Real/Fake classification loss
    criterion_class = nn.BCELoss() # Multi-label classification loss
    ## Create batch of latent vectors that we will use to visualize

    ## The progression of the generator
    # fixed_noise = torch.randn(64, 24, 1, 1, device=device)
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Setup Adam optimizers for Generator
    lr = 0.0002
    lr_dis = 0.0002
    beta1 = 0.5
    num_epochs = 20001
    optimizerGen = optim.Adam(Gen.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDis = optim.Adam(Dis.parameters(), lr=lr_dis, betas=(beta1, 0.999))
    # optimizerClassif = optim.Adam(Classif.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training
    His_Train_Process = dict(
        generate_loss=np.zeros((num_epochs, )), classify_loss=np.zeros((num_epochs, )),
        classify_acc=np.zeros((num_epochs, )), discriminate_loss=np.zeros((num_epochs, ))
    )

    print('-----start_training-----')

    # Save Training Process
    itr = 2
    lamda = 0.2
    update = False
    for epoch in range(num_epochs):
        if epoch > 6000:
            lamda = 0.2 # + 3 * (epoch / num_epochs)
            if epoch % 2 == 0:
                update = True
            elif epoch % 2 != 0:
                update = False
        else:
            if epoch % 1 == 0:
                update = True
            elif epoch % 1 != 0:
                update = False

        CLASSIFY_LOSS = 0.0
        GENERATOR_LOSS = 0.0
        DISCRIMINATOR_LOSS = 0.0
        CLASSIFY_ACC = 0.0
        for a, (real_images, class_labels, class_labels_class) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs}')):
            real_images = real_images.to(device)
            class_labels = class_labels.to(device)
            class_labels_class = class_labels_class.to(device)

            # Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
            # Discriminate real image 
            b_size = real_images.size(0)
            true_label_update = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = Dis(real_images, class_labels).view(-1)
            Dis_real_loss = criterion(output, true_label_update)
            D_x = output.mean().item()
            # Discriminate fake images
            ## 1. Generate fake images 
            noise_condition = 512
            noise = torch.randn(b_size, noise_condition, 1, 1, device=device)-1
            # class_labels_expanded = class_labels.view(b_size, 24, 1, 1)
            fake_images = Gen(noise, class_labels)
            ## 2. discriminate fake images 
            fake_label_update = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output = Dis(fake_images.detach(), class_labels).view(-1)
            Dis_fake_loss = criterion(output, fake_label_update)
            D_G_z1 = output.mean().item()
            errD = Dis_real_loss + Dis_fake_loss
            DISCRIMINATOR_LOSS += errD.item()

            # # Update Generator network: maximize log(D(G(z))) + loss (classfier)
            # ## Update Classifier on fake images
            with torch.no_grad():
                class_out_fake = Classif(fake_images.detach())          
                class_loss_fake = criterion_class(class_out_fake, class_labels_class)
                out, onehot_labels = class_out_fake.cpu(), class_labels_class.cpu()
                acc = 0
                total = 0
                for i in range(b_size):
                    k = int(onehot_labels[i].sum().item())
                    total += k
                    _, outi = out[i].topk(k)
                    _, li = onehot_labels[i].topk(k)
                    for j in outi:
                        if j in li:
                            acc += 1
                CLASSIFY_ACC += acc / total

            # Update Discriminator
            if update: # 5 Times in first 6000 epoch
                Dis.train()
                optimizerDis.zero_grad()
                # errD += lamda * class_loss_fake
                errD.backward(retain_graph=True)
                optimizerDis.step()

            ## Update Classifier on fake images
            Gen.train()
            optimizerGen.zero_grad()
            output = Dis(fake_images, class_labels).view(-1)
            Generator_loss = 0.8 * criterion(output, true_label_update) + lamda * class_loss_fake
            GENERATOR_LOSS += Generator_loss.item()
            CLASSIFY_LOSS += class_loss_fake.item()
            Generator_loss.backward()
            optimizerGen.step()

            # Test evaluation

        CLASSIFY_LOSS /= len(train_dataloader)
        GENERATOR_LOSS /= len(train_dataloader)
        DISCRIMINATOR_LOSS /= len(train_dataloader)
        CLASSIFY_ACC /= len(train_dataloader)
        
        His_Train_Process["generate_loss"][epoch] = GENERATOR_LOSS
        His_Train_Process["classify_acc"][epoch] = CLASSIFY_ACC
        His_Train_Process["classify_loss"][epoch] = CLASSIFY_LOSS
        His_Train_Process["discriminate_loss"][epoch] = DISCRIMINATOR_LOSS

        print(f'Epoch [{epoch}/{num_epochs:04d}], class_loss: {CLASSIFY_LOSS:.4f}, class_acc: {CLASSIFY_ACC:.4f}, g_loss: {GENERATOR_LOSS:.4f}, discrim_loss: {DISCRIMINATOR_LOSS:.4f}')
        if epoch % 100 == 0:
            checkpoint = {
                'epoch': epoch,
                'Gen_state_dict': Gen.state_dict(),
                'Gen_optimizer': optimizerGen.state_dict(),
                'Dis_state_dict': Dis.state_dict(),
                'Dis_optimizer': optimizerDis.state_dict(),
                }
            torch.save(checkpoint, os.path.join(savepath, f"Model-ep{epoch}.pth"))
        np.savez(os.path.join('training_process_cGAN_3.npz'), **His_Train_Process)
if __name__ == "__main__":
    main()