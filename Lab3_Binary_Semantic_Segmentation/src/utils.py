import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def training_curve(hist_unet, hist_resunet):
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.plot(hist_unet['acc'], color = (143/255,170/255,220/255), label = 'Unet Train')
    plt.plot(hist_unet['val_acc'],color = (47/255,82/255,143/255), label = 'Unet Valid')
    plt.plot(hist_resunet['acc'], color = (244/255,177/255,131/255), label = 'ResUnet Train')
    plt.plot(hist_resunet['val_acc'], color = (174/255,90/255,33/255), label = 'ResUnet Valid')
    plt.title('Dice score')
    plt.xlabel('Epochs')
    plt.legend(frameon=False)
    plt.show()
    plt.savefig("training curve.jpg")
    
def dice_score(output, label):
    output = torch.sigmoid(output) 
    preds = output > 0.5
    label = label > 0.5
    intersection = (preds & label).float().sum()
    total_size = preds.float().sum() + label.float().sum()
    
    dice = (2. * intersection+1e-6) / (total_size+1e-6) 
    return dice.mean()

def draw_pic(TestData, Model1, best_epoch1, savepath_unet, fig_num, pic_try, device, a):
    test_model_path1 = os.path.join(savepath_unet, "{}-ep{}.pth".format("Model", best_epoch1))
    checkpoint = torch.load(test_model_path1, map_location="cpu")
    Model1.load_state_dict(checkpoint["state_dict"])
    # evaluate(Model1, test_loader, loss_fn, device)
    Model1.eval()
    dice_scores = []
    for i, images_dist in enumerate(TestData):
        if i == fig_num:
            # Move data to the appropriate device (CPU or GPU)
            images = images_dist['image'][0:1,:,:,:].to(device)
            labels = images_dist['mask'][0:1,:,:,:].to(device)
            outputs = Model1(images)
            Dice_score = dice_score(outputs, labels)
            dice_scores.append(Dice_score.item())
            average_dice_score = str(round(sum(dice_scores) / len(dice_scores), 3))
            labels = torch.squeeze(labels, 1)
            outputs = torch.squeeze(outputs, 1)
            images = torch.squeeze(images, 0)

    images = images.permute(1, 2, 0).cpu().numpy()
    outputs = outputs.permute(1, 2, 0).detach().cpu().numpy()
    labels = labels.permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(images)
    axs[0].axis('off')
    axs[2].set_title('Dice score = '+ average_dice_score)
    axs[2].imshow(outputs)
    axs[2].axis('off')
    # axs[2].set_title('prediction')
    axs[1].imshow(labels)
    axs[1].axis('off')
    axs[1].set_title('GT')
    plt.show()
    if a == 0:
        axs[0].set_title('Unet test on Fig.{}'.format(fig_num))
        plt.savefig("Prediction_{}_Unet_Fig{}.jpg".format(pic_try, fig_num))
    else:
        axs[0].set_title('ResUnet test on Fig.{}'.format(fig_num))
        plt.savefig("Prediction_{}_ResUnet_Fig{}.jpg".format(pic_try, fig_num))

def draw_pic_on_pretrained(TestData, Model1, test_model_path1, fig_num, pic_try, device, a):
    checkpoint = torch.load(test_model_path1, map_location="cpu")
    Model1.load_state_dict(checkpoint["state_dict"])
    # evaluate(Model1, test_loader, loss_fn, device)
    Model1.eval()
    dice_scores = []
    for i, images_dist in enumerate(TestData):
        if i == fig_num:
            # Move data to the appropriate device (CPU or GPU)
            images = images_dist['image'][0:1,:,:,:].to(device)
            labels = images_dist['mask'][0:1,:,:,:].to(device)
            outputs = Model1(images)
            Dice_score = dice_score(outputs, labels)
            dice_scores.append(Dice_score.item())
            average_dice_score = str(round(sum(dice_scores) / len(dice_scores), 3))
            labels = torch.squeeze(labels, 1)
            outputs = torch.squeeze(outputs, 1)
            images = torch.squeeze(images, 0)

    images = images.permute(1, 2, 0).cpu().numpy()
    outputs = outputs.permute(1, 2, 0).detach().cpu().numpy()
    labels = labels.permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(images)
    axs[0].axis('off')
    axs[2].set_title('Dice score = '+ average_dice_score)
    axs[2].imshow(outputs)
    axs[2].axis('off')
    # axs[2].set_title('prediction')
    axs[1].imshow(labels)
    axs[1].axis('off')
    axs[1].set_title('GT')
    plt.show()
    if a == 0:
        axs[0].set_title('Unet test on Fig.{}'.format(fig_num))
        plt.savefig("Prediction_{}_Unet_Fig{}.jpg".format(pic_try, fig_num))
    else:
        axs[0].set_title('ResUnet test on Fig.{}'.format(fig_num))
        plt.savefig("Prediction_{}_ResUnet_Fig{}.jpg".format(pic_try, fig_num))