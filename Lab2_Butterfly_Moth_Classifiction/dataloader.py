import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import os

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('dataset/train.csv')
    elif mode == 'test':
        df = pd.read_csv('dataset/test.csv')
    else:
        df = pd.read_csv('dataset/valid.csv')
       
    path = df['filepaths'].tolist()
    label = df['label_id'].tolist()

    # clean data check the list
    valid_indices = [i for i, p in enumerate(path) if os.path.exists(os.path.join('dataset', p))]
    path = [path[i] for i in valid_indices]
    label = [label[i] for i in valid_indices]
    return path, label

class BufferflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        if self.mode == 'test':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                # Convert to tensor (this also normalizes pixel values to [0, 1])
            ])

        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """step1. Get the image path from 'self.img_name' and load it."""
        path = os.path.join(self.root, 'dataset', self.img_name[index])
        img = Image.open(path).convert('RGB')

        """step2. Get the ground truth label from self.label"""
        label = self.label[index]

        """step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]"""
        img = self.transform(img)

        # print(img.shape)
        """step4. Return processed image and label"""
        return img, label
