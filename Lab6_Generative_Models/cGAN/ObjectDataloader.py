import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import os
import json
import torch
import numpy as np

class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label_mapping, json_labels):
        self.root = root
        self.label_mapping = label_mapping
        self.json_labels = json_labels
        self.img_name = list(self.json_labels.keys())
        self.transform = transforms.Compose([
                               transforms.Resize((64, 64)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        print("> Found %d images..." % len(self.img_name))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        path = os.path.join(self.root, 'train', self.img_name[index])
        img = Image.open(path).convert('RGB')

        # 多標籤處理
        img_file = self.img_name[index]
        labels = self.json_labels.get(img_file, [])

        # 將多標籤轉換為多熱編碼
        label_vector = np.zeros(len(self.label_mapping), dtype=np.float32)
        label_vector_class = np.zeros(len(self.label_mapping), dtype=np.float32)
        for label in labels:
            label_index = self.label_mapping.get(label)
            if label_index is not None:
                label_vector[label_index] = label_index+1
                label_vector_class[label_index] = 1

        img = self.transform(img)
        label_vector = torch.tensor(label_vector)
        label_vector_class = torch.tensor(label_vector_class)
        # print(label_vector)
        return img, label_vector, label_vector_class
