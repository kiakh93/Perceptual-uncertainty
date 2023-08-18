import os
import torch.nn as nn
import scipy.io as sio
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from AffineT import Affine
import numpy as np
import random

class ImageDataset(Dataset):
    def __init__(self, root):
        self.filesI = os.path.join(root, 'SRS')
        self.list_I = os.listdir(self.filesI)
        self.filesT = os.path.join(root, 'HE')
        self.list_T = os.listdir(self.filesT)

    def __getitem__(self, index):
        # Image processing
        I_name = os.path.join(self.filesI, self.list_I[index % len(self.list_I)])
        im = sio.loadmat(I_name)
        img = im['ss']
        for i in range(1, 5):
            noise = np.random.normal(random.uniform(0, .2), random.uniform(0, .2), (500, 500))
            img[:, :, i] += noise
        img = np.clip(img, 0, 5)
        img = (img - 2.5) / 2.5
        
        T_name = os.path.join(self.filesT, self.list_I[index % len(self.list_T)])
        im = sio.loadmat(T_name)
        img_T = im['xx']

        # Transforms
        lr_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Affine
        affine_transform = Affine(
            rotation_range=random.uniform(0, 180),
            translation_range=(random.uniform(-.15, 0.15), random.uniform(-.15, 0.15)),
            shear_range=random.uniform(-.15, 0.15),
            zoom_range=(random.uniform(.85, 1.15), random.uniform(.85, 1.15))
        )

        CRx, CRy = random.randint(0, 243), random.randint(0, 243)
        fact = 256

        name = self.list_I[index % len(self.list_I)]
        img = affine_transform(lr_transform(img)) - 2.5
        img_T = affine_transform(hr_transform(img_T)) - 1 + 1

        return {'srs': img[1:, CRx:CRx+fact, CRy:CRy+fact], 
                'he': img_T[:, CRx:CRx+fact, CRy:CRy+fact],
                'name': name}

    def __len__(self):
        return len(self.list_T)

    
class ImageDataset_test(Dataset):
    def __init__(self, root):
        self.filesI = os.path.join(root, 'SRS')
        self.list_I = os.listdir(self.filesI)
        self.filesT = os.path.join(root, 'HE')
        self.list_T = os.listdir(self.filesT)

    def __getitem__(self, index):
        # Same processing steps as above, but with minor differences
        I_name = os.path.join(self.filesI, self.list_I[index % len(self.list_I)])
        im = sio.loadmat(I_name)
        img = im['ss']
        for i in range(1, 5):
            noise = np.random.normal(random.uniform(0, .2), random.uniform(0, .2), (500, 500))
            img[:, :, i] += noise
        img = np.clip(img, 0, 5)
        img = (img - 2.5) / 2.5

        T_name = os.path.join(self.filesT, self.list_I[index % len(self.list_T)])
        im = sio.loadmat(T_name)
        img_T = im['xx']

        lr_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        affine_transform = Affine(
            rotation_range=random.uniform(0, 180),
            translation_range=(random.uniform(-.15, 0.15), random.uniform(-.15, 0.15)),
            shear_range=random.uniform(-.15, 0.15),
            zoom_range=(random.uniform(.85, 1.15), random.uniform(.85, 1.15))
        )

        CRx, CRy = random.randint(0, 19), random.randint(0, 19)
        fact = 480

        name = self.list_I[index % len(self.list_I)]
        img = lr_transform(img) - 2.5
        img_T = hr_transform(img_T) - 1 + 1

        return {'srs': img[1:, CRx:CRx+fact, CRy:CRy+fact], 
                'he': img_T[:, CRx:CRx+fact, CRy:CRy+fact],
                'name': name}

    def __len__(self):
        return len(self.list_T)
