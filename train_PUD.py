import glob
import random
import os
import sys
import torch.nn as nn

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from IPython.core.debugger import set_trace
from AffineT import *
from train_DEns import *

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import resnet50
import torch.optim as optim


class Imagedataset_for_PUD(Dataset):
    def __init__(self, root, augment=False):
        self.filesI = os.path.join(root, 'SRS')
        self.list_I = os.listdir(self.filesI)
        self.filesT = os.path.join(root, 'HE')
        self.list_T = os.listdir(self.filesT)
        self.filesU = os.path.join(root, 'PU')
        self.list_U = os.listdir(self.filesU)
        self.augment = augment

    def __getitem__(self, index):
        I_name = os.path.join(self.filesI, self.list_I[index % len(self.list_I)])
        im = sio.loadmat(I_name)
        img = im['ss']
        if self.augment:
            img2 = img*0
            for i in range(1,5):
                noise = np.random.normal(random.uniform(0,.2),random.uniform(0,.2),(500,500))
                img2[:,:,i] = img[:,:,i] + noise
            img = img2
        img[img > 5] = 5
        img[img < 0] = 0
        img = (img - 2.5) / 2.5

        T_name = os.path.join(self.filesT, self.list_I[index % len(self.list_T)])
        im = sio.loadmat(T_name)
        img_T = im['xx']

        U_name =  os.path.join(self.filesU, self.list_I[index % len(self.list_U)])[:-4] + '.pt'
        uq = torch.load(U_name)

        lr_transforms = [
            transforms.ToTensor(),]
        hr_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)

        name = self.list_I[index % len(self.list_I)]
        img = self.lr_transform(img)
        img_T = self.hr_transform(img_T)
        img_T[img_T > 1] = 1
        img_T[img_T < -1] = -1

        return {'input': img[1:, :, :], 'he': img_T, 'name': name, 'uq': uq,}

    def __len__(self):
        return len(self.list_T)



class Surrogate_ResNet50_Backbone(torch.nn.Module):
    def __init__(self, in_channel, config='tune'):
        super().__init__()
        if config=='tune':
            pretrained, requires_grad = True, True
        elif config=='fix':
            pretrained, requires_grad = True, False
        elif config=='scratch':
            pretrained, requires_grad = False, False
        else:
            raise NotImplementedError

        if in_channel != 3:
            self.in_layer = torch.nn.Sequential(
                *[nn.Conv2d(in_channels=in_channel, out_channels=3, kernel_size=3, stride=1, padding=1), nn.ReLU()])
        else:
            self.in_layer = None

        self.resnet_layers = torch.nn.Sequential(*list(resnet50(pretrained=pretrained).children())[:-2])

        self.out_layer = torch.nn.Sequential(
            *[nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
              nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
              nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1), nn.Softplus()])

        if not requires_grad:
            for param in self.resnet_layers:
                param.requires_grad = False

    def forward(self, X):
        if self.in_layer is not None:
            f1 = self.in_layer(X)
        else:
            f1 = X
        f2 = self.resnet_layers(f1)
        f3 = self.out_layer(f2)
        return f3


def Evaluate_Surrogate(Set, Surrogate, save_img_path=None, dsc_t=0.06):
    if save_img_path is not None:
        os.makedirs(save_img_path, exist_ok=True)

    Tensor = torch.cuda.FloatTensor
    Surrogate.eval()
    MSEs = 0.
    DSCs = 0.

    for i in range(len(Set)):
        batch = Set[i]
        labels = (batch['input'].unsqueeze(0).type(Tensor))
        uq = (batch['uq'].unsqueeze(0).type(Tensor))
        uq_hat = Surrogate(labels)
        uq, uq_hat = uq.detach().cpu(), uq_hat.detach().cpu()

        # MSE:
        MSEs += F.mse_loss(uq, uq_hat).item() / len(Set)

        # DSC:
        target2 = uq.detach().clone()
        uq_map2 = uq_hat.detach().clone()
        target2[target2 < dsc_t] = 0
        target2[target2 >= dsc_t] = 1
        uq_map2[uq_map2 < dsc_t] = 0
        uq_map2[uq_map2 >= dsc_t] = 1
        DSCs += (2 * (target2 * uq_map2).sum() / (target2.sum() + uq_map2.sum() + 1e-6)).item() / len(Set)

        if i % 100 == 0 and save_img_path is not None:
            plt.figure(figsize=(6, 3))
            uq_min, uq_max = uq.min(), uq.max()
            uq = F.interpolate(uq, scale_factor=labels.size(3) / uq.size(2), mode='bicubic')  # (B, 1, H, W)
            uq_hat = F.interpolate(uq_hat, scale_factor=labels.size(3) / uq_hat.size(2), mode='bicubic')

            ax = plt.subplot(121)
            ax.imshow(uq[0,0,:,:].detach().cpu(), cmap='jet', vmin=uq_min, vmax=uq_max)
            ax.set(xticks=[], yticks=[])
            plt.title('UQ truth')

            ax = plt.subplot(122)
            ax.imshow(uq_hat[0,0,:,:].detach().cpu(), cmap='jet', vmin=uq_min, vmax=uq_max)
            ax.set(xticks=[], yticks=[])
            plt.title('UQ prediction')

            plt.tight_layout()
            plt.savefig(save_img_path + f'/{i}.png')
            # plt.show()
            plt.close('all')

    return round(MSEs, 6), round(DSCs, 6)


def main(save_path,
         config='tune'
         ):

    os.makedirs(save_path, exist_ok=True)

    # Hyperparameters
    device = 'cuda:0'
    batch_size = 16
    n_cpu = 8
    n_epochs = 30
    eval_interval = 200

    # Model
    Surrogate = Surrogate_ResNet50_Backbone(4, config)
    Surrogate = convert_model(Surrogate)
    Surrogate.to(device)

    optimizer_S = torch.optim.Adam(Surrogate.parameters(), lr=1e-4, betas=(0.9, 0.99))
    scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, milestones=[20, 24, 27], gamma=0.5)

    if torch.cuda.device_count() >= 1:
        Surrogate = nn.DataParallel(Surrogate)
        Surrogate.to(device)

    # Datasets:
    train_val_set = Imagedataset_for_PUD(r"Datasets/SRS/data from same section/train", augment=False)
    train_set, val_set = torch.utils.data.random_split(
        train_val_set, [len(train_val_set) - len(train_val_set)//10, len(train_val_set)//10]
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, )
    test_set = Imagedataset_for_PUD(r"Datasets/SRS/data from same section/test", augment=False)

    # Start training:
    Tensor = torch.cuda.FloatTensor
    count = 0
    best_score, best_count = 1e10, 0  # for early stopping
    test_mse, test_dsc = np.nan, np.nan

    for epoch in range(n_epochs):

        for i, batch in enumerate(train_loader):

            labels = (batch['input'].type(Tensor))
            uq = (batch['uq'].type(Tensor))

            optimizer_S.zero_grad()
            uq_hat = Surrogate(labels)
            loss = F.mse_loss(uq_hat, uq)
            loss.backward()
            optimizer_S.step()

            count += 1
            sys.stdout.write("\r[Epoch %d] [Count %d] [MSE loss: %f] Best-[Count: %d, MSE: %f, DSC: %f]"
                             % (epoch, count, round(loss.item(), 4), best_count, test_mse, test_dsc))

            if count % eval_interval == 0:
                vaMSE, vaDSC = Evaluate_Surrogate(val_set, Surrogate, None)
                current_score = 1000*vaMSE + (1.-vaDSC)  # stopping criterion

                if current_score < best_score:
                    best_score = current_score
                    best_count = count
                    save_path_e = save_path + f'/Final' # f'/e{epoch} (#{count})'
                    os.makedirs(save_path_e, exist_ok=True)
                    torch.save(Surrogate.state_dict(), save_path_e + '/Surrogate W&B')
                    test_mse, test_dsc = Evaluate_Surrogate(test_set, Surrogate, save_path_e + '/Images')

        scheduler_S.step()


if __name__ == '__main__':
    main(f'outputs/PUD/Tune', config='tune')