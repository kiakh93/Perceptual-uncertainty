from torchvision.models import vgg19, alexnet, resnet50
from PIL import Image
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn.functional as F
import os
import scipy.io as sio
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import wasserstein_distance
import statistics as stats

class Pretrained_network(torch.nn.Module):
    def __init__(self, pretrained_model, last_layer_num=-1, requires_grad=False):
        super().__init__()
        try:
            pretrained_layers = pretrained_model(pretrained=True).features
        except:
            pretrained_layers = list(pretrained_model(pretrained=True).children())

        if last_layer_num == -1:
            self.network = torch.nn.Sequential(*pretrained_layers[:-2])
        else:
            self.network = torch.nn.Sequential(*pretrained_layers[:last_layer_num])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        return self.network(X)

def Extract_feature(img, pretrained_net):
    device = img.device
    pretrained_net.to(device)
    pretrained_net.eval()
    features = pretrained_net(img / 2 + 0.5)
    return features


# Pretrained networks to consider:
VGG19 = Pretrained_network(vgg19, 36)
AlexNet = Pretrained_network(alexnet, 12)
ResNet50 = Pretrained_network(resnet50, -1)


def main(patch_number=100000,
         device='cuda:0'):

    # Log 2AFC successful count and perceptual distances (to later compute W, JSdiv).
    y1_dist = [[], [], []]  # VGG, AlexNet, ResNet
    yprime_dist = [[], [], []]
    twoAFC_scores = [0., 0., 0.]

    # Transformations
    normal_transform = A.Compose([
        A.Normalize(mean=(.5, 0.5, 0.5), std=(.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    # "Reasonable" perturbations to generate y => y_1
    distortion_transform = A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
        A.GaussNoise(var_limit=(0, 50.0), mean=0, per_channel=True, p=1),
        A.ISONoise(color_shift=(0, 0.05), intensity=(0, 0.5), p=1),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4,
                           value=None, mask_value=None, approximate=True, same_dxdy=True, p=1),
        A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4,
                         value=None, mask_value=None, normalized=False, p=1),
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4,
                            value=None, p=1),
        A.Normalize(mean=(.5, 0.5, 0.5), std=(.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    # Image path of y (Modify accordingly...)
    filesI = r"Datasets/SRS/data from same section/train/HE"
    list_I = os.listdir(filesI)
    path_list = [filesI + '/' + L for L in list_I]

    random.shuffle(path_list)
    path_list2 = path_list.copy()
    random.shuffle(path_list2)  # Generate path for y' (to compare w/ y_1)

    count = 0
    break_flag=False

    for n in range(1000):

        for I in path_list:
            img = sio.loadmat(I)['xx']
            img_prime = sio.loadmat(path_list2[count % len(path_list)])['xx']

            count += 1

            img0 = normal_transform(image=img)['image'].to(device)  # y
            img1 = distortion_transform(image=img)['image'].to(device)  # y_1
            img2 = normal_transform(image=img_prime)['image'].to(device)  # y'

            for j, net in enumerate([VGG19, AlexNet, ResNet50]):
                df0 = Extract_feature(img0.unsqueeze(0), net)
                df1 = Extract_feature(img1.unsqueeze(0), net)
                df2 = Extract_feature(img2.unsqueeze(0), net)

                d1 = F.mse_loss(df0, df1).item()  # perceptual score (y, y_1)
                d2 = F.mse_loss(df0, df2).item()  # perceptual score (y, y')

                y1_dist[j] += [d1]
                yprime_dist[j] += [d2]
                if d1 < d2:
                    twoAFC_scores[j] += 1.  # Count successful case

            if count == patch_number:
                break_flag = True
                break

            if count % 100 == 0:
                print(f'Fin. w/ {count} patches...')

        if break_flag:
            break

    # Compute W & JSdiv from logged (y, y_1), (y, y') distances:
    def kl(p, q):
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        return np.sum(p * np.log(p / q))

    # In order, VGG, AlexNet, ResNet
    for j in range(3):
        d1 = y1_dist[j]
        d2 = yprime_dist[j]
        d_avg = d1 + d2

        # Normalize (for fair comparison)
        mu, std = stats.mean(d_avg), stats.stdev(d_avg)
        d1 = [(d - mu)/std for d in d1]
        d2 = [(d - mu) / std for d in d2]
        d_avg = [(d - mu) / std for d in d_avg]

        # Discretize (for JSdiv)
        binrange = np.arange(-4, 4, 0.25)
        histavg, bin_range = np.histogram(d_avg, bins=binrange, density=True)
        hist1 = np.histogram(d1, bins=bin_range, density=True)[0]
        hist2 = np.histogram(d2, bins=bin_range, density=True)[0]

        # Smoothen & Clamp
        hist1 = gaussian_filter(hist1, sigma=3)
        hist2 = gaussian_filter(hist2, sigma=3)
        histavg = gaussian_filter(histavg, sigma=3)
        hist1 = [max(h, 1e-3) for h in hist1]
        hist2 = [max(h, 1e-3) for h in hist2]
        histavg = [max(h, 1e-3) for h in histavg]

        # Compute metrics (higher is better)
        TAFC = twoAFC_scores[j] / count  # 2AFC (%)
        Wasserstein = wasserstein_distance(d1, d2)  # W
        JSdiv = (kl(hist1, histavg) + kl(hist2, histavg))/2  # JSdiv
        print(TAFC, Wasserstein, JSdiv)


if __name__ == '__main__':
    main(patch_number=100)