from torchvision.models import vgg19, alexnet, resnet50
from tqdm import tqdm

from train_DEns import *
from model import *
from twoAFC import Pretrained_network


def Get_Features(img, net_name=resnet50, layer_num=-1):
    device = img.device
    Extractor = Pretrained_network(net_name, layer_num)
    Extractor = nn.DataParallel(Extractor)
    Extractor.to(device)
    Extractor.eval()
    features = Extractor(img / 2 + 0.5)
    return features


def Get_Feature_DEns(labels, DEns_member_index, net_name=resnet50, layer_num=-1):  # (B, 4, H, W)
    device = labels.device
    netG = UNet(in_channels=4, n_classes=3, depth=5, wf=6, padding=True, batch_norm=True, up_mode='upsample')
    netG = nn.DataParallel(netG)
    netG.to(device)

    resnet = Pretrained_network(net_name, layer_num)
    resnet = nn.DataParallel(resnet)
    resnet.to(device)

    Synth_he_list = []
    for i in DEns_member_index:
        netG.load_state_dict(torch.load(f'outputs/SRS DEns/Ensemble No. {i}/e68 (#30000)/netG W&B'))
        netG.eval()
        Synth_he_list += [netG(labels).detach()]
    Synth_he = torch.stack(Synth_he_list, dim=0)
    N, B, _, H, W = Synth_he.shape

    resnet.eval()
    Synth_he2 = Synth_he.reshape((N*B, 3, H, W))
    features = resnet(Synth_he2/2 + 0.5)
    _, f_channel, f_size, _ = features.shape
    features = features.reshape((N, B, f_channel, f_size, f_size))

    f_median = torch.median(features, dim=0, keepdim=True)[0]
    f_MAD = (torch.median(torch.abs(features - f_median), dim=0)[0] * 1.4826) ** 2 # (B, f_channel, f_size, f_size)
    pu_map = f_MAD.mean(1).unsqueeze(1)  # (B, 1, f_size, f_size)

    return  Synth_he, pu_map, features


def main(file_path="Datasets/SRS/data from same section/test/",
         net_name=resnet50, layer_num=-1
         ):

    os.makedirs(file_path + 'PU', exist_ok=True)
    os.makedirs(file_path + 'Raw Features', exist_ok=True)
    os.makedirs(file_path + 'Raw Features (Ground truth)', exist_ok=True)

    lr_transforms = transforms.Compose([transforms.ToTensor(),])
    hr_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    filesI =  file_path + 'SRS'
    list_I = os.listdir(filesI)

    for I in tqdm(list_I):
        im = sio.loadmat(filesI + '/' + I)
        img = im['ss']
        img[img > 5] = 5
        img[img < 0] = 0
        img = (img - 2.5) / 2.5
        img = lr_transforms(img)
        labels = img[1:,:,:].float().to('cuda')
        _, uq, features = Get_Feature_DEns(labels.unsqueeze(0), list(range(1, 51)), net_name, layer_num)
        torch.save(uq[0, :], file_path + f"PU/{I[:-4]}.pt")
        torch.save(features[:,0,:,:,:], file_path + f"Raw Features/{I[:-4]}.pt")

        he = sio.loadmat(file_path + 'HE' + '/' + I)
        he = he['xx']
        he = hr_transforms(he)
        he[he>1]=1
        he[he<-1]=-1
        he = he.unsqueeze(0).float().to('cuda')
        gt_features = Get_Features(he, net_name, layer_num)
        torch.save(gt_features[0, :], file_path + f"Raw Features (Ground truth)/{I[:-4]}.pt")

        sys.exit()


if __name__ == '__main__':
    main()