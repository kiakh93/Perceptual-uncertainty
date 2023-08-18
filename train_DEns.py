import statistics as stats
import matplotlib.pyplot as plt
import sys
from torch.utils.data import DataLoader

from sync_batchnorm import convert_model
from models.networks import *
from datasets import *
from model import *


class GANLoss(nn.Module):
    def __init__(self, gan_mode = 'hinge', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = 'ls'
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


def evaluate_set(Set, netG, save_img_path=None):
    if save_img_path is not None:
        os.makedirs(save_img_path, exist_ok=True)

    Tensor = torch.cuda.FloatTensor
    MSEs = []
    netG.eval()

    for i in range(len(Set)):
        batch = Set[i]
        he = (batch['he'].unsqueeze(0).type(Tensor))
        labels = (batch['srs'].unsqueeze(0).type(Tensor))
        Synth_he = netG(labels).detach().cpu()
        he = he.detach().cpu()
        MSEs += [F.mse_loss(Synth_he, he).item()]

        if i%50==0 and save_img_path is not None:
            plt.figure(figsize=(6,3))
            ax = plt.subplot(121)
            ax.imshow(((he[0,:]/2 + 0.5).numpy()*255).astype(int).transpose(1,2,0))
            ax.set(xticks=[], yticks=[])
            ax = plt.subplot(122)
            ax.imshow(((Synth_he[0,:]/2 + 0.5).numpy()*255).astype(int).transpose(1,2,0))
            ax.set(xticks=[], yticks=[])
            plt.tight_layout()
            plt.savefig(save_img_path + f'/{i}.png')

    val_metrics = {}
    val_metrics['MSE'] = round(stats.mean(MSEs), 6)
    return val_metrics


def main(save_path):
    os.makedirs(save_path, exist_ok=True)

    device = 'cuda:0'
    batch_size = 10
    n_cpu = 24
    n_epochs = 70

    netG = UNet(in_channels=4, n_classes=3, depth=5, wf=6, padding=True, batch_norm=True, up_mode='upsample')
    netD = MultiscaleDiscriminator(in_ch=3)

    netG = convert_model(netG)
    netD = convert_model(netD)
    netD.init_weights('xavier',.02)

    if torch.cuda.device_count() >= 1:
        netD = nn.DataParallel(netD)
        netD.to(device)
        netG = nn.DataParallel(netG)
        netG.to(device)
        # Loading the pretrain model
        netG.load_state_dict(torch.load('outputs/SRS DEns/MSE pretrained/e1 (#600)/netG W&B'))

    gan_loss = GANLoss()
    mse_loss = torch.nn.MSELoss()
    Tensor = torch.cuda.FloatTensor
    count = 0

    train_set = ImageDataset("../data from same section/train")
    test_set = ImageDataset_test(r"../data from same section/test")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_cpu)

    for epoch in range(n_epochs):
        for i, batch in enumerate(train_loader):
            LR = 0.0001 * (.97 ** (count // 2000))
            optimizer_G = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.999, 0.9))
            optimizer_D = torch.optim.Adam(netD.parameters(), lr=LR*1, betas=(0.999, 0.9))

            he = (batch['he'].type(Tensor))
            labels = (batch['srs'].type(Tensor))

            optimizer_G.zero_grad()
            # Synthesizing H&E images
            Synth_he = netG(labels)
            pred_fake = netD(Synth_he)
            loss_gan_a = gan_loss(pred_fake, target_is_real=True, for_discriminator=False)
            loss_mse = mse_loss(Synth_he, he)
            loss_G = loss_mse + loss_gan_a
            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            loss_real = gan_loss(netD(he), target_is_real=True, for_discriminator=True)
            loss_fake = gan_loss(netD(Synth_he.detach()), target_is_real=False, for_discriminator=True)
            loss_D = (loss_real + loss_fake)/2
            loss_D.backward()
            optimizer_D.step()

            count += 1
            if count % 5000 == 0:
                # Evaluate and save
                save_path_e = save_path + f'/e{epoch} (#{count})'
                os.makedirs(save_path_e, exist_ok=True)
                torch.save(netG.state_dict(), save_path_e + '/netG W&B')
                torch.save(netD.state_dict(), save_path_e + '/netD W&B')
                evaluate_set(test_set, netG, save_path_e + '/Images')

            sys.stdout.write("\r[Epoch %d] [Count %d] [D loss: %f]" % (epoch, count, loss_D.item(),))


if __name__ == '__main__':
    for i in range(50):
        main(
            save_path = f'outputs/SRS DEns/Ensemble No. {i+1}'
        )