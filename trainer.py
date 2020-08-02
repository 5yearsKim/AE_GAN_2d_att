import torch
from torch import nn
from inference import Inferencer
from utils import AverageMeter


class Trainer:
    def __init__(self, AE, Disc, AE_optimizer, Disc_optimizer, train_loader, dev_loader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.AE = AE.to(self.device)
        self.Disc = Disc.to(self.device)
        self.AE_optimizer = AE_optimizer
        self.Disc_optimizer = Disc_optimizer
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.epoch=0
        # self.validate()

    def train(self, epochs, print_freq=10, val_freq=1):
        self.AE.train()
        self.Disc.train()
        rc_meter = AverageMeter('reconstruction_loss')
        AE_gan_meter = AverageMeter('AE_GAN_loss')
        Disc_gan_meter = AverageMeter('Disc_GAN_loss')
        for epoch in range(epochs):
            self.epoch = epoch
            for idx, ((x, in_cond), (y, out_cond)) in enumerate(self.train_loader):
                self.train_step(x, y, in_cond, out_cond, rc_meter, AE_gan_meter, Disc_gan_meter)
                if (idx + 1) % print_freq == 0:
                    print(f"{epoch} epoch | step {idx + 1} : RC_loss = {rc_meter.avg:.2f}, ")
                    print(f"---AE_gan_loss = {AE_gan_meter.avg:.2f}, Disc_gan_loss = {Disc_gan_meter.avg:.2f}")
            print(f"{epoch} epoch total result: RC_loss = {rc_meter.avg:.2f}, AE_gan_loss = {AE_gan_meter.avg:.2f}, Disc_gan_loss = {Disc_gan_meter.avg:.2f}")
            rc_meter.reset()
            AE_gan_meter.reset()
            Disc_gan_meter.reset()
            if epoch % val_freq == 0:
                self.validate()
                self.AE.train()
                self.Disc.train()

    def train_step(self, x, y, in_cond, out_cond, rc_meter, AE_gan_meter, Disc_gan_meter):
        x, y, in_cond, out_cond = x.to(self.device), y.to(self.device), in_cond.to(self.device), out_cond.to(self.device)

        # step 1 - train for AutoEncoder

        rc_img = self.AE(x, in_cond, out_cond)
        rc_loss = self.mse_loss(rc_img, y).to(self.device) * 500

        b_size = x.size(0)
        real_flag = torch.full((b_size, 1), 1, device=self.device)
        fake_flag = torch.full((b_size, 1), 0, device=self.device)

        gan_rc_loss = self.bce_loss(self.Disc(rc_img, out_cond), real_flag)

        idx = torch.randperm(b_size)
        mix_cond = out_cond[idx]
        mix_img = self.AE(x, in_cond, mix_cond)
        gan_mix_loss = self.bce_loss(self.Disc(mix_img, mix_cond), real_flag)

        total_loss = rc_loss + 1 * (gan_rc_loss + gan_mix_loss)

        self.AE_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.AE.parameters(), 0.5)
        if AE_gan_meter.avg > 1.:
            self.AE_optimizer.step()
        rc_meter.update(rc_loss.to('cpu').item())
        AE_gan_meter.update(gan_rc_loss.to('cpu').item() + gan_mix_loss.to('cpu').item())


        # step2 - train for Discriminator
        real_loss = self.bce_loss(self.Disc(x, out_cond), real_flag)
        fake_rc_loss = self.bce_loss(self.Disc(rc_img.detach(), out_cond), fake_flag)
        fake_mix_loss = self.bce_loss(self.Disc(mix_img.detach(), mix_cond), fake_flag)

        total_loss = real_loss + fake_rc_loss + fake_mix_loss

        self.Disc_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Disc.parameters(), 0.5)
        if Disc_gan_meter.avg > 1.:
            self.Disc_optimizer.step()
        Disc_gan_meter.update(total_loss.to('cpu').item())




    def validate(self):
        print("validating...")
        self.AE.eval()
        self.Disc.eval()
        # loss_meter = AverageMeter("val_loss")

        # for (x, in_cond), (y, out_cond) in self.dev_loader:
        #     x, y, in_cond, out_cond = x.to(self.device), y.to(self.device), in_cond.to(self.device), out_cond.to(self.device)
        #     img = self.model(x, in_cond, out_cond)
        #     loss = self.criterion(img, y).to(self.device)
        #     loss_meter.update(loss.to('cpu').item())

        # save sample pictures
        inferencer = Inferencer(self.AE)
        (x, in_cond), (y, out_cond) = next(iter(self.dev_loader))
        x, in_cond, y, out_cond = x[:16].to(self.device), in_cond[:16].to(self.device),\
                                  y[:16].to(self.device), out_cond[:16].to(self.device)
        inferencer.inference(x, in_cond, out_cond, save_path=f"validate/epoch{self.epoch}.png")
        self.save_model('ckpts/best.pt')

    def save_model(self, save_path):
        state = {
            'AE': self.AE.state_dict(),
            'Disc': self.Disc.state_dict(),
        }
        torch.save(state, save_path)
        print("model saved")

    def load_model(self, load_path):
        state = torch.load(load_path)
        self.AE.load_state_dict(state['AE'])
        self.Disc.load_state_dict(state['Disc'])
        print('model loaded')








