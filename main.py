import torch
from torch import optim
from dataloader import ImgDatasets
from model import AutoEncoder, Discriminator
from trainer import Trainer
from config import *

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if __name__ == "__main__":
    AE = AutoEncoder(nc_img, nc_cond, n_filter, n_key, dim_key, dim_val)
    Disc = Discriminator(nc_img, nc_cond, n_filter)

    AE_optimizer = optim.Adam(AE.parameters(), lr=LR, weight_decay=1e-4)
    Disc_optimizer = optim.Adam(Disc.parameters(), lr=LR, weight_decay=1e-4)

    train_set = ImgDatasets(DATA_ROOT, mode='train')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    dev_set = ImgDatasets(DATA_ROOT, mode='validate')
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False)

    trainer = Trainer(AE, Disc, AE_optimizer, Disc_optimizer, train_loader, dev_loader)
    trainer.train(EPOCHS, PRINT_FREQ, VAL_FREQ)


