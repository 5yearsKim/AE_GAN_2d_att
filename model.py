import torch
from torch import nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, nc_img=3, nc_cond=1, n_filter=32, n_key=8, dim_key=4, dim_val=4):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder(nc_img, nc_cond, n_filter, n_key, dim_key, dim_val)
        self.dec = Decoder(nc_img, nc_cond, n_filter, dim_key, dim_val)

    def forward(self, x, in_cond, out_cond):
        key, val = self.enc(x, in_cond)
        img = self.dec(key, val, out_cond)
        return img

class Encoder(nn.Module):
    def __init__(self, nc_img=3, nc_cond=1, n_filter=32, n_key=8, dim_key=4, dim_val=4):
        super(Encoder, self).__init__()
        self.n_key = n_key
        self.dim_key = dim_key
        self.dim_val = dim_val
        self.encoder = nn.Sequential(
            nn.Conv2d(nc_img + nc_cond, n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_filter),
            nn.Conv2d(n_filter, n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_filter),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(n_filter, n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_filter),
            nn.Conv2d(n_filter, 2*n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(2*n_filter, 2*n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),
            nn.Conv2d(2*n_filter, 4*n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*n_filter),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(4*n_filter, 4*n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*n_filter),
            nn.Conv2d(4*n_filter, n_key * (dim_key + dim_val), 3, stride=1, padding=1),
            nn.AvgPool2d(8, stride=1)
        )

    def forward(self, x, cond):
        x = torch.cat((x, cond), dim=1)
        x = self.encoder(x)
        x = x.view(-1, self.n_key, (self.dim_key + self.dim_val))
        key, val = torch.split(x, [self.dim_key, self.dim_val], dim=2)

        return key, val


class Decoder(nn.Module):
    def __init__(self, nc_img=3, nc_cond=1, n_filter=32, dim_key=4, dim_val=4):
        super(Decoder, self).__init__()
        self.attention2d = Attention2d()
        self.qgen = QueryGenerator(nc_cond=nc_cond, n_filter=n_filter, dim_key=dim_key)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim_val + n_filter, 2*n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),
            nn.ConvTranspose2d(2*n_filter, 2*n_filter, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),

            nn.Conv2d(2*n_filter, 4*n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*n_filter),
            nn.ConvTranspose2d(4*n_filter, 4*n_filter, 4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*n_filter),

            nn.Conv2d(4*n_filter, nc_img, 3, stride=1, padding=1)
        )

    def forward(self, key, value, cond):
        query, ds_cond = self.qgen(cond) # return query and downsampled condition
        att_val = self.attention2d(query, key, value)

        combined = torch.cat((att_val, ds_cond), dim=1)
        logit = self.decoder(combined)
        img = torch.sigmoid(logit)
        return img






class Attention2d(nn.Module):
    def __init__(self):
        super(Attention2d, self).__init__()

    def forward(self, query, key, value):
        dim_val = value.size(2)
        dim_key = query.size(1)
        height, width = query.size(2), query.size(3)
        query = query.view(-1, dim_key, height*width)
        attn = F.softmax(torch.bmm(key, query), dim=1) # bs, num_key, height*width
        #value : bs, num_key, dim_val
        att_value = torch.bmm(torch.transpose(value, 1, 2), attn)

        return att_value.view(-1, dim_val, height, width)

class QueryGenerator(nn.Module):
    def __init__(self, nc_cond=1, n_filter=32, dim_key=4):
        super(QueryGenerator, self).__init__()
        self.dim_key = dim_key
        self.n_filter = n_filter
        self.generator = nn.Sequential(
            nn.Conv2d(nc_cond, n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_filter),
            nn.Conv2d(n_filter, 2*n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(2*n_filter, 2*n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),
            nn.Conv2d(2*n_filter, n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_filter),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(n_filter, dim_key + n_filter, 3, stride=1, padding=1)

        )

    def forward(self, cond):
        key_cond = self.generator(cond)
        key, cond = torch.split(key_cond, (self.dim_key, self.n_filter), dim=1)
        return key, cond

class Discriminator(nn.Module):
    def __init__(self, nc_img=3, nc_cond=1, n_filter=32):
        super(Discriminator, self).__init__()
        self.dim_fc = 4*n_filter
        self.discriminator = nn.Sequential(
            nn.Conv2d(nc_img + nc_cond, n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_filter),
            nn.Conv2d(n_filter, 2*n_filter, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),

            nn.Conv2d(2*n_filter, 2*n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*n_filter),

            nn.Conv2d(2*n_filter, 4*n_filter, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*n_filter),

            nn.Conv2d(4*n_filter, 4*n_filter, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*n_filter),
            nn.Conv2d(4*n_filter, 4*n_filter, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*n_filter),

            nn.Conv2d(4*n_filter, 4*n_filter, 3, stride=1, padding=1),
            nn.AvgPool2d(8, stride=1),
        )
        self.fc = nn.Linear(self.dim_fc, 1)

    def forward(self, x, cond):
        x = torch.cat((x, cond), dim=1)
        x = self.discriminator(x)
        x = x.view(-1, self.dim_fc)
        out = self.fc(x)
        return torch.sigmoid(out)




if __name__ == "__main__":
    from dataloader import ImgDatasets

    dset = ImgDatasets("dataset/cars")
    loader = torch.utils.data.DataLoader(dset, batch_size=4,
                            shuffle=True)

    (x, in_cond), (y, out_cond) = next(iter(loader))

    # enc = Encoder()
    # dec = Decoder()
    #
    # key, val = enc(x, in_cond)
    # restore = dec(key, val, out_cond)
    # print(restore.shape)

    disc = Discriminator()
    y = disc(x, in_cond)
    pass