import torch
import torchvision
from model import AutoEncoder
from dataloader import ImgDatasets
from config import *

class Inferencer:
    def __init__(self, model):
        self.model = model

    def load_model(self, load_path):
        state = torch.load(load_path)
        self.model.load_state_dict(state['AE'])
        print('model loaded')

    def inference(self, x, in_cond, out_cond, idx=0, save_path="inference_data/test.png"):
        reconstructed = self.reconstruct(x, in_cond, out_cond)
        transferred = self.condition_transfer(x, in_cond, out_cond, idx=idx)

        nrow = int(SAMPLE_SIZE**0.5)
        original = torchvision.utils.make_grid(x.to('cpu'), nrow=nrow, padding=2, pad_value=255)
        reconstructed = torchvision.utils.make_grid(reconstructed, nrow=nrow, padding=2, pad_value=255)
        transferred = torchvision.utils.make_grid(transferred, nrow=nrow, padding=2, pad_value=255)

        combined = torchvision.utils.make_grid([original, reconstructed, transferred], padding=4)
        torchvision.utils.save_image(combined, save_path)

    @torch.no_grad()
    def reconstruct(self, x, in_cond, out_cond):
        img = self.model(x, in_cond, out_cond)
        return img.to('cpu')

    @torch.no_grad()
    def condition_transfer(self, x, in_cond, out_cond, idx=0):
        B = out_cond.size(0)
        target = [out_cond[idx] for _ in range(B)]
        target = torch.stack(target)
        img = self.model(x, in_cond, target)
        return img.to('cpu')


if __name__ == "__main__":
    model = AutoEncoder(nc_img, nc_cond, n_filter, n_key, dim_key, dim_val)
    inferencer = Inferencer(model)
    inferencer.load_model(LOAD_PATH)

    test_set = ImgDatasets(DATA_ROOT, mode='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=SAMPLE_SIZE)

    (x, in_cond), (y, out_cond) = next(iter(test_loader))

    inferencer.inference(x, in_cond, out_cond, idx=TRANSFER_IDX, save_path="inference_data/best.png")




