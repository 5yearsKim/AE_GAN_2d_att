import torch
import os
from torchvision import transforms
from PIL import Image


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

class ImgDatasets(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode="train"):
        self.mode = mode
        if mode == "train":
            self.img_files = files_to_list(os.path.join(root_dir, "train.txt"))
        else:
            self.img_files = files_to_list(os.path.join(root_dir, "test.txt"))
        self.root_dir = root_dir

        self.base_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip()
        ])
        self.gray_transform = transforms.Compose([
            transforms.Grayscale()
        ])
        self.ToTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, "imgs", self.img_files[index])
        image = Image.open(img_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.base_transform(image)
        gray_img = self.gray_transform(image)
        image, gray_img = self.ToTensor(image), self.ToTensor(gray_img)
        if self.mode == 'train':
            noise = torch.randn_like(image) * 0.003
            image += noise
        return (image, gray_img), (image, gray_img)



if __name__ == "__main__":
    file_from = "dataset/cars/train.txt"

    dset = ImgDatasets("dataset/cars", mode='train')

    (x, in_cond), _ = dset[4]

    transforms.ToPILImage()(x).show()
