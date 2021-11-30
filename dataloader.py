import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms
import random
from PIL import Image

class USDataset(Dataset):
    def __init__(self, noisy_path, clean_path, transforms):
        super().__init__()
        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.noisy_list = os.listdir(self.noisy_path)
        self.clean_list = os.listdir(self.clean_path)

        self.transforms = transforms

    def __getitem__(self, index):
        noisy_img = Image.open(self.noisy_path + self.noisy_list[index])
        clean_img = Image.open(self.clean_path + self.clean_list[index])
        noisy_img = noisy_img.convert('L')
        clean_img = clean_img.convert('L')

        if self.transforms is not None:
            noisy_img = self.transforms(noisy_img)
            clean_img = self.transforms(clean_img)

        return (noisy_img, clean_img)

    def __len__(self):
        return len(self.noisy_list)

