import torch
from torch import nn
import torch.nn.functional as F
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, denoised_img, clean_img):
        Loss =  self.loss(clean_img,predicted_img)
        return Loss



