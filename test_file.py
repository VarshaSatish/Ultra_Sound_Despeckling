import torch
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path
from networks import U_Net
import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2
import numpy as np
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compare_ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image = Image.open(Path('/home/varsha/US_MRI/DRN_withoutkernel/srdata/benchmark/TEST/noisy_1c/Size_128/00001.jpg'))
Ref_image = Image.open(Path('/home/varsha/US_MRI/DRN_withoutkernel/srdata/benchmark/TEST/clean_1c/00001.jpg'))

transforms = T.Compose([T.ToTensor(),])
inputs = transforms(image)

model = U_Net(1,1)
inputs = inputs.to(device)

checkpoint = torch.load('/home/varsha/US_MRI/U_Net/01-weights/01-epoch-0010_tota-loss-0.005.pth')
# checkpoint = torch.load(Path('/home/varsha/US_MRI/U_Net/02-weights/02-epoch-0011_tota-loss-0.048.pth'))
# checkpoint = torch.load(Path('/home/varsha/US_MRI/U_Net/03-weights/03-epoch-0100_tota-loss-0.372.pth'))
# print(checkpoint.keys())
# model.load_state_dict(checkpoint['Unet_state_dict'])
model.load_state_dict(checkpoint)
model.cuda() 
model.eval()
inputs = inputs.to(device).view(1,1,128,128)
output = model(inputs).squeeze(0)
output = output.squeeze(0).detach().cpu()
# print(output.shape)
output1 = np.array(output)
out = ( np.array(output1) - np.amin(output1) ) / ( np.amax(output1) - np.amin(output1))

final_image = np.dot(np.array(out), 255)
ref_image = np.array(Ref_image, dtype=float)
# print(final_image.shape)
# print(ref_image.shape)


# print(np.amin(ref_image))
# print(np.amin(final_image))
# print(np.amax(ref_image))
# print(np.amax(final_image))
# exit()
# (score, diff) = compare_ssim(ref_image, final_image, full=True)
# diff = (diff * 255).astype("uint8")
# print("SSIM: {}".format(score))

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from math import log10, sqrt
import cv2
import numpy as np
  
def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


# m = mse(ref_image, final_image)
# s = ssim(ref_image, final_image)
# p = PSNR(ref_image, final_image)

# print("MSE ",m)
# print("SSIM ",s)
# print("PSNR ",p)
# save_image(output,'unet_output_ssim.jpg')

