import numpy as np
import os
import nibabel as nib
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
path1 = '/home/super_mudi/Data/subject1/HR/MB_Re_t_moco_registered_applytopup_resized.nii.gz'

path2 = '/home/super_mudi/Data/subject1/LR/MB_Re_t_moco_registered_applytopup_isotropic_voxcor.nii.gz'

path3 = '/home/super_mudi/Data/subject2/HR/MB_Re_t_moco_registered_applytopup_resized.nii.gz'

path4 = '/home/super_mudi/Data/subject2/LR/MB_Re_t_moco_registered_applytopup_isotropic_voxcor.nii.gz'

path5 = '/home/super_mudi/Data/subject3/HR/MB_Re_t_moco_registered_applytopup.nii.gz'

path6 = '/home/super_mudi/Data/subject3/LR/MB_Re_t_moco_registered_applytopup_isotropic_voxcor.nii.gz'

path_test = '/home/super_mudi/Data/Test/MB_Re_t_moco_registered_applytopup_isotropic_voxcor.nii.gz'

data_HR = nib.load(path1)
data_HR = data_HR.get_fdata()
data_LR = nib.load(path2)
data_LR = data_LR.get_fdata()


out_path_HR = './Data_all/HR/'
k =0
subject = 2
for i in range(0,data_HR.shape[3]):
    for j in range(0,data_HR.shape[2]):
        arr = data_HR[:,:,j,i]
        #HR_image = Image.fromarray(arr)
        #HR_image = HR_image.convert('L')
        #HR_image.save(out_path_HR+str(subject)+'_'+str(k)+'_'+str(j)+'_'+str(i)+'.png')
        cv2.imwrite(out_path_HR+str(subject)+'_'+str(k)+'_'+str(j)+'_'+str(i)+'.png',arr)


out_path_LR = './Data_all/LR/'
subject = 2
k =0
for i in range(0,data_LR.shape[3]):
    s=0
    for j in range(0,data_LR.shape[2]):
        arr = data_LR[:,:,j,i]
        #LR_image = Image.fromarray(arr)
        #LR_image = LR_image.convert('L')
        if j==0:
            s=s+1
        else:
            s=s+2
        #LR_image.save(out_path_LR+str(subject)+'_'+str(k)+'_'+str(s)+'_'+str(i)+'.png')
        cv2.imwrite(out_path_LR+str(subject)+'_'+str(k)+'_'+str(s)+'_'+str(i)+'.png',arr)
"""
out_path_HR = './Data_all/HR_V/'
subject = 2
k =1
for i in range(0,data_HR.shape[3]):
    for j in range(0,data_HR.shape[1]):
        arr = data_HR[:,j,:,i]
        #HR_image = Image.fromarray(arr)
        #HR_image = HR_image.convert('L')
        #HR_image.save(out_path_HR+str(subject)+'_'+str(k)+'_'+str(j)+'_'+str(i)+'.png')
        cv2.imwrite(out_path_HR+str(subject)+'_'+str(k)+'_'+str(j)+'_'+str(i)+'.png',arr)


out_path_LR = './Data_all/test/LR/'
subject = 6
k =1
for i in range(0,data_LR.shape[3]):
    s=0
    for j in range(0,data_LR.shape[1]):
        arr = data_LR[:,j,:,i]
        #LR_image = Image.fromarray(arr)
        #LR_image = LR_image.convert('L')
        if j==0:
            s=s+1
        else:
            s=s+2
        #LR_image.save(out_path_LR+str(subject)+'_'+str(k)+'_'+str(s)+'_'+str(i)+'.png')
        cv2.imwrite(out_path_LR+str(subject)+'_'+str(k)+'_'+str(s)+'_'+str(i)+'.png',arr)


out_path_HR = './Data_all/HR_V/'
subject = 2
k =2
for i in range(0,data_HR.shape[3]):
    for j in range(0,data_HR.shape[0]):
        arr = data_HR[j,:,:,i]
        #HR_image = Image.fromarray(arr)
        #HR_image = HR_image.convert('L')
        #HR_image.save(out_path_HR+str(subject)+'_'+str(k)+'_'+str(j)+'_'+str(i)+'.png')
        cv2.imwrite(out_path_HR+str(subject)+'_'+str(k)+'_'+str(j)+'_'+str(i)+'.png',arr)


out_path_LR = './Data_all/test/LR/'
subject = 6
k =2
for i in range(0,data_LR.shape[3]):
    s=0
    for j in range(0,data_LR.shape[0]):
        arr = data_LR[j,:,:,i]
        #LR_image = Image.fromarray(arr)
        #LR_image = LR_image.convert('L')
        if j==0:
            s=s+1
        else:
            s=s+2
        #LR_image.save(out_path_LR+str(subject)+'_'+str(k)+'_'+str(s)+'_'+str(i)+'.png')
        cv2.imwrite(out_path_LR+str(subject)+'_'+str(k)+'_'+str(s)+'_'+str(i)+'.png',arr)

"""
path_train_HR = "./Data_all/train/HR/"
path_train_LR = "./Data_all/train/LR/"
path_val_HR = "./Data_all/val/HR/"
path_val_LR = "./Data_all/val/LR/"
path_test_LR = "./Data_all/test_LR/"
import os, random
import shutil


destination1 = path_train_HR
destination2 = path_train_LR
destination3 = path_val_HR
destination4 = path_val_LR
destination5 = path_test_LR

random.seed(42)
all_HR_img = os.listdir(out_path_HR)
all_LR_img = os.listdir(out_path_LR)

images = random.sample(all_LR_img,200)
indecies = []
k = 0
for j in images:
    indecies.append(all_LR_img.index(j))


for i in indecies:
    img_LR = all_LR_img[i]
    img_HR = img_LR
    #print(img_LR)
    #print(img_HR)
    shutil.move(out_path_HR+img_HR,destination1)
    shutil.move(out_path_LR+img_LR,destination2)
'''
# ######### val images ###########

random.seed(42)
all_HR_img = os.listdir(out_path_HR)
all_LR_img = os.listdir(out_path_LR)
images = random.sample(all_LR_img,50)
indecies = []
k = 0
for j in images:
    indecies.append(all_LR_img.index(j))


for i in indecies:
    img_LR = all_LR_img[i]
    img_HR = img_LR
    shutil.move(out_path_HR+img_HR,destination3)
    shutil.move(out_path_LR+img_LR,destination4)



'''
print("Done :)")
