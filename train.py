
from networks import U_Net
from networks import init_weights
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import dataloader
import losses
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import dill
from torchvision.utils import save_image

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
LR = 0.0001
WORKERS = 8
DEVICE = 'cuda:0'
SCALE = 1.0
LR_DECAY = 0.5
LR_STEP = 10
TRAIN_LR = '/home/varsha/US_MRI/DRN_withoutkernel/srdata/DATA/noisy_1c/Size_128/'
TRAIN_HR = '/home/varsha/US_MRI/DRN_withoutkernel/srdata/DATA/clean_1c/'
VAL_LR = '/home/varsha/US_MRI/DRN_withoutkernel/srdata/benchmark/TEST/noisy_1c/Size_128/'
VAL_HR = '/home/varsha/US_MRI/DRN_withoutkernel/srdata/benchmark/TEST/clean_1c/'
EXP_NO = 1
LOAD_CHECKPOINT = None #'checkpoint.pth.tar'
TENSORBOARD_LOGDIR = f'{EXP_NO:02d}-tboard'
END_EPOCH_SAVE_SAMPLES_PATH = f'{EXP_NO:02d}-epoch_end_samples'
WEIGHTS_SAVE_PATH = f'{EXP_NO:02d}-weights'
MSE_LOSS_WEIGHT = 1.0
BATCHES_TO_SAVE = 1
SAVE_EVERY = 10
EPOCHS = 100
## keep tracking the losses
class Bookkeeping:
    def __init__(self, tensorboard_log_path=None, suffix=''):
        self.loss_names = ['mse']
        self.genesis()
        ## initialize tensorboard objects
        self.tboard = dict()
        if tensorboard_log_path is not None:
            if not os.path.exists(tensorboard_log_path):
                os.mkdir(tensorboard_log_path)
            for name in self.loss_names:
                self.tboard[name] = SummaryWriter(os.path.join(tensorboard_log_path, name + '_' + suffix))
            
    def genesis(self):
        self.losses = {key: 0 for key in self.loss_names}
        self.count = 0

    def update(self, **kwargs):
        for key in kwargs:
            self.losses[key]+=kwargs[key]
        self.count +=1

    def reset(self):
        self.genesis()

    def get_avg_losses(self):
        avg_losses = dict()
        for key in self.loss_names:
            avg_losses[key] = self.losses[key] / self.count
        return avg_losses

    def update_tensorboard(self, epoch):
        avg_losses = self.get_avg_losses()
        for key in self.loss_names:
            self.tboard[key].add_scalar(key, avg_losses[key], epoch)

def save_checkpoint(epoch, Unet, best_metrics, optimizer, lr_scheduler, filename='checkpoint.pth.tar'):
    state = {'epoch': epoch, 'Unet_state_dict': Unet.state_dict(),
             'best_metrics': best_metrics, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    torch.save(state, filename, pickle_module=dill)


def save_images(path, LR_images,  SR_images, HR_images, epoch, batchid):

    images_path = os.path.join(path, f'{epoch:04d}')

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    for i, tensor in enumerate(LR_images):
        #np.save(f'{images_path}/{batchid}_{i:02d}_noisy.jpg', tensor)
        save_image(tensor, f'{images_path}/{batchid}_{i:02d}_LR.jpg')
    for i, tensor in enumerate(SR_images):
        tensor = tensor.cpu()
        #np.save(f'{images_path}/{batchid}_{i:02d}_denoised.jpg', tensor)
        save_image(tensor,f'{images_path}/{batchid}_{i:02d}_SR.jpg')
    for i, tensor in enumerate(HR_images):
        tensor = tensor.cpu()
        #np.save(f'{images_path}/{batchid}_{i:02d}_clean.jpg', tensor)
        save_image(tensor,f'{images_path}/{batchid}_{i:02d}_HR.jpg')




def pbar_desc(label, epoch, total_epochs, loss_val, losses):
    return f'{label}:{epoch:04d}/{total_epochs} | {loss_val: .3f} | mse: {losses["mse"]}'

def train(Unet,trn_dl,epoch,epochs,MSE,opt,train_losses):
    Unet.train()
    t_pbar = tqdm(trn_dl, desc=pbar_desc('train',epoch,epochs,0.0,{'mse':0.0}))
    for lr_imgs, hr_imgs in t_pbar:
        lr_imgs = lr_imgs.to(DEVICE)
        hr_imgs = hr_imgs.to(DEVICE)
        
        sr_imgs = Unet(lr_imgs)
        #print(lr_imgs.shape)
        #print(hr_imgs.shape)        
        #print(sr_imgs.shape)
        mse_loss = MSE(hr_imgs,sr_imgs)
        mse_display = mse_loss.detach().cpu().item()

        loss = MSE_LOSS_WEIGHT * mse_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        t_pbar.set_description(pbar_desc('train',epoch,EPOCHS,loss.item(), {'mse': round(mse_display,3)}))

        train_losses.update(mse = mse_loss.item())


def evaluate(Unet,val_dl,epoch,epochs,MSE,val_losses,best_val_loss):
    Unet.eval()
    v_pbar = tqdm(val_dl,desc=pbar_desc('valid', epoch , epochs, 0.0, {'mse':0.0}))
    with torch.no_grad():
        e = 0.0
        for lr_imgs,hr_imgs in v_pbar:
            e = e+1
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)
            sr_imgs = Unet(lr_imgs)

            mse_loss = MSE(hr_imgs,sr_imgs)
            mse_display = mse_loss.detach().cpu().item()

            loss = MSE_LOSS_WEIGHT * mse_loss

            val_losses.update(mse = mse_loss.item())
            v_pbar.set_description(pbar_desc('valid',epoch, EPOCHS , loss.item(), {'mse':round(mse_display,3)}))
            if e == 10:
                save_images(END_EPOCH_SAVE_SAMPLES_PATH, lr_imgs.detach().cpu(), sr_imgs.detach().cpu(), hr_imgs, epoch, e)
    ## save best model weights
    avg_val_losses = val_losses.get_avg_losses()
    avg_val_loss = avg_val_losses['mse']
    if avg_val_loss < best_val_loss or epoch % SAVE_EVERY == 0:
        best_val_loss = loss.item()
        torch.save(Unet.state_dict(), f'{WEIGHTS_SAVE_PATH}/{EXP_NO:02d}-epoch-{epoch:04d}_tota-loss-{avg_val_loss:.3f}.pth')
    return best_val_loss



def main():
    transforms = T.Compose([T.ToTensor(),])
    trn_ds = dataloader.USDataset(TRAIN_LR, TRAIN_HR,transforms )
    trn_dl = DataLoader(trn_ds, TRAIN_BATCH_SIZE, shuffle = False, num_workers = WORKERS)
    val_ds = dataloader.USDataset(VAL_LR, VAL_HR,transforms )
    val_dl = DataLoader(val_ds, VAL_BATCH_SIZE, shuffle = False, num_workers = WORKERS)
    start_epoch = 1
    best_val_loss = float('inf')

    Unet = U_Net(1,1)
    print(Unet)
    print('U-Net parameters:', sum(p.numel() for p in Unet.parameters()))
    init_weights(Unet)
    opt = optim.Adam(Unet.parameters(), lr = LR)
    sched = optim.lr_scheduler.StepLR(opt, LR_STEP, gamma=LR_DECAY)

    if not os.path.exists(WEIGHTS_SAVE_PATH):
        os.mkdir(WEIGHTS_SAVE_PATH)

    if LOAD_CHECKPOINT is not None:
        checkpoint = torch.load(LOAD_CHECKPOINT, pickle_module = dill)
        start_epoch = checkpoint['epoch']
        Unet.load_state_dict(checkpoint['Unet_state_dict'])
        opt = checkpoint['optimizer']
        sched = checkpoint['lr_scheduler']

    Unet.to(DEVICE)

    MSE = nn.MSELoss()
    L1 = nn.L1Loss()
#    MSE.to(DEVICE)
    L1.to(DEVICE)
    
    train_losses = Bookkeeping(TENSORBOARD_LOGDIR, suffix='trn')
    val_losses = Bookkeeping(TENSORBOARD_LOGDIR, suffix='val')

    for epoch in range(start_epoch, EPOCHS+1):
        ## training loop
        train(Unet, trn_dl,epoch,EPOCHS,MSE,opt,train_losses)

        ## validation loop
        best_val_loss = evaluate(Unet, val_dl, epoch, EPOCHS, MSE, val_losses, best_val_loss)
        sched.step()

        save_checkpoint(epoch, Unet, None, opt, sched, )

        train_losses.update_tensorboard(epoch)
        val_losses.update_tensorboard(epoch)

        ## Reset all losses for the new epoch
        train_losses.reset()
        val_losses.reset()


if __name__=='__main__':
    main()


