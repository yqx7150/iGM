import os
from natsort import natsorted
import cv2
import numpy as np
import torch
import torch.nn as nn
import glob
from models.cond_refinenet_dilated import CondRefineNetDilated
#from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.misc import imread,imresize
from skimage.measure import compare_psnr,compare_ssim 
__all__ = ['Test_6ch']

def write_Data(result_all,i):
    with open(os.path.join('./bedroom_6ch/',"psnr_6ch3_320000_3ch320000"+".txt"),"w+") as f:
        for i in range(len(result_all)):
            f.writelines('current image {} PSNR : '.format(i) + str(result_all[i][0]) + '    SSIM : ' + str(result_all[i][1]))
            f.write('\n')

class Test_6ch():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()

        batch_size = 1 
        samples = 2
        files_list = glob.glob('./ground turth/*.png')
        files_list = natsorted(files_list)
        length = len(files_list)
        result_all = np.zeros([101,2])
        for z,file_path in enumerate(files_list):
            img = cv2.imread(file_path)
            img2 = cv2.imread('./iGM-3C/img_{}_Rec_x_end_rgb.png'.format(z))
            img = cv2.resize(img, (128, 128))
            
            YCbCrimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
            x0 = img.copy()
            x1 = np.concatenate((img2,YCbCrimg2),2)

            original_image = img.copy()
            x0 = torch.tensor(x0.transpose(2,0,1),dtype=torch.float).unsqueeze(0) / 255.0
            x1 = torch.tensor(x1.transpose(2,0,1),dtype=torch.float).unsqueeze(0) / 255.0
            x_stack = torch.zeros([x0.shape[0]*samples,x0.shape[1],x0.shape[2],x0.shape[3]],dtype=torch.float32)      
            
            for i in range(samples):
                x_stack[i*batch_size:(i+1)*batch_size,...] = x0
            x0 = x_stack

            gray = (x0[:,0,...] + x0[:,1,...] + x0[:,2,...]).cuda()/3.0
            gray1 = (x1[:,0,...] + x1[:,1,...] + x1[:,2,...]+ x1[:,3,...] + x1[:,4,...] + x1[:,5,...]).cuda()/6.0

            gray_mixed = torch.stack([gray,gray,gray],dim=1)
            gray_mixed_1 = torch.stack([gray1,gray1,gray1,gray1,gray1,gray1],dim=1)

            x0 = nn.Parameter(torch.Tensor(samples*batch_size,6,x0.shape[2],x0.shape[3]).uniform_(-1,1)).cuda()
            x01 = x0.clone()

            step_lr=0.0003 * 0.04#bedroom 0.04   church 0.02

            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                               0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 100
            max_psnr = 0
            max_ssim = 0
            for idx, sigma in enumerate(sigmas):
                lambda_recon = 1./sigma**2
                labels = torch.ones(1, device=x0.device) * idx
                labels = labels.long()

                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                
                print('sigma = {}'.format(sigma))
                for step in range(n_steps_each):
                    print('current step %03d iter' % step)                    
                    x0_mix = (x01[:,0,...] + x01[:,1,...] + x01[:,2,...])/3.0
                    x1_mix = (x01[:,0,...] + x01[:,1,...] + x01[:,2,...] + x01[:,3,...] + x01[:,4,...] + x01[:,5,...])/6.0
                    
                    error = torch.stack([x0_mix,x0_mix,x0_mix],dim=1) - gray_mixed
                    error1 = torch.stack([x1_mix,x1_mix,x1_mix,x1_mix,x1_mix,x1_mix],dim=1) - gray_mixed_1

                    noise_x = torch.randn_like(x01) * np.sqrt(step_size * 2)

                    grad_x0 = scorenet(x01, labels).detach()
                 
                    x0 = x01 + step_size * (grad_x0)
                    x0 = x0 - 0.1 * step_size * lambda_recon * error1     #bedroom 0.1  church 1.5
                    x0[:,0:3,...] = x0[:,0:3,...] - step_size * lambda_recon * (error)

                    x0 = torch.mean(x0,dim=0)     
                    x0 = torch.stack([x0,x0],dim=0)
                    x01 = x0.clone() + noise_x
                   	
                    x_rec = x0.clone().detach().cpu().numpy().transpose(0,2,3,1)
                    
                    for j in range(x_rec.shape[0]):
                        x_rec_ = np.squeeze(x_rec[j,...])                        
                        x_rec_ycbcr2rgb = cv2.cvtColor(x_rec_[...,3:], cv2.COLOR_YCrCb2BGR)
                        x_rec_ycbcr2rgb = np.clip(x_rec_ycbcr2rgb,0,1)

                    x_rec_ycbcr2rgb = x_rec_ycbcr2rgb[np.newaxis,...]

                    x_rec = (x_rec[...,:3] + x_rec_ycbcr2rgb)/2
                    original_image = np.array(original_image,dtype = np.float32)

                    for i in range(x_rec.shape[0]):
                        psnr = compare_psnr(x_rec[i,...]*255.0,original_image,data_range=255)
                        ssim = compare_ssim(x_rec[i,...],original_image/255.0,data_range=1,multichannel=True)
                        print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)
                    if max_psnr < psnr :
                        result_all[z,0] = psnr
                        max_psnr = psnr
                        cv2.imwrite(os.path.join(self.args.image_folder, 'img_{}_Rec_6ch_finally.png'.format(z)),(x_rec[i,...]*256.0).clip(0,255).astype(np.uint8))
                        result_all[length,0] = sum(result_all[:length,0])/length
                        
                    if max_ssim < ssim:
                        result_all[z,1] = ssim
                        max_ssim = ssim
                        result_all[length,1] = sum(result_all[:length,1])/length
                    
                    write_Data(result_all,z)                 
                   
            x_save = x0.clone().detach().cpu().numpy().transpose(0,2,3,1)

            for j in range(x_save.shape[0]):
                x_save_ = np.squeeze(x_save[j,...])
                print(np.max(x_save_),np.min(x_save_))                        
                x_save_ycbcr2rgb = cv2.cvtColor(x_save_[...,3:], cv2.COLOR_YCrCb2BGR)
                print(np.max(x_save_ycbcr2rgb),np.min(x_save_ycbcr2rgb))

            x_save_ycbcr2rgb = torch.tensor(x_save_ycbcr2rgb)
            x_save_ycbcr2rgb = torch.unsqueeze(x_save_ycbcr2rgb,0)
            x_save_ycbcr2rgb = np.array(x_save_ycbcr2rgb)

            x_save = (x_save[...,:3] + x_save_ycbcr2rgb)/2
            x_save = np.array(x_save).transpose(0, 3, 1, 2)
            x_save_R = x_save[:,2:3,:,:]
            x_save_G = x_save[:,1:2,:,:]
            x_save_B = x_save[:,0:1,:,:]
            x_save = np.concatenate((x_save_R,x_save_G,x_save_B),1)

            self.write_images(torch.tensor(x_save).detach().cpu(), 'x_end.png',1,z)
    def write_images(self, x,name,n=7,z=0):
        x = x.numpy().transpose(0, 2, 3, 1)
        d = x.shape[1]
        panel = np.zeros([1*d,n*d,3],dtype=np.uint8)
        for i in range(1):
            for j in range(n):
                panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (256*(x[i*n+j])).clip(0,255).astype(np.uint8)[:,:,::-1]

        cv2.imwrite(os.path.join(self.args.image_folder, 'img_{}_Rec_6ch_'.format(z) + name), panel)
