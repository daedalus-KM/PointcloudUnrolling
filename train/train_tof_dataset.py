import torch
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
import os
import h5py
from glob import glob
import torch.nn.functional as F

# from .apss import add_curvature

from torch.utils.data import Dataset, DataLoader

def normalize_per_channel(d):
    dd = d.reshape([d.shape[0], -1])
    dmax = torch.max(dd, dim=1, keepdims=True)[0].unsqueeze(2)
    dmin = torch.min(dd, dim=1, keepdims=True)[0].unsqueeze(2)
    return (d - dmin) / (dmax - dmin)

def get_refl_scales(r, nscale=3):
    rr = torch.reshape(r[:,nscale,:,:], [r.shape[0], -1])
    # rr = r.reshape([r.shape[:], -1])
    scales = torch.max(rr, dim=1, keepdims=True)[0]
    return scales.unsqueeze(2).unsqueeze(3)


###Input : f S 순 (scale 상관없음)
def image_to_pcl(img, szpixel, gammaz_meter, nheight=0, get_xy=False, istrain=True):
    #------------------------------------------------
    # img : [B x C x H x W]
    # returns : [N x 3]
    #------------------------------------------------
    if istrain:
        if nheight != 128:
            img = img[:,:,:nheight,:nheight]
        else:
            img1 = img[:,:,:nheight,:nheight]
            img2 = img[:,:,:nheight,nheight:]
            img3 = img[:,:,nheight:,:nheight]
            img4 = img[:,:,nheight:,nheight:]
            img = torch.cat([img1, img2, img3, img4], dim=0) #4*B x L x H x W
    # img: 0 ~ 1 사이의 값


    B, L, H, W = img.shape
    print("@@", img.shape)

    pp = torch.zeros([B, H*W, L], dtype=torch.float32)
    grid_xy = None;
    img_tr = img.reshape(B, L, H*W).transpose(2,1)

    xyscale = 8.0
    # xyscale = 0.8

    if get_xy:
        grid_xy = torch.zeros([H*W, 2], dtype=torch.float32)
        grid = torch.meshgrid(torch.arange(H),torch.arange(W))
        grid_xy[:,0] = grid[0].flipud().reshape(-1) * (szpixel*xyscale) #.expand(B, H*W).reshape(B, H*W) 
        grid_xy[:,1] = grid[1].reshape(-1) * (szpixel*xyscale) #.expand(B, H*W).reshape(B, H*W)

        pp[:,:,:] = img_tr * gammaz_meter
        
        pc = pp.clone()
        pc[:, :, ::2] = pp[:, :, :int(L/2)]
        pc[:, :, 1::2] = pp[:, :, int(L/2):]

        return pc, grid_xy 

         
    pp[:,:,:] = img_tr * gammaz_meter

    return pp, grid_xy


class TofDataset(Dataset): 
    def __init__(self, fname, nscale, istrain=True, pnorm=0, feat=0, nheight=128,
        szpixel=0.001, gammaz_meter=1024*0.003):
        """
        Args:
            fname : h5 file name
            
        """
        self.nheight = nheight

        with h5py.File(fname) as f:
            if nscale > 8:
                self.depths = torch.tensor(np.array(f['depths'])[:, :10, :, :]) 
            else:
                self.depths = torch.tensor(np.array(f['depths'])[:, :8, :, :]) 


            print("@ self.depths shape:", self.depths.shape)

            self.depths_gt = None
            self.depths_gt2 = None 
            self.depths_gts = None 

            self.feat = feat
            
            self.istrain = istrain
            self.available_gt = False
            if istrain:
                self.available_gt = True
                self.depths_gt = torch.tensor(np.array(f['depths_gt'])) 
                self.depths_gt2 = torch.tensor(np.array(f['depths_gt2'])) 

                if len(self.depths_gt.shape) == 3:
                    self.depths_gt = self.depths_gt.unsqueeze(1)
                    self.depths_gt2 = self.depths_gt2.unsqueeze(1)
    
                self.depths_gt, _ = image_to_pcl(self.depths_gt, szpixel, gammaz_meter, nheight=nheight, get_xy=False, istrain=istrain)
                self.depths_gt2, _ = image_to_pcl(self.depths_gt2, szpixel, gammaz_meter, nheight=nheight, get_xy=False, istrain=istrain)
                self.depths_gts = torch.cat([self.depths_gt, self.depths_gt2], dim=2) 

        if len(self.depths.shape) == 3:
            L, H, W = self.depths.shape[0], self.depths.shape[1], self.depths.shape[2]
            self.depths = torch.reshape(self.depths, [-1, L, H, W])

        self.nscale = nscale
        self.pnorm = pnorm
        self.scales = None 

        self.height = self.depths.shape[2]
        self.width = self.depths.shape[3]
        
        self.depths, self.grid_xy = image_to_pcl(self.depths, szpixel, gammaz_meter, nheight=nheight, get_xy=True, istrain=istrain)
        self.depths = self.depths[:, :, :nscale]
 
        self.K_unfold = 2

        if istrain == False:
            self.depths = torch.cat([self.grid_xy.reshape([1, -1, 2]), self.depths], dim=2)


    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        d = self.depths[idx,:]
        d_gt = self.depths_gts[idx,:]

        r = 0.0
        r_gt = 0.0

        d = torch.cat([self.grid_xy, d], dim=1)
        return d, r, d_gt, r_gt, idx
