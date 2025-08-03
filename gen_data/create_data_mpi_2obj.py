# The goal is to generate training data from Middlebury dataset
# Output: depth [nscale x H x W], reflectivity [nscale x K x H x W]
# nscale = number of scales
# K = number of wavelengths
# 21/23 images (436, 1024)

import numpy as np
import matplotlib.pyplot as plt
import torch

import os
import sys 

from gen_initial_multiscale_2obj import *
from scipy import io
from PIL import Image



btrain = 1

ddata = "./data/mpi/"
os.makedirs(f"{ddata}images", exist_ok=True)


#------------------------------------------------
# simulation params
#------------------------------------------------
T = 1024
T_max_val = 300

sbr = 64.0
ppp = 64.0

if len(sys.argv) > 1:
    ppp = float(sys.argv[1])
    if len(sys.argv) > 2:
        sbr = float(sys.argv[2])

nscale = 4
nirf = 2
KK = [1, 3, 7, 13] 

szratio = 1
step = 6

fname = f"train_T={T}_ppp={ppp}_sbr={sbr}_2obj.h5"

H_patch = 256
W_patch = 256
stride_patch = 48

# train_idx = [1,3,4,5,6,7,8,9]

list_depths = []
list_refls = []
list_depths_gt = []
list_refls_gt = []
list_depths_gt2 = []
idx_train = 0

#
#---------------------------------------
# construct impulse response function
#---------------------------------------
F = io.loadmat("F_real2_100s.mat")["F"]
# F = MAT.matread("F_real2_100s.mat")["F"]
F = F[:,99]
IF = get_IF(F, T, (T - F.shape[0])//2)
IF_mid = IF[:,int(T//2)]

plt.plot(IF_mid)

# prepare different impulse response functions
h1_original = IF_mid.reshape([1,-1])
h1 = shift_h(IF_mid, T)
trailing = np.argmax(h1 < 1e-5) - 1 # right side of IRF
attack = np.argmax(np.flip(h1, 0) < 1e-5) - 1 # left side of IRF
# attack depth trailing

dname = f"{ddata}train/"

import glob

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.') and f.find("mountain_1")==-1 and f.find("temple_3")==-1:
            yield f

seqs = list(listdir_nohidden(f"{ddata}training/clean_left/"))

def get_depth_refl(seq):
    fname_gt = f"{ddata}/training/clean_left/{seq}/{T}_{ppp}_{sbr}.mat"
    np.random.seed(1)

    # disp = Int.(reinterpret(UInt8, load(ddata*"/raw/"*seq*"/disp1.png")))
    disp = np.asarray(Image.open(f"{ddata}/training/disparities_viz/{seq}/frame_0001.png"))
    refl = np.asarray(Image.open(f"{ddata}/training/clean_left/{seq}/frame_0001.png").convert("L"))

    print(seq, disp.shape)
    
    disp = disp[18:-18, :]
    refl = refl[18:-18, :]

    disp0 = np.copy(disp)
    H_ori, W_ori = disp.shape

    #------------------------------------------------
    # 1. fill out outliers in disp
    #------------------------------------------------
    disp = discard_outliers(disp, step)
    disp = discard_outliers(disp, step)
    disp = discard_outliers(disp, step)
        
    for ii in range(20):
        noutliers = (disp == 0.0).sum()
        if noutliers > 0:
            print("noutliers:", noutliers)
        else:
            break
        
        disp = discard_outliers(disp, step + (ii+1)*2)
        
        if ii == 13:
            print(seq, "error")
            disp[np.where(disp <= 15)] = 16
            break        

    assert((disp == 0).sum() == 0)

    depth_ = T_max_val - disp
    
    # save images to check
    # [ ] TODO: display {disp}
    plt.imsave(f"{ddata}/images/{seq}-depth.png", depth_)

    #------------------------------------------------
    # 2. generate depth images
    #------------------------------------------------
    depth = depth_
    reflg = refl

    # scale depth to lie 50 ~ 250
    depth_quant = depth - 1 # python index starts from 0
    
    reflg = reflg / reflg.mean()
    return depth_quant, reflg


for i in range(0, len(seqs)-1, 2):
    seq = seqs[i]
    seq2 = seqs[i+1]
    print("@", seq, seq2)

    depth_quant, reflg = get_depth_refl(seq)
    H, W = reflg.shape

    depth_quant2, reflg2 = get_depth_refl(seq2)

    depth_quant2 += 400
    
    #------------------------------------------------
    # 3. generate ToF data and GT
    #------------------------------------------------
    S = np.zeros([T, H, W])

    for iy, ix in np.ndindex((H, W)):
        S[depth_quant[iy, ix], iy, ix] = reflg[iy, ix]
        S[depth_quant2[iy, ix], iy, ix] = reflg2[iy, ix]
    
    # convolution in time IF * intensity
    # [T x T] x [N x T]'
    N = H*W


    tt = IF.dot( S.reshape(1024, -1) )
    S_conv = tt.reshape([T, H, W])

    Lev_S = sbr * 0.5*ppp / ( 1 + sbr)
    Lev_B = 0.5*ppp - Lev_S 

    bg = np.random.poisson(Lev_B/T, (T, H, W)) # 
    p1 = np.random.poisson(S_conv * Lev_S)
    tof_data = p1 + bg

    print("ppp", np.mean(np.sum(tof_data, 0)))
    print("sbr", Lev_S / Lev_B)

    d_gt_n = (depth_quant + 1 ) / T
    d_gt_n2 = (depth_quant2 + 1 ) / T
    
    r_gt_n = reflg * Lev_S

    
    #------------------------------------------------
    # 4. estimate the initial depth and intensity
    #------------------------------------------------
    tof_HWT = tof_data.transpose([1,2,0])
    h1_inp = np.flip(h1).copy()

    depths, refls = gen_initial_multiscale(tof_HWT, h1_inp, nscale*nirf, KK, attack=attack, trailing=trailing)

    d_gt_n = torch.tensor(d_gt_n)
    d_gt_n2 = torch.tensor(d_gt_n2)
    r_gt_n = torch.tensor(r_gt_n)

    

    if btrain:
        for stride in [1, 2]:
            for ww in range(0, W-stride*W_patch+1, stride_patch):
                for hh in range(0, H-stride*H_patch+1, stride_patch):
                    list_depths.append(depths[:nscale*nirf*2, hh:hh+stride*H_patch:stride, ww:ww+stride*W_patch:stride])
                    # list_refls.append(refls[:nscale*nirf, hh:hh+stride*H_patch:stride, ww:ww+stride*W_patch:stride])
                    list_depths_gt.append(d_gt_n[hh:hh+stride*H_patch:stride, ww:ww+stride*W_patch:stride])
                    list_depths_gt2.append(d_gt_n2[hh:hh+stride*H_patch:stride, ww:ww+stride*W_patch:stride])
                    # list_refls_gt.append(r_gt_n[hh:hh+stride*H_patch:stride, ww:ww+stride*W_patch:stride])

#------------------------------------------------
# save training data
#-----------------------------------------------
import h5py
if btrain:
    with h5py.File(f"{ddata}/{fname}", 'w') as hf:
        hf["depths"] = torch.stack(list_depths, 0)
        # hf["refls"] = torch.stack(list_refls, 0)
        hf["depths_gt"] = torch.stack(list_depths_gt, 0).unsqueeze(1)
        hf["depths_gt2"] = torch.stack(list_depths_gt2, 0).unsqueeze(1)
        # hf["refls_gt"] = torch.stack(list_refls_gt, 0).unsqueeze(1)
