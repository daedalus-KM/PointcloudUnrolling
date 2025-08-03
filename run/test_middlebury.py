import mat73
import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
import time
os.sys.path.append("../")
from model.vanilla import Model
from gen_initial_multiscale_2obj import gen_initial_multiscale, shift_h, get_IF
from compute_uncertainty_2obj import *


use_cuda = True if torch.cuda.is_available() else False

def image_to_pcl(img, szpixel = 0.001, gammaz_meter = 1024*0.003):
    #------------------------------------------------
    # img : [B x C x H x W]
    # returns : [N x 3]
    #------------------------------------------------
    B, L, H, W = img.shape
    print("@@", img.shape)

    pp = torch.zeros([B, H*W, L], dtype=torch.float32)
    grid_xy = None;
    img_tr = img.reshape(B, L, H*W).transpose(2,1)

    xyscale = 8.0

    grid_xy = torch.zeros([H*W, 2], dtype=torch.float32)
    grid = torch.meshgrid(torch.arange(H),torch.arange(W))
    grid_xy[:,0] = grid[0].flipud().reshape(-1) * (szpixel*xyscale) #.expand(B, H*W).reshape(B, H*W) 
    grid_xy[:,1] = grid[1].reshape(-1) * (szpixel*xyscale) #.expand(B, H*W).reshape(B, H*W)

    pp[:,:,:] = img_tr * gammaz_meter
    
    pc = pp.clone()
    pc[:, :, ::2] = pp[:, :, :int(L/2)]
    pc[:, :, 1::2] = pp[:, :, int(L/2):]

    return pc, grid_xy 

def main():
    f = './Art_4.0_4.0.mat'
    fdict = scipy.io.loadmat(f)
    tof = torch.FloatTensor(fdict["Y"])

    use_cuda = False
    if torch.cuda.is_available():
        print("use cuda")
        use_cuda = True


    use_halfsize = False
    if use_halfsize:
        tof = tof[0:-1:2, 0:-1:2, :]
    
    H, W, T = tof.shape
    L = 8 #4 scale * 2 target
    irf_ = scipy.io.loadmat("./irf/F_real2_100s.mat")["F"][:, 99] 


    t0 = time.time()



    IF = get_IF(irf_, T, (T - irf_.shape[0])//2)
    IF_mid = IF[:,int(T//2)]
    h1 = shift_h(IF_mid, T)
    trailing = np.argmax(h1 < 1e-5) - 1 # right side of IRF
    attack = np.argmax(np.flip(h1, 0) < 1e-5) - 1 # left side of IRF


    h1_inp = torch.FloatTensor(np.flip(h1).copy())

    depths, _ = gen_initial_multiscale(tof, h1_inp, attack=attack, trailing=trailing, use_cuda = use_cuda)
    depths = depths.unsqueeze(0)
    depths, grid_xy = image_to_pcl(depths)
    pc = torch.cat([grid_xy, depths.squeeze(0)], dim=1).unsqueeze(0)
    del tof

    with torch.no_grad():
        model = Model(nscale=L, att=10, k=6, nblock=3, nconv=3).cuda()

        if use_cuda:
            torch.cuda.empty_cache()
            model.load_state_dict(torch.load("baseline.pth")['model_state_dict'])
            model.cuda()
            pc = pc.cuda()
        else:
            model.load_state_dict(torch.load("baseline.pth", map_location=torch.device('cpu'))['model_state_dict'])
        model.eval()
        out = model(pc, debug=1) # if you don't need to compute uncertainty, debug=False 

    t1 = time.time()
    print("@ elapsed time:", t1 - t0)

    pc_final = out[0][-1].cpu()

    if os.path.isdir("../result/") == False:
        os.makedirs("../result/")

    pc_final = pc_final.numpy()

    fpeak = pc_final[..., 0][0].copy().astype(float) #First peak
    speak = pc_final[..., 1][0].copy().astype(float) #Second peak

    fpeak = np.clip(fpeak, 0.2, 0.7)

    speak -= 400*0.003 #translation to save image
    speak = np.clip(speak, 0.2, 0.7)

    plt.imsave(f"../result/depth_first.png", fpeak.reshape([H,W]))
    plt.imsave(f"../result/depth_second.png", speak.reshape([H,W]))


    uncertainties = compute_uncertainty_2obj(out, H, W, nscale = L)
    uncertainty_mean = uncertainties.mean()
    print(f"uncertainty : {uncertainty_mean:.4f}")

    uncertainties = np.clip(uncertainties, 0, 0.01)

    plt.imsave(f'../result/depth_first_uncertainty_f.png', uncertainties[0, :, 0].reshape(H, W), cmap = 'gray')
    plt.imsave(f'../result/depth_first_uncertainty_s.png', uncertainties[0, :, 1].reshape(H, W), cmap = 'gray')


    exit()


    ###----


    to_meter = T * 0.003
    plt.imshow(depth_final[0,0,:,:]*to_meter); plt.clim(0.2, 0.8); plt.colorbar()
    plt.savefig("../result/depth_final.png"); plt.clf()

    scipy.io.savemat(f"../result/depth.mat", {"depth":depth_final[0,0,:,:]*to_meter })
    print("The result is saved in the folder ../result/")
    
    del depths
    # for i in range(model.nblock):
    #     del out[0][0]

    # from compute_uncertainty import compute_uncertainty
    # uncertainty = compute_uncertainty(out)

if __name__ == "__main__":
    main()
    main()