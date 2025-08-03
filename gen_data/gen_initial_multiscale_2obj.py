import torch
import numpy as np
# import jax.numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"]='expandable_segments:True'

def get_IF(F, T, leftT):  #leftT=100
    F[F < 0.01*F.max()] = 0
    h = np.concatenate([  np.zeros(leftT), F, np.zeros(leftT)  ])
    h /= sum(h)

    maxF = np.argmax(h)

    IF = np.zeros([len(h), len(h)])
    
    for i in range(len(h)):
        IF[:,i] = np.roll(h, i - maxF)
        # IF = IF.at[:, i].set(np.roll(h, i - maxF))
        # IF[:,i] = np.roll(h, i - maxF)

    return IF[:T, :T]
    # h = [np.zeros(leftT, 1); F; np.zeros([leftT, 1])]


"discard outliers for middlebury and mpi"
def discard_outliers(disp_, step, thresh=15, check = False, temp = False, h_range = (0, 0), w_range = (0, 0)):
    H, W = disp_.shape
    disp_ = disp_.copy()
    ii, jj = np.where(disp_ <= thresh)

    check_i = 0

    hmin, hmax = h_range
    wmin, wmax = w_range
    
    for it in range(len(ii)):
        idx0, idx1 = ii[it], jj[it]

        if temp:
            if (idx0 >= hmin) and (idx0 <= hmax):
                if (idx1 >= wmin) and (idx1 <= wmax):
                    continue

        vv = []
        for j in range(max(0, idx1-step), min(W-1, idx1+step+1)): #여기에 +1추가
            for i in range(max(0, idx0-step), min(H-1, idx0+step+1)):
                
                vv.append(disp_[i,j])

        # disp_ = disp_[idx0, idx1].set(np.median(vv))
        disp_[idx0, idx1] = np.median(vv)
        if check == True:
            check_i += 1
    
    if check == True:
        print(check_i)

    return disp_



def shift_h(F, T, shift=False):
    irf = np.zeros(T)
    irf[0:len(F)] = F
    irf = irf / np.sum(irf)
    attack = irf.argmax() + 1

    h = np.roll(irf, -attack)

    return h.copy()

def conv3d_separate(inp, K):
    # inp shape: [4, 1, T, H, W]
    
    # if K < 10:
    conv = torch.nn.functional.conv3d
    w1 = torch.ones(1, 1, K, 1, 1) / K
    w2 = torch.ones(1, 1, 1, K, 1) / K
    w3 = torch.ones(1, 1, 1, 1, K) / K
    w1 = w1.cuda()
    w2 = w2.cuda()
    w3 = w3.cuda()

    if inp.device != torch.device('cpu'):        
        return conv(conv(conv(inp, w1, padding="same"), w2, padding="same"), w3, padding="same")
    
    # for saving memory
    # obsolote code
    else:
        print("saving memory")

        for i in range(inp.shape[0]):
            t1 = conv(inp[i:i+1,:].cuda(), w1, padding="same")
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            t1 = conv(t1, w2, padding="same")
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            inp[i:i+1,:] = conv(t1, w3, padding="same").cpu()
            torch.cuda.empty_cache(); torch.cuda.synchronize()

            # inp[i:i+1,:] = conv(inp[i:i+1,:].cuda(), w2, padding="same").cpu()
            # inp[i:i+1,:] = conv(inp[i:i+1,:].cuda(), w3, padding="same").cpu()

        return inp


def conv3d_separate_cpu(inp, K):
    inp_np = inp.cpu().numpy()
    w = np.ones(K) / K
    inp_np = scipy.ndimage.convolve1d(inp_np, w, axis=2, mode='constant')
    inp_np = scipy.ndimage.convolve1d(inp_np, w, axis=3, mode='constant')
    inp_np = scipy.ndimage.convolve1d(inp_np, w, axis=4, mode='constant')
    return torch.tensor(inp_np)
    
#input (IRF와 correlation을 구한 histogram)과 filter size
def conv2d(inp, K): 
    # inp: [Tx1xHxW]
    if inp.device == torch.device('cpu'):
        inp_np = inp.numpy()
        w = np.ones(K) / K #w는 k가 3이면 0.3333 씩 3개가 담김 (1차원)

        #H축에 convolution (filter 뒤집는데 unfiorm이라 상관 x, 앞뒤(여기선 위 아래)로 zero padding추가) 
        inp_np = scipy.ndimage.convolve1d(inp_np, w, axis=2, mode='constant') 
        inp_np = scipy.ndimage.convolve1d(inp_np, w, axis=3, mode='constant') #W축에 convolution
        #계산해보니 3x3 uniform filter로 2d convolution한거랑 같음 
        return torch.tensor(inp_np) #tensor 변환해서 반환

    else:
        w2 = torch.ones(1, 1, K, 1, dtype=inp.dtype) / K
        inp1 = torch.nn.functional.conv2d(inp, w2.to(inp.device), padding="same")
        w2 = torch.ones(1, 1, 1, K, dtype=inp.dtype) / K
        return torch.nn.functional.conv2d(inp1, w2.to(inp.device), padding="same")

def compute_refl(d0, attack, trailing, tof_conv, T):
    # d0 shape: [4 x H x W]
    nscale, H, W = d0.shape
    idx_start = d0 - attack
    idx_start[idx_start<0] = 0
    idx_end = d0 + trailing
    idx_end[idx_end > T-1] = T-1
    r0 = torch.zeros(d0.shape)

    for i in range(nscale):
        for iy, ix in np.ndindex((H,W)):
            r0[i, iy, ix] = tof_conv[i,0,idx_start[i, iy,ix]:idx_end[i, iy,ix]+1, iy, ix].sum()

    return r0

#tof는 histogram, h1은 IRF, nscale_totla = nscale*nirf (여기서는 4 * 2), kk는 2d convolution filter size, 
def gen_initial_multiscale(tof, h1, nscale_total, KK=[1,3,7,13], attack=None, trailing=None, use_cuda=False):
    if len(h1.shape) != 1:
        assert 0, "IRF should have be of 1D"
    if len(tof.shape) != 3:
        assert 0, "ToF shape should be of [height x width x time bin]"

    brefl = True if attack is not None else False

    if use_cuda == None:
        use_cuda = 1 if torch.cuda.is_available() else 0

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("CPU mode is used and it will take several minutes (<= 4 minutes), depending on the data.")

    KK = np.array(KK)
    H, W, T = tof.shape
    
    if type(tof) == np.ndarray:
        tof = torch.tensor(tof)
        h1 = torch.tensor(h1)

    #----------------- 1. cross correlation of histogram with IRF
    # assume that h1 is already shifted
    tof_NxT = tof.reshape(tof.shape[0]*tof.shape[1], tof.shape[2]) #HxW와 T차원으로 만들고 

    #histogram의 FFT와 IRF의 FFT를 곱하고 ifft (주파수 영역에서 곱하고 되돌림으로서 cross correlation 계산)
    tof_irf = torch.real(torch.fft.ifft( torch.fft.fft(tof_NxT.to(device), dim=1) * torch.fft.fft(h1.to(device)), dim=1)) 
    tof_Tx1xHxW = tof_irf.transpose(0,1).reshape(T, 1, H, W) #그 결과를 다시 T x 1 x H x W로 reshape

    #여기까지가 histogram과 IRF의 correlation을 구한 것 그 결과는 tof_Tx1xHxW에 저장

    # tof_Tx1xHxW = torch.real(torch.fft.ifft( torch.fft.fft(tof_NxT.to(device), dim=1) * torch.fft.fft(h1.to(device)), dim=1)).transpose(0,1).reshape(T, 1, H, W)
    # tof_NxT = tof_NxT.cpu()

    del tof_NxT, tof, tof_irf
    if use_cuda:
        torch.backends.cuda.cufft_plan_cache.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    

    # (4, 1, T, H, W)

    ##### new conv
    with torch.no_grad():
        tof_conv = tof_Tx1xHxW.clone().squeeze(1).unsqueeze(0).repeat(4, 1, 1, 1).unsqueeze(1) #4 1 T H W
        del tof_Tx1xHxW
        tof_conv[1] = conv2d(tof_conv[1, 0].unsqueeze(1), KK[1]).transpose(0, 1)
        tof_conv[2] = conv2d(tof_conv[2, 0].unsqueeze(1), KK[2]).transpose(0, 1)
        tof_conv[3] = conv2d(tof_conv[3, 0].unsqueeze(1), KK[3]).transpose(0, 1)
    tof_conv = tof_conv.to(torch.float32).cpu()

    # ##### original conv
    # tof_conv = torch.zeros(4, 1, T, H, W) #original
    # tof_conv[1, :] = conv2d(tof_Tx1xHxW, KK[1]).transpose(0, 1).reshape(1, 1, -1, H, W) #3x3 filter (reshape에 -1은 차원 자동 계산)
    # tof_conv[2, :] = conv2d(tof_Tx1xHxW, KK[2]).transpose(0, 1).reshape(1, 1, -1, H, W) #7x7 filter
    # tof_conv[3, :] = conv2d(tof_Tx1xHxW, KK[3]).transpose(0, 1).reshape(1, 1, -1, H, W) #13x13 filter
    # tof_conv[0] = tof_Tx1xHxW.squeeze(1).unsqueeze(0)
    ###------------------
    #아래 한줄이 원래
    # tof_conv[0, 0, :] = tof_Tx1xHxW.cpu().squeeze(1) #이게 1x1 (squeeze(1)은 차원 크기 1인거 삭제하는 것 -> ,TxHxW로 바뀜)

    # print("8 additional scales will be generated by separable 3d convolution.")
    #----------------- 3. 3D conv to generate two other sets 
    if False: # use_cuda:
        tof_conv = tof_conv.cuda()
        d0 = torch.argmax(tof_conv, dim=2).cpu().squeeze()
        
        if nscale_total >= 8:
            d1 = torch.argmax(conv3d_separate(tof_conv, KK[-2]), dim=2).cpu().squeeze()

        del tof_conv

    else:
        


        print("@, trailing*2=", trailing*2)
        #T축기준 가장 큰 값의 인덱스 (d0는 4xHxW) -> 각 scale에 대해 같은 가장 큰 값의 T index를 추출 
        #(ex: 첫번 째 scale의 H=0, W=0에는 1024개의 T값이 있고, 그중 가장 큰 값을 가지는 T의 index가 담김
        d0 = torch.argmax(tof_conv[:,:,:,:,:], dim=2).cpu().squeeze()
        # d0 = torch.argmax(tof_conv[:,:,:300,:,:], dim=2).cpu().squeeze()
        tof_conv_cpu = tof_conv.clone()
        

        ##### remove highest peak
        # for i in range(4):
        #     width = trailing*2 + KK[i]*2
        #     for j in range(H):
        #         for k in range(W):
        #             #모든 scale, H, W에 대해 d0(가장 큰 값의 인덱스) - width 부터 d0 + width 까지 다 0.0으로 설정 -> 논문의 first peak과 그 around 삭제한다는 부분
        #             tof_conv_cpu[i,0,max(d0[i,j,k]-width,0):min(d0[i,j,k]+width,T-1),j,k] = 0.0 
        #####



        #####Vecotrization (위 6줄 대체)
        # tof_conv_cpu = tof_conv.clone()  # clone 한 번만

        # width 계산 (4,) → (4, 1, 1) → (4, H, W)
        KK = torch.tensor(KK, device=tof_conv.device)
        width = (trailing * 2 + KK * 2).view(-1, 1, 1).expand(-1, H, W)  # (4, H, W)

        # start, end index 계산
        start = (d0 - width).clamp(min=0)
        # end = (d0 + width + 1).clamp(max=T)
        end = (d0 + width).clamp(max=T - 1)  # no +1
        # end   = (d0 + width).clamp(max=T - 1)  # (4, H, W)

        # 시간 인덱스 텐서 (T,)
        time_idx = torch.arange(T, device=tof_conv.device).view(1, T, 1, 1)  # (1, T, 1, 1)

        # (4, H, W) → (4, 1, H, W)
        start = start.unsqueeze(1)
        end   = end.unsqueeze(1)

        # (4, T, H, W) 마스크 생성
        mask = (time_idx >= start) & (time_idx < end)  # < instead of <=

        # mask = (time_idx >= start) & (time_idx <= end)  # (4, T, H, W)

        # tof_conv shape: (4, 1, T, H, W) → squeeze(1) → (4, T, H, W)
        tof_conv_cpu.squeeze_(1)  # (4, T, H, W)

        # 적용
        tof_conv_cpu[mask] = 0.0

        # 다시 차원 맞추려면 (optional)
        tof_conv_cpu = tof_conv_cpu.unsqueeze(1)  # (4,1,T,H,W)
        #####----- Vector end


        #두번째로 큰 값 뽑기 (위에서 0.0으로 설정해서 가장 큰 값은 안담김)
        d0_second = torch.argmax(tof_conv_cpu, dim=2).squeeze()
        
        # d0_second = torch.argmax(tof_conv_cpu[:,:,300:,:,:], dim=2).squeeze() + 300
        # plt.imsave("../result/debug.png", d0[2])
        # plt.imsave("../result/debug1.png", d0_second[2])

        if False: #nscale_total >= 8: #3D cpmv 빼기
            tof_conv2 = conv3d_separate_cpu(tof_conv, KK[-2]) #3차원 convolution with 7x7 filter (tof_conv에 TxHxW)
            d1 = torch.argmax(tof_conv2, dim=2).squeeze() #또 큰값 뽑고 

            for i in range(4):
                width = trailing*2 + KK[i]*2
                for j in range(H):
                    for k in range(W):
                        tof_conv2[i,0,max(d1[i,j,k]-width,0):min(d1[i,j,k]+width,T-1),j,k] = 0.0 #주변 지우고

            d1_second = torch.argmax(tof_conv2, dim=2).squeeze() #second 큰값 또 뽑기

        del tof_conv

    #인덱스별로 (HxW) 2개의 peak의 위치가 엇갈릴 수 있음 그거를 T인덱스 크기에 맞춰 주는 것(뽑을 때 큰 값 기준으로만 뽑았기 때문에 인덱스를 기준으로 앞뒤로 나눠주는 것) 

    #d0가 2d conv, d1이 3dconv(d0에 대해 연산한 것)
    d0_min = torch.min(d0, d0_second) #각 인덱스에서 작은 값만 선택 (예를 들어 [1, 4], [2, 3]이면 결과는 [1, 3])
    d0_max = torch.max(d0, d0_second)
    # d1_min = torch.min(d1, d1_second)
    # d1_max = torch.max(d1, d1_second)
    #4, 4, 4, 4이고 d0_min이 first peak 4개, d0_max가 second peak 4개, d1은 3d conv
    # depths_ = torch.cat([d0_min, d0_max, d1_min, d1_max], dim=0) #depths_는 16xHxW
    depths_ = torch.cat([d0_min, d0_max], dim=0) #d1 (3D conv 뺌)
    del d0
    
    # add 1 to be consistent with Julia and normalize wrt. T as a preprocessing
    depths = (depths_ + 1.0) / T  # 1/1024 ~ 1


    return depths, None #16xHxW

if __name__ == "__main__":
    import time
    tof_cpu = torch.zeros([4, 1, 1024, 500, 500])
    tof = tof_cpu.cuda()
    a = torch.argmax(conv3d_separate(tof, 5), dim=2)
    t0 = time.time()
    a = torch.argmax(conv3d_separate(tof, 5), dim=2)
    print(time.time() - t0)
