import numpy as np
import scipy.special
import scipy.io
import matplotlib.pyplot as plt

#punroll은 meter output 이므로, to_meter = 1.0
def compute_uncertainty_2obj(out, H, W, to_meter=1.0, foutmat=None, nscale = 8):
    """
        - out: the raw output of the network with the debug option
        - to_meter: scaling factor to scale the output of network in the unit of meter
    """
    if out[2] == None:
        assert 0, "The input should be obtained by the inference with the debug option"

    ilast = -1
    # out: x, x_uncertainty, d, w1s, wbars, att_weights

    ########
    x = out[1]
    d = out[2]
    w2 = out[4]


    for npeak in range(len(x)):
        for nblock in range(len(x[0])):
            x[npeak][nblock] = x[npeak][nblock].reshape(1, 1, H, W) #B C H W (이미지처럼)
            d[npeak][nblock] = d[npeak][nblock].transpose(2, 1).reshape(1, -1, H, W) 
            if nblock == len(x[0]) -1:
                continue
            w2[npeak][nblock] = w2[npeak][nblock].transpose(2, 1).reshape(1, -1, H, W) 



    

    ###----------------------------
    # First peak
    ###----------------------------

    ilast = -1
    beta = 0
    alpha = 0

    w2_f = w2[0]
    d_f = d[0]

    d0_f = d_f[0].cpu().numpy() * to_meter
    d1_f = d_f[1].cpu().numpy() * to_meter

    xi_f = x[0][-1].cpu().numpy() * to_meter
    ibatch=0
    
    ww_f = w2_f[0].cpu().numpy()[0,:,:,:]
    ww1_f = w2_f[1].cpu().numpy()[0,:,:,:]

    ###--------------

    ww_norm = scipy.special.softmax(1-ww_f, 0)
    C1 = ww_norm * np.absolute(d0_f[ibatch,:,:,:] - xi_f[ibatch,0:1,:,:]) / (int(nscale / 2) + 2)
    ww_norm = scipy.special.softmax(1-ww1_f, 0)
    C2 = ww_norm * np.absolute(d1_f[ibatch,:,:,:] - xi_f[ibatch,0:1,:,:]) / (int(nscale / 2) + 2)
    epsilon_f = np.sum(C1 + C2, axis=0) / 2 #2로 수정, 우리는 stage = 3


    ###----------------------------
    # Second peak
    ###----------------------------

    ilast = -1
    beta = 0
    alpha = 0

    w2_s = w2[1]
    d_s = d[1]

    d0_s = d_s[0].cpu().numpy() * to_meter
    d1_s = d_s[1].cpu().numpy() * to_meter

    xi_s = x[1][-1].cpu().numpy() * to_meter
    ibatch=0

    ww_s = w2_s[0].cpu().numpy()[0,:,:,:]
    ww1_s = w2_s[1].cpu().numpy()[0,:,:,:]

    ###--------------

    ww_norm = scipy.special.softmax(1-ww_s, 0)
    C1 = ww_norm * np.absolute(d0_s[ibatch,:,:,:] - xi_s[ibatch,0:1,:,:]) / (int(nscale / 2) + 2)
    ww_norm = scipy.special.softmax(1-ww1_s, 0)
    C2 = ww_norm * np.absolute(d1_s[ibatch,:,:,:] - xi_s[ibatch,0:1,:,:]) / (int(nscale / 2) + 2)
    epsilon_s = np.sum(C1 + C2, axis=0) / 2 #2로 수정, 우리는 stage = 3




    # if foutmat is not None:
        # scipy.io.savemat(foutmat, {"depth":x, "refl":rr, "epsilon_f":epsilon_f, "epsilon_s":epsilon_s} )
    
    epsilons = np.concatenate((epsilon_f.reshape(1, H*W, 1), epsilon_s.reshape(1, H*W, 1)), axis=2)
    return epsilons

# def compute_uncertainty_2obj(out):
#     x = out[0] #x: 3 x 1 x HW x 2 (block, 1, HW x 2)
#     x = np.array(out[0].cpu())
#     x_f = x[:, :, :, 0]
#     x_s = x[:, :, :, 1]
#     print(x_f.shape)
#     exit()
#     d = out[1] #d: 3 x 1 x HW x 10
#     w1s = out[2] #w1s: 3 x 1 x HW x 8
#     wbars = out[3] #wbars: 3 x 1 x HW x 8
#     # att_weights = out[4]






## example
# eps = compute_uncertainty(out)
# plt.imshow(eps, vmin=0, vmax=0.01)
# plt.title("uncertainty")