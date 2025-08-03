# This code is inspired by the original method: https://github.com/luost26/score-denoise/blob/main/train.py
import time
#base model
import matplotlib.pyplot as plt #for visualize features

import sys #for check memory size

def total_size(o, handlers={}):
    if id(o) in handlers:
        return 0
    handlers[id(o)] = True
    size = sys.getsizeof(o)
    if isinstance(o, (str, bytes, bytearray)):
        return size
    if isinstance(o, dict):
        size += sum([total_size(v, handlers) for v in o.values()])
        size += sum([total_size(k, handlers) for k in o.keys()])
    elif hasattr(o, '__dict__'):
        size += total_size(o.__dict__, handlers)
    elif hasattr(o, '__iter__') and not isinstance(o, (str, bytes, bytearray)):
        size += sum([total_size(i, handlers) for i in o])
    return size

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from . import feature
from . import GAT as gat # 위에서 수정
from torch_geometric.utils import add_self_loops, degree, grid
import gc
class Model(nn.Module):
    
    def __init__(self, nscale=12, att=2, k=8, nconv=2, nblock=3, pret=0, feat=0, nheight=128):
        super().__init__()
        self.att = att
        self.featmode = feat #feat은 현재 0
        self.act = torch.nn.LeakyReLU(0.2)
        
        self.graphmode = 0
        self.nblock = nblock

        # geometry
        self.k = k
        self.pretransform = pret

        self.nscale = nscale
        self.nfeat0 = nscale+2
        self.nfeat = nscale
        feat_interim = 32

        self.nheight = nheight
        
        self.ngraph = int(self.nscale / 2)

        # networks
        self.attention_blocks = nn.ModuleList()
        self.expands = nn.ModuleList()
        self.transforms = nn.ModuleList()

        pret2 = 0
        self.use_xy_as_featuers = False #only use z1, z2

        if self.use_xy_as_featuers == False:
            nf0 = self.nscale
        else:
            nf0 = self.nscale + 2


        # feature.FeatureExtraction(nscale+2, nscale)
        for i in range(nblock):
            
            pret_inside = 0
            if (i == 0 and self.featmode < 100) or (i==1 and self.featmode % 100 < 20) or (i==2 and self.featmode % 100 < 10) or (self.multigraph == 1) or (self.multigraph == 2): 
                #att, in channel, conv channel
                self.attention_blocks.append( gat.FeatureExtraction(self.att, nf0, self.nfeat, k=k, num_convs=nconv, pretransform=pret_inside, feat_interim=feat_interim) ) # feature extraction
                self.attention_blocks.append( gat.FeatureExtraction(self.att, self.nfeat, self.nfeat, k=k, num_convs=nconv, pretransform=pret2, feat_interim=feat_interim) ) # attention

                if i < nblock-1: #expands아직 안함
                    self.expands.append(gat.FeatureExtraction(self.att, nf0, self.nfeat, k=k, num_convs=nconv, pretransform=pret) ) # F.E. of d-x
                    self.expands.append( gat.FeatureExtraction(self.att, self.nfeat, self.nfeat, k=k, num_convs=nconv, pretransform=pret2) ) # alternative PAConv  
                    self.expands.append( gat.FeatureExtraction(self.att, self.nfeat, self.nfeat, k=k, num_convs=nconv, pretransform=pret2) ) 
#-----
            else:
                exit("not supported")
                

        self.sm = nn.Softmax(3)
        self.sm_for_uncertainty = nn.Softmax(2)

        self.tau = 10.0

        self.use_linear_combination = True

        self.edge_index_grid = None
        self.use_reduced_x = True


    def forward(self, d1, r1=None, debug=0):
        """
        Args:
            d1 [B x N x (2+L)]

        Returns:
            x: list. x[0]: [B x N x 3]
        """
        x = [None] * self.nblock
        # r = [None] * self.nblock; m = [None] * self.nblock

        B, N = d1.shape[0], d1.shape[1]
        d1_xy = d1[:,:,0:2] #B x n x 2 (x, y좌표만 추출)

        #For Uncertainty
        if debug:
            x_uncertainty = [[None] * self.nblock for _ in range(2)]    
            d = [[None] * self.nblock for _ in range(2)]
            w1s = [[None] * self.nblock for _ in range(2)]
            wbars = [[None] * self.nblock for _ in range(2)]
            att_weights = [[None] * self.nblock for _ in range(2)]

            d[0][0] = d1[:, :, 2::2].cpu()
            d[1][0] = d1[:, :, 3::2].cpu()



        for i in range(self.nblock):
            #---------------------------------------
            # squeeze
            #---------------------------------------
            edge_index = gat.knn_graph(d1, self.k, B)

            if self.use_xy_as_featuers == True:
                d1_inp = d1
            else:
                d1_inp = d1[:,:,2:]

            d1_features = self.attention_blocks[i*2](d1_inp, edge_index, featmode=self.featmode) # out: [B x N x (nfeat)]
            d1_features = self.act(d1_features)


            w1 = self.attention_blocks[i*2+1](d1_features, edge_index)


            onehot = F.gumbel_softmax(w1[:, :, ::2], tau=self.tau, hard=True, dim=2) #first peak
            x1 = torch.sum(d1[:,:, 2::2] * onehot, dim=2).reshape(B, N, 1)


            onehot = F.gumbel_softmax(w1[:, :, 1::2], tau=self.tau, hard=True, dim=2)
            x2 = torch.sum(d1[:,:, 3::2] * onehot, dim=2).reshape(B, N, 1)

            #For uncertainty
            if debug:
                w1s[0][i] = self.sm_for_uncertainty(w1[:, :, 0::2])
                w1s[1][i] = self.sm_for_uncertainty(w1[:, :, 1::2])

            x[i] = torch.cat([x1, x2], dim=2)
            if debug:
                x_uncertainty[0][i] = x1
                x_uncertainty[1][i] = x2
            if i == self.nblock-1: 
                x[i] = torch.cat([x1, x2], dim=2)
            
                break


                #---------------------------------------
                #--------------------- expansion
                #---------------------------------------
                # assert(self.nscale == 8, "only support when nscale = 8")


            x1_expand = torch.zeros((B, N, self.nscale, 2)).cuda() 
            for ii in range(self.ngraph):
                x1_expand[:, :, ii*2, 0] = x1.squeeze() #x1: B x HW x 1

            for ii in range(self.ngraph):
                x1_expand[:, :, ii*2+1, 0] = x2.squeeze() #x2: B x HW x 1

            x1_expand[:, :, :, 1] = d1[:,:, 2:]



            d_x = torch.cat((torch.absolute(d1[:,:,2::2] - x1[:,:,0:1]), torch.absolute(d1[:,:,3::2] - x2[:,:,0:1])), dim = 2)

            x1_features = d_x.clone()
            x1_features[:, :, ::2] = d_x[:, :, :self.ngraph] #first peak 
            x1_features[:, :, 1::2] = d_x[:, :, self.ngraph:] #second peak 



            x1_features = self.expands[3*i](x1_features, edge_index)
            x1_features = self.act(x1_features)



            x1_features_ws = self.expands[3*i+1](x1_features, edge_index)
            del x1_features
            d1_features_ws = self.expands[3*i+2](d1_features, edge_index)
            del d1_features

            
            #wbar: B x N x L x 2
            wbar = self.sm(torch.cat([x1_features_ws.unsqueeze(3), d1_features_ws.unsqueeze(3)], dim = 3))
            del x1_features_ws, d1_features_ws

            if self.use_linear_combination: 
                wx = wbar * x1_expand
                d12 = torch.sum(wx, dim=-1).reshape(B, -1, self.nscale)

                
            
            if debug:
                wbars[0][i] = wbar[:,:,0::2, 0].reshape(wbar.shape[0], wbar.shape[1], int(self.nscale / 2)).cpu()
                wbars[1][i] = wbar[:,:,1::2, 0].reshape(wbar.shape[0], wbar.shape[1], int(self.nscale / 2)).cpu()

            if debug:
                d[0][i+1] = d12[:, :, 0::2].cpu()
                d[1][i+1] = d12[:, :, 1::2].cpu()
            
            del wbar, x1_expand
            d1 = torch.cat([d1_xy, d12], dim=2)
                
            #-------------------------------------------
            # change x,y
            if self.featmode >= 1000:
                pass

        if debug==False:
            return x, None
        else:
            return x, x_uncertainty, d, w1s, wbars, att_weights

    
#######--------------- test grid
def main():
    import torch
    from torch_geometric.utils import add_self_loops, degree, grid
    import torch.nn.functional as F
    K = 2
    unfold = torch.nn.Unfold(kernel_size=(K, K),padding=0)
    nheight=128
    # edge_index, grid_pos = grid(nheight, nheight)
    b = 2
    d1 = torch.zeros(b, 128*128, 14)
    d1_img = d1[:,:,2:].reshape(d1.shape[0], nheight, -1, d1.shape[-1]-2) # [B x H x W x 1]
    d1_bchw = d1_img.permute([0, 3, 1, 2])

    d1_bchw_pad = F.pad(d1_bchw, (0,1,0,1), mode='replicate')

    d1_BxCCxHW = unfold(d1_bchw_pad)
    d1_fold = d1_BxCCxHW.permute([0, 2, 1]).reshape(d1.shape[0], d1.shape[1], -1)
    d1_augmented = torch.cat([d1[:,:,:2], d1_fold], dim=2)
