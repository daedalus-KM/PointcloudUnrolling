# https://github.com/luost26/score-denoise/blob/main/models/feature.py

import torch
import torch.nn as nn
import pytorch3d.ops
import torch_geometric as pyg



from .GATConv import GATv1Conv


from .feature import FCLayer
import torch.nn.functional as F

def get_knn_idx(x, y, k, offset=0):
    """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, N, k)
    """

    _, knn_idx, _ = pytorch3d.ops.knn_points(x, y, K=k+offset) # offset = 1 
    return knn_idx[:, :, offset:]

# # https://github.com/rusty1s/pytorch_cluster/blob/906497e66488901a7b67d24709f0551261c975b5/torch_cluster/knn.py
# @torch.jit.script

#pos_ = d1_scaled (b x hw x L), k = self.k(8), B = d1.shape[0]

#---------------
#---------------
def knn_graph(pos_, k, b):
    # pos: [B N d]
    pos = pos_.reshape(b, -1, pos_.shape[-1]) 

    knn_idx = get_knn_idx(pos, pos, k, offset=1)  
    row = knn_idx.reshape([knn_idx.shape[0], -1])   
    col = torch.arange(pos.shape[1]*pos.shape[0]).repeat_interleave(k, 0).to(pos_.device) 
    edge_index = torch.stack([row.reshape(-1), col], dim=0) 
    return edge_index 
#---------------

class GATConv(nn.Module):
    """
    Graph Attention Network (GAT) layer
    """
    def __init__(self, att, dim_in, dim_out, bias=True, heads=1):
        super().__init__()
        self.att = att % 10
        
        if self.att == 0: # GATv1
            self.model = GATv1Conv(dim_in, dim_out, heads=heads, concat=False, bias=bias)
        else:
            assert 0, "invalid param: att"

    def forward(self, x, edge_index, return_attention_weights=False):
        if return_attention_weights == False:
            if self.att == 2 or self.att == 0 or self.att == 3:
                x = self.model(x, edge_index) 
            
            else:
                row, col = edge_index[0], edge_index[1]
                if self.att == 4:
                    xixj = x[col,:] * x[row,:]
                elif self.att == 5:
                    xixj = None
                elif self.att == 6:
                    xixj = x[col,:] * x[row,:]
                
                x = self.model(x, edge_index, edge_attr=xixj)
            return x
        else:
            if self.att == 2 or self.att == 0 or self.att == 3:
                x, att_weights = self.model(x, edge_index, return_attention_weights=True)
            
            else:
                row, col = edge_index[0], edge_index[1]
                if self.att == 4:
                    xixj = x[col,:] * x[row,:]
                elif self.att == 5:
                    xixj = None
                elif self.att == 6:
                    xixj = x[col,:] * x[row,:]
                
                x, att_weights = self.model(x, edge_index, edge_attr=xixj, return_attention_weights=True)
                
            return x, att_weights


class FeatureExtraction(nn.Module):
    def __init__(self, att, 
        in_channels, conv_channels, num_convs=1, k=6,
        pretransform=0, bias=True, graphmode=0, apply_act=True, heads=1, feat_interim=32):
        super().__init__()

        if att >= 20:
            print("multihead used")
            heads = 2

        

        self.in_channels = in_channels
        self.num_convs = num_convs

        self.act = torch.nn.LeakyReLU(0.2)
        self.apply_act = apply_act
        self.skip_connection = False
        if att >= 10:
            self.skip_connection = True
        
        self.graphmode = graphmode

        self.convs = nn.ModuleList()
        
        self.k = k


        
        if num_convs == 1:
            self.convs.append(GATConv(att, in_channels, conv_channels, bias, heads=heads))
        else:
            for i in range(num_convs):
                if i == 0:
                    self.convs.append(GATConv(att, in_channels, feat_interim, bias, heads=heads))
                elif i == num_convs-1:
                    self.convs.append(GATConv(att, feat_interim, conv_channels, bias, heads=heads))
                else:
                    self.convs.append(GATConv(att, feat_interim, feat_interim, bias, heads=heads))

        

    # @property
    # def out_channels(self):
    #     return self.convs[-1].out_channels

    def forward(self, x_, edge_index, featmode=0, debug=0):
        assert edge_index is not None
        b, HW, d = x_.shape
        x = x_.reshape(-1, d) # N x d
        # if self.trans is not None:
        #     x = self.trans(x)


        if debug:
            att_weights = []
        for i in range(self.num_convs):
            x_conv = self.convs[i](x, edge_index)

            # if debug == False:
            #     x_conv = self.convs[i](x, edge_index)
            # else:
            #     x_conv, att = self.convs[i](x, edge_index, return_attention_weights=True)
            #     att_weights.append(att)
            
            if self.skip_connection == True and x.shape[1] == x_conv.shape[1]:
                x = x_conv + x
            else:
                x = x_conv

            if i < self.num_convs-1:
                if self.apply_act:
                    x = self.act(x)
        #edge_index_out = edge_index  # reuse the graph from the previous layer
        if self.graphmode >= 1:
            edge_index_out = None
        
        return x.reshape(b, HW, -1)#, edge_index_out 함 꺼보겠음 184번 줄도 같이
        # if debug == False:
            
        # else:
        #     return x.reshape(b, HW, -1), edge_index_out, att_weights
