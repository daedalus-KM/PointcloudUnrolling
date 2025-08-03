# https://github.com/luost26/score-denoise/blob/main/models/feature.py

import torch
from torch.nn import Module, Linear, ModuleList, Sequential as Seq
from torch_geometric.nn import knn_graph
#import pytorch3d.ops
from torch_geometric.nn import MessagePassing

class DiagNet(torch.nn.Module):
    def __init__(self, dim_in):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(DiagNet, self).__init__()
        # self.param = torch.nn.Parameter(torch.ones(dim_in)).reshape([1, 1, dim_in]).cuda() #.reshape([1, 1, 2, 1])

        # self.p1 = torch.nn.Parameter(torch.ones(1), requires_grad=True).cuda()
        # self.p2 = torch.nn.Parameter(torch.ones(1), requires_grad=True).cuda()

        self.register_parameter('p1', param=torch.nn.Parameter(10*torch.ones(1).cuda()))
        self.register_parameter('p2', param=torch.nn.Parameter(torch.ones(1).cuda()))
        # self.p1 = self.p1.cuda()
        # self.p2 = self.p2.cuda()

        # if dim_in <= 14:
        #     self.mode = 0

        # else:
        #     self.mode = 1
        #     self.register_parameter(name='p3', param=torch.nn.Parameter(torch.ones(1).cuda()))


    def forward(self, x):
        # x [B x N x C]
        pp = torch.cat([self.p1.expand(2), self.p2.expand(x.shape[2]-2)], 0).reshape([1, 1, x.shape[2]])
            
        # else:
        #     pp = torch.cat([self.p1.expand(2), self.p2.expand(12), self.p3.expand(12)], 0).reshape([1, 1, x.shape[2]])
        return pp * x

def get_knn_idx(x, y, k, offset=0):
    """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, N, k)
    """
    _, knn_idx, _ = pytorch3d.ops.knn_points(x, y, K=k+offset)
    return knn_idx[:, :, offset:]

# # https://github.com/rusty1s/pytorch_cluster/blob/906497e66488901a7b67d24709f0551261c975b5/torch_cluster/knn.py
# @torch.jit.script
def my_knn_graph(pos_, k, b):
    # pos: [B N d]
    pos = pos_.reshape(b, -1, pos_.shape[-1])
    knn_idx = get_knn_idx(pos, pos, k, offset=1)
    row = knn_idx.reshape([knn_idx.shape[0], -1])
    col = torch.arange(pos.shape[1]*pos.shape[0]).repeat_interleave(k, 0).to(pos_.device)
    edge_index = torch.stack([row.reshape(-1), col], dim=0)
    return edge_index

class FCLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, activation=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        if activation is None:
            self.activation = torch.nn.Identity()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'silu':
            self.activation = torch.nn.SiLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.2)
        else:
            raise ValueError()

    def forward(self, x):
        return self.activation(self.linear(x))

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="max"):
        super().__init__(aggr=aggr) #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       torch.nn.ReLU())
                       #,
                       #Linear(out_channels, out_channels))

        self.out_channels = out_channels

    def forward(self, x, edge_index):
        # x: [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
    

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6, aggr="max"):
        super().__init__(in_channels, out_channels, aggr=aggr)
        self.k = k
        # self.out_channels = out_channels

    def forward(self, x, b=None):
        # x : [BN x F]
        # edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        # print("@ knn_graph before", x.shape, self.k, b)
        edge_index = my_knn_graph(x, self.k, b)
        # print("@ knn_graph end")
        return super().forward(x, edge_index)

class FeatureExtraction(Module):

    def __init__(self, 
        in_channels, conv_channels, num_convs=1, k=6, activation='lrelu', aggr="max",
        pretransform=0, conv0_channels=0, bias=True, graphmode=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_convs = num_convs
        self.k = k

        if conv0_channels < 1:
            conv0_channels = conv_channels
        
        self.graphmode = graphmode
        if graphmode >= 2:
            use_dynamic = 1
        else:
            use_dynamic = 0

        # Edge Convolution Units
        self.transforms = ModuleList()
        self.convs = ModuleList()
        self.use_dynamic = use_dynamic

        for i in range(num_convs):
            if i == 0:
                if pretransform:
                    trans = FCLayer(in_channels, in_channels, bias=bias)
                    # trans = torch.nn.Linear(in_channels, in_channels, bias=bias)
                else:
                    trans = torch.nn.Identity()
                    
                if use_dynamic:
                    conv = DynamicEdgeConv(in_channels, conv0_channels, k)
                else:
                    conv = EdgeConv(in_channels, conv0_channels, aggr=aggr)
            else:
                # A -> B -> C
                # A -> B -> C
                # AA -> BB -> CC
                # if use_dynamic or pretransform:
                if pretransform:
                    trans = FCLayer(conv0_channels, conv0_channels, activation=activation, bias=bias)
                else:
                    trans = torch.nn.Identity()

                if use_dynamic:
                    conv = DynamicEdgeConv(conv0_channels, conv_channels, k)
                else:
                    conv = EdgeConv(conv0_channels, conv_channels, aggr=aggr)

            self.transforms.append(trans)
            self.convs.append(conv)

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def forward(self, x_, edge_index = None):
        b, HW, d = x_.shape
        x = x_.reshape(-1, d) # N x d
        
        if edge_index is None:
            edge_index = my_knn_graph(x, self.k, b)

        if self.graphmode >= 1:
            for i in range(self.num_convs):
                x = self.transforms[i](x)
                if self.use_dynamic:
                    x = self.convs[i](x, b)
                else:
                    x = self.convs[i](x, edge_index)

            return x.reshape(b, HW, -1), None

        elif self.graphmode == 0: # reuse the graph from the previous layer
            for i in range(self.num_convs):
                x = self.transforms[i](x)
                x = self.convs[i](x, edge_index)

            return x.reshape(b, HW, -1), edge_index

        else:
            assert 0, "check feature.py"
