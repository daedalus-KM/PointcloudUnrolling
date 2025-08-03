import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision

import pyrallis
import mat73

from tqdm import tqdm


os.sys.path.append("../")
os.sys.path.append("./") # necessary when debugging in vs code

from parse_args import TrainConfig, ROOT_DIR

args = pyrallis.parse(config_class=TrainConfig)

ddata, dresult = args.get_ddata(), args.get_dresult()
print(args)


CHECKPOINT_PATH = dresult


from model.vanilla import Model

# some settings
torch.backends.cudnn.benchmark=True
torch.manual_seed(1)
np.random.seed(1)
# #------------------------------------------------
# # load data and the model
# #------------------------------------------------
datasets = []

pppsbr = ((4.0, 64.0),)

print(pppsbr)


import train_tof_dataset as dataset


for ps in pppsbr:
    ff = f"{ddata}/train_T=1024_ppp={ps[0]}_sbr={ps[1]}_2obj.h5"
    datasets.append(dataset.TofDataset(ff, nscale=args.nscale))
train_dataset = torch.utils.data.ConcatDataset(datasets)
del datasets

train_loader = torch.utils.data.DataLoader(train_dataset, args.b, shuffle=True, drop_last=True)

#------------------------------------------------
# model
#------------------------------------------------
#nsclae = 8, att=10, k = 6, feat = 0, nblock = 3, nconv = 3, pret = 0, run2 = 0

model = Model(nscale=args.nscale, att=args.att, k=args.k, feat=args.feat, nblock=args.nblock, nconv=args.nconv, pret=args.pret).cuda()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
nparams = sum([np.prod(p.size()) for p in model_parameters])
print(f"nparams: {nparams}")

opt = torch.optim.Adam(model.parameters(), args.lr) 

# #------------------------------------------------
# # training
# #------------------------------------------------

#training mode
model.train()
nepoch = args.nepoch

for epoch in range(nepoch):
    # #------------------
    # # Train
    # #------------------
    model.train()
    loss_sum = 0.0
    for i, ds_batch in enumerate(tqdm(iter(train_loader))):

        depth, d_gt = ds_batch[0], ds_batch[2]
        opt.zero_grad()
        x, d = model(depth.cuda()) 
            
        if d_gt.shape[2] == 4:
            xgt = d_gt[:,:,2:].cuda()
        elif d_gt.shape[2] == 2:
            xgt = d_gt.cuda()
        else:
            assert 0, "error on d_gt.shape"

        loss = F.l1_loss(xgt[..., 0], x[-1][..., 0], reduction='mean') \
                + F.l1_loss(xgt[..., 1], x[-1][..., 1], reduction='mean')

        loss.backward()
        opt.step()

        loss_sum += loss
 
    loss_sum = loss_sum / len(train_loader)
    print(f"epoch:{epoch}, Train Loss: {loss_sum} ")

    torch.save({ # Save our checkpoint loc
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'loss': loss,
    }, CHECKPOINT_PATH + f"ckpt_{epoch}.pth")