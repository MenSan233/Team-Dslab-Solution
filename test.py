from data import get_loader
from args import get_args
from tqdm import tqdm
from sklearn.metrics import f1_score
import os
import numpy as np
from timm.models import create_model
from evaluation import eval_final
from evaluation import eval1
from evaluation import test
import csv
from torch.optim import AdamW
import torch.nn as nn
import torch
from pvt_v2 import PyramidVisionTransformerV2
from functools import partial


args = get_args()
train_loaders, val_loaders = get_loader(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'



model = PyramidVisionTransformerV2(num_classes=1,embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,drop_rate=0.0,
        drop_path_rate=0.1,norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1])


model.load_state_dict(torch.load('0.86.pt'), strict=False)
#model.load_state_dict(torch.load("latenico_mse.pt"), strict=False)
#model.eval()
model = nn.DataParallel(model,device_ids=[0])
device=torch.device('cuda',0)
model = model.to(device)

#val_f1 = evaluation(model,val_loaders)
#val_f1 = eval_final(model)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
test(model)

