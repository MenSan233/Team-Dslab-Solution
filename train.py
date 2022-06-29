from data import get_loader
from args import get_args
from tqdm import tqdm
from sklearn.metrics import f1_score
import os
import numpy as np
from timm.models import create_model
from evaluation import eval_final
from evaluation import eval1
import csv
from torch.optim import AdamW
import torch.nn as nn
import torch
from pvt_v2 import PyramidVisionTransformerV2
from functools import partial


args = get_args()
train_loaders, val_loaders = get_loader(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '5'



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


 
              

for _ in range(args.epoch):
    train_loss = 0.0
    train_pos = 0.0
    train_neg = 0.0
    sum = 0

    train_loaders, val_loaders = get_loader(args)   
    model.train()
    for pos_img, neg_img in tqdm(zip(train_loaders['pos'], train_loaders['neg'])):
        sum = sum +1
        
        ct_b, img_b, c, h, w = pos_img.size()
        
        pos_img = pos_img.reshape(-1, c, h, w).cuda()
        neg_img = neg_img.reshape(-1, c, h, w).cuda()
      
        optimizer.zero_grad()
        
        pos_output = model(pos_img)
        neg_output = model(neg_img)

        pos_target = torch.ones_like(pos_output).cuda()
        neg_target = torch.ones_like(neg_output).cuda()
        neg_target = -1*neg_target

        pos_loss = criterion(pos_output, pos_target)
        neg_loss = criterion(neg_output, neg_target)
        
        Loss = pos_loss + neg_loss
        Loss.backward()
        optimizer.step()
        
        train_loss += Loss.item()
        train_pos += pos_loss.item()
        train_neg += neg_loss.item()
        

    
    train_loss /= min(len(train_loaders['pos']), len(train_loaders['neg']))
    print(len(train_loaders['pos']), len(train_loaders['neg']),sum)
    train_pos /= min(len(train_loaders['pos']), len(train_loaders['neg']))
    train_neg /= min(len(train_loaders['pos']), len(train_loaders['neg']))
    print('Epoch: ', _)    
    print('Training Loss: {:.6f} \tPos Loss: {:.6f} \tNeg Loss: {:.6f}'.format(
            train_loss, train_pos, train_neg))
    
    val_f1 = 0
    if _>=0 and _ % 1 == 0:
        val_f1 = eval1(model)  
        if val_f1 > 0.86 and val_f1 <0.87 :
            torch.save(model.module.state_dict(), '0.86.pt')  
        if val_f1 > 0.87 and val_f1 <0.88 :
            torch.save(model.module.state_dict(), '0.87.pt')   
        if val_f1 > 0.88 and val_f1 <0.89:
            torch.save(model.module.state_dict(), '0.88.pt') 
        if val_f1 > 0.89:
            torch.save(model.module.state_dict(), '0.89.pt') 
            break     
print('one time eval')
for _ in range(args.val_epoch):
        val_f1 = eval1(model)            
print('several times eval')
val_f1 = eval_final(model)   

