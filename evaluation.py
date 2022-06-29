from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
import torch
import numpy as np
from data import get_loader
from args import get_args
args = get_args()

    


def eval1(model):
    val_acc = 0
    pos_acc = 0
    neg_acc = 0
    true_label = []
    pred_label = []
    train_loaders, val_loaders = get_loader(args) 
    model.eval()
    
    for pos_img in val_loaders['pos']:
        
        ct_b, img_b, c, h, w = pos_img.size()
        pos_img = pos_img.reshape(-1, c, h, w).cuda()
        pos_img = pos_img.cuda().squeeze(0)
        pos_output = model(pos_img)
        pos_numpy = pos_output.cpu().flatten().detach().numpy()
        pos_numpy = np.mean(pos_numpy)
        #print(pos_numpy)
        if pos_numpy > 0:
            pos_acc += 1
            pred_label.append(1)
        else:
            pred_label.append(0)
        true_label.append(1)

    for neg_img in val_loaders['neg']:

        ct_b, img_b, c, h, w = neg_img.size()
        neg_img = neg_img.reshape(-1, c, h, w).cuda()
        neg_img = neg_img.cuda().squeeze(0)
        neg_output = model(neg_img)
        neg_numpy = neg_output.cpu().flatten().detach().numpy()
        neg_numpy = np.mean(neg_numpy)
        #print(neg_numpy)
        if neg_numpy < 0:
            neg_acc += 1
            pred_label.append(0)
        else:
            pred_label.append(1)
        true_label.append(0)

    val_acc = pos_acc + neg_acc
    val_acc /= (len(val_loaders['pos']) + len(val_loaders['neg']))
    pos_acc /= len(val_loaders['pos'])
    neg_acc /= len(val_loaders['neg'])
    val_f1 = f1_score(true_label, pred_label, average='macro')
    print('Val F1: {:.6f} \tVal Acc: {:.6f} \tPos Acc: {:.6f} \tNeg Acc: {:.6f}'.format(
            val_f1, val_acc, pos_acc, neg_acc))
    return val_f1


def eval_final(model):
    pos = []
    neg = []  
    val_acc = 0
    pos_acc = 0
    neg_acc = 0
    true_label = []
    pred_label = []  
    model.eval()
    for _ in range(args.val_epoch):
        sum = 0
        all_pos_output = []
        all_neg_output = []
        train_loaders, val_loaders = get_loader(args)   
        for pos_img in val_loaders['pos']:            
            ct_b, img_b, c, h, w = pos_img.size()           
            pos_img = pos_img.reshape(-1, c, h, w).cuda()           
            pos_output = model(pos_img) 
            pos_numpy = pos_output.cpu().flatten().detach().numpy()
            pos_numpy = np.mean(pos_numpy)
            all_pos_output.append(pos_numpy)

        for neg_img in val_loaders['neg']:        
            ct_b, img_b, c, h, w = neg_img.size()           
            neg_img = neg_img.reshape(-1, c, h, w).cuda()            
            neg_output = model(neg_img)     
            neg_numpy = neg_output.cpu().flatten().detach().numpy()
            neg_numpy = np.mean(neg_numpy)
            all_neg_output.append(neg_numpy)
        pos.append(all_pos_output)
        neg.append(all_neg_output)

    for i,menber in enumerate (pos[0]):
        ind = 0
        for j in pos:
            if j[i]>=0:
                ind +=1
            else:
                ind -=1
        if ind >=0:
            pos_acc += 1
            pred_label.append(1)
        else:
            pred_label.append(0)
        true_label.append(1)

    for i,menber in enumerate (neg[0]):
        ind = 0
        for j in neg:
            if j[i]>=0:
                ind +=1
            else:
                ind -=1
        if ind < 0:
            neg_acc += 1
            pred_label.append(0)
        else:
            pred_label.append(1)
        true_label.append(0)


    val_acc = pos_acc + neg_acc
    val_acc /= (len(val_loaders['pos']) + len(val_loaders['neg']))
    pos_acc /= len(val_loaders['pos'])
    neg_acc /= len(val_loaders['neg'])
    val_f1 = f1_score(true_label, pred_label, average='macro')
    print('Val F1: {:.6f} \tVal Acc: {:.6f} \tPos Acc: {:.6f} \tNeg Acc: {:.6f}'.format(
            val_f1, val_acc, pos_acc, neg_acc))
    return val_f1 


    