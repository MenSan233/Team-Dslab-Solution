import cv2
import numpy as np
import os
import random
import glob
import math
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


m = 0.39221061670618984
s = 0.11469786773730418

class Covid19(Dataset):
    def __init__(self, data_name, path, img_batch, transform = None):
        super().__init__()  
        data_path = os.path.join(path, data_name, 'ct_scan*')
        #weired_path = ['train/pos/ct_scan_31', 'train/pos/ct_scan_47', 'train/pos/ct_scan_609', 'train/neg/ct_scan_853', 'train/neg/ct_scan_781', 'train/neg/ct_scan_354', 'train/neg/ct_scan_537']
        #check_path = [os.path.join(path, i) for i in weired_path]
        
        self.mode = data_name.split('/')[0]
        
        if self.mode == 'train':
            self.data_list = []
            for i in glob.glob(data_path):
                if len(os.listdir(i)) >= 25:
                    self.data_list.append(i)
        else:
            self.data_list = glob.glob(data_path)
            
        self.img_batch = img_batch
        self.transform = transform
        
        if self.mode == 'train':
            self.to_tensor = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((m, m, m), (s, s, s))])
        else:
            self.to_tensor = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize((224,224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((m, m, m), (s, s, s))])
        
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index):
        img_list = []
        img_path_l = os.listdir(self.data_list[index])
        for i in img_path_l:
            i.strip()
            i.strip("\n")
            i.strip("\t")
            i.strip("\"")
            x = i.split('.')[0]
            if str.isdecimal(x):
                img_list.append(float(x))
        # img_list = [int(i.split('.')[0]) for i in img_path_l]
        index_sort = sorted(range(len(img_list)), key=lambda k: img_list[k])
        ct_len = len(img_list)
        start_idx = int(round(ct_len / 10 * 2.5, 0))
        end_idx = int(round(ct_len / 10 * 7.5, 0)) + 1
        len_idx = end_idx - start_idx +1
        mean = (len_idx - 1) / 2
        sqrt_12 = math.sqrt(12)
        stddev = len_idx / sqrt_12
        
        if self.mode == 'train':
            sample_idx = [int(mean)]
            img_sample = torch.zeros((self.img_batch, 3, 224, 224))
            for i in range(self.img_batch-1):  
                img_index = int(random.normalvariate(mean, stddev) + 0.5)
                if img_index < 0 or img_index > len_idx:
                    sample_idx.append(int(mean)+1)
                else:
                    sample_idx.append(img_index+start_idx)
            
        else:
            sample_idx = [int(mean)]
            img_sample = torch.zeros((min(end_idx-start_idx, self.img_batch), 3, 224, 224))
            for i in range(self.img_batch-1):  
                img_index = int(random.normalvariate(mean, stddev) + 0.5)
                if img_index < 0 or img_index > len_idx:
                    sample_idx.append(int(mean)+1)
                else:
                    sample_idx.append(img_index+start_idx)
                    
        for count, idx in enumerate(sample_idx):
            img_path = os.path.join(self.data_list[index], img_path_l[index_sort[idx]])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if (self.transform is not None) & (self.mode == 'train'):
                img = self.transform(image=img)['image']
            
            img = self.to_tensor(img)
            img_sample[count] = img[:]
        return img_sample     
