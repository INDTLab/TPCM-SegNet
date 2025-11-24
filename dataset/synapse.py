# -*- coding: utf-8 -*-
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import os
import numpy as np
import random
import h5py
from PIL import Image


def data_aug(image,label,re_size):
    image = TF.resize(image,(re_size,re_size), antialias=True)
    label = TF.resize(label,(re_size,re_size),interpolation=TF.InterpolationMode.NEAREST, antialias=True)
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image,angle)
        label = TF.rotate(label,angle)
    if random.random() > 0.5:
        image = TF.hflip(image)
        label = TF.hflip(label)
    if random.random() > 0.5:
        image = TF.vflip(image)
        label = TF.vflip(label)
    return image,label


def val_data_aug(image,label,re_size):
    image = TF.resize(image,(re_size,re_size), antialias=True)
    label = TF.resize(label,(re_size,re_size),interpolation=TF.InterpolationMode.NEAREST, antialias=True)
    return image,label



def get_synapse_train_file_lst_text(file_path, text_path):
    
    file_list = os.listdir(file_path)
    num_train = len(file_list)
    path_lst = []
    for f in file_list:
        name = f[:-4]
        f_path = os.path.join(file_path,f)
        text_name = name + '.txt'
        caption_path = os.path.join(text_path,text_name)
        path_lst.append((f_path,caption_path))
    train_lst = path_lst
    return train_lst

def get_synapse_val_file_lst_text(file_path, text_path):
    
    file_list = os.listdir(file_path)
    num_train = len(file_list)
    path_lst = []
    for f in file_list:
        name = f[:-7]
        f_path = os.path.join(file_path,f)
        text_name = name + '.txt'
        caption_path = os.path.join(text_path,text_name)
        path_lst.append((f_path,caption_path))
    val_lst = path_lst
    return val_lst


class SynapseDataset_train_text(Dataset):
    def __init__(self,train_lst,image_size):
        self.path_lst =train_lst
        self.image_size = image_size
        print(f"got {len(self.path_lst)} images,{len(self.path_lst)} masks")
    def __getitem__(self,index):
        data = np.load(self.path_lst[index][0])
        img = data['image']
        mask_ = data['label']
        
        image = TF.to_tensor(img).float().contiguous()
        mask = TF.to_tensor(mask_).long().contiguous()
        
        with open(f'{self.path_lst[index][1]}', 'r', encoding='utf-8') as file:
            text = file.readline().strip()

        image,mask = data_aug(image,mask,self.image_size)
        
        return {
            'image': image,
            'mask': mask,
            'text': text
        }
    def __len__(self):
        return len(self.path_lst)
    
class SynapseDataset_test_text(Dataset):
    def __init__(self,data_list,re_size):
        super(SynapseDataset_test_text,self).__init__()
        self.data_list = data_list
        self.re_size = re_size
        print(f"got {len(self.data_list)} images,{len(self.data_list)} masks")
    def __getitem__(self,index):
        file = self.data_list[index][0]
        file_name = file.split('/')[-1][:-7]
        data = h5py.File(file)
        image, label = data['image'][:], data['label'][:]
        image = torch.tensor(image).contiguous()
        label = torch.tensor(label).contiguous()

        with open(f'{self.data_list[index][1]}', 'r', encoding='utf-8') as file:
            text_list = [line.strip() for line in file]  


        image,label = val_data_aug(image,label,self.re_size)
        sample = {'image': image, 'label': label,'name':file_name,'text':text_list}
        return sample
    def __len__(self):
        return len(self.data_list)

        
        
        
        