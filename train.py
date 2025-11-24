# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import numpy as np
import random
import time
import argparse
import os.path as osp
from torch.utils.data import DataLoader
from dataset.synapse import get_synapse_train_file_lst_text,get_synapse_val_file_lst_text, SynapseDataset_train_text, SynapseDataset_test_text
from model.__init__ import model_bulider
from utils.metric import calculate_metric_percase
import clip

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="TPCMSegNet",help="model name")
    parser.add_argument('--num_class', type=int, default=9,help="Synapse num classes")
    parser.add_argument('--in_channels', type=int, default=1,help="input channels")
    parser.add_argument('--num_epochs', type=int, default=450,help="training epochs")
    parser.add_argument('--resize', type=int, default=224,help="image resize")
    parser.add_argument('--batch_size', type=int, default=4,help="train batch_size, val batch_size default=0")
    parser.add_argument('--drop_path', type=float, default=0.0,help="drop path rate")
    parser.add_argument('--optim', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'],help="select optimizer")
    parser.add_argument('--weight_decay', type=float, default=0.05,help="weight_decay in optim")
    parser.add_argument('--lr', type=float, default=5e-5,help="learning rate")
    parser.add_argument('--save_dir', type=str, default="./checkpoints/synapse",help="log and weight save_dir")
    parser.add_argument('--cuda_id', type=int, default=4,help="id of cuda device,default:0")

    config = parser.parse_args()
    config.model = config.model.replace('\r', '')
    return config

def train(config):

    device = torch.device(f'cuda:{config.cuda_id}')

    log_path = f'./{config.save_dir}/{config.model}/{config.model}_batch{config.batch_size}_{config.num_epochs}epoch_lr{config.lr}_oncuda{config.cuda_id}_log/'
    if not osp.exists(log_path):
        os.makedirs(log_path)

    train_lst = get_synapse_train_file_lst_text("./medical/Synapse/train_npz","./medical/Synapse/text_tr")
    train_set = SynapseDataset_train_text(train_lst,image_size=config.resize)
    train_loader = DataLoader(train_set,batch_size = config.batch_size,shuffle=True,num_workers=8,pin_memory=True)


    test_list = get_synapse_val_file_lst_text("./medical/Synapse/test_vol_h5","./medical/Synapse/text_val")
    test_dataset = SynapseDataset_test_text(test_list,config.resize)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

    net = model_bulider(config.model, config.in_channels, config.num_class).to(device)

    if config.optim == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(),lr = config.lr,weight_decay=config.weight_decay)
    elif config.optim == 'adam':
        optimizer = torch.optim.Adam(net.parameters(),lr = config.lr,weight_decay=config.weight_decay)
    elif config.optim == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(),lr = config.lr,weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {config.optim}")

    ce_criterion = nn.CrossEntropyLoss()

    max_iters = config.num_epochs * len(train_loader)

    num_iter = 0
    best_dice = 0
    for epoch in range(config.num_epochs):
        net.train()
        tic = time.time()
        total_ce_loss = 0

        for data in train_loader:
            optimizer.zero_grad()
            image = data['image'].to(device)
            label = data['mask'].squeeze(1).to(device)
            text = clip.tokenize(data['text']).to(device)
            pred = net(image,text)
            loss_ce = ce_criterion(pred, label)
            
            total_ce_loss = total_ce_loss + loss_ce.item() 
            loss = loss_ce      
            loss.backward()
            optimizer.step()
            lr_ = config.lr * (1-num_iter/max_iters) ** 0.9
            num_iter += 1
            for param in optimizer.param_groups:
                param['lr'] = lr_      

        toc = time.time()
        print(f"epoch: {epoch+1}/{config.num_epochs}, ce_loss: {total_ce_loss/len(train_loader):.5f}, train time: {toc-tic}")
        net.eval()
        test_dice_list = 0.0

        with torch.no_grad():
            for data in test_loader:
                metric_lst = []
                image,label = data['image'][0],data['label'][0]
                textlist = data['text']

                label = label.detach().numpy()
                prediction = np.zeros_like(label)

                for index in range(image.shape[0]):
                    slice_ = image[index,:,:]

                    text = clip.tokenize(textlist[index]).to(device)

                    input_tensor = slice_.unsqueeze(0).unsqueeze(0).float().to(device)
                    pred = net(input_tensor,text)
                    out = torch.argmax(torch.softmax(pred, dim=1), dim=1).squeeze(0)

                    out = out.cpu().detach().numpy()

                    prediction[index] = out

                for i in range(1,config.num_class):
                    metric_lst.append(calculate_metric_percase(prediction==i,label==i))

                test_dice_list += np.array(metric_lst)
        test_dice_list /= len(test_dataset)
        if(np.mean(test_dice_list,axis=0)[0]>best_dice):
            best_dice = np.mean(test_dice_list,axis=0)[0]
            state_dict = {'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
            torch.save(state_dict, log_path + f'{config.model.lower()}_best_dice_val.pth')
            print(f'save {config.model} best dice val model on epoch: {epoch}')
                         
        print(f'mean dice on test set: {np.mean(test_dice_list,axis=0)[0]} mean hd95: {np.mean(test_dice_list,axis=0)[1]}')
        
                
if __name__ == '__main__':

    config = get_parser()
    start_time = time.time()
    train(config)
    end_time = time.time()
    hours = 1.0*(end_time-start_time) / 3600
    minutes = (hours - int(hours)) * 60
    print(f'total time: {int(hours)} hours {int(minutes)} minutes')

