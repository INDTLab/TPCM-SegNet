# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from dataset.synapse import get_synapse_val_file_lst_text, SynapseDataset_test_text
from model.__init__ import model_bulider
from utils.metric import calculate_metric_percase
import clip

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="",help="model name")
    parser.add_argument('--load_path', type=str, default="",help="model weight load path")
    parser.add_argument('--num_class', type=int, default=9,help="lIACi num classes")
    parser.add_argument('--in_channels', type=int, default=1,help="input channels")
    parser.add_argument('--resize', type=int, default=224,help="image resize")
    parser.add_argument('--drop_path', type=float, default=0.0,help="drop path rate")
    parser.add_argument('--cuda_id', type=int, default=2,help="id of cuda device,default:0")
    config = parser.parse_args()
    return config

def test(config):
    device = torch.device(f'cuda:{config.cuda_id}')

    test_list = get_synapse_val_file_lst_text("./medical/Synapse/test_vol_h5","./medical/Synapse/text_val")
    test_dataset = SynapseDataset_test_text(test_list,config.resize)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

    net = model_bulider(config.model, config.in_channels,  config.num_class).to(device)
    model_weight = torch.load(config.load_path)
    net.load_state_dict(model_weight['net'])

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

    print(f"dice: {np.mean(test_dice_list,axis=0)[0]}, hd95: {np.mean(test_dice_list,axis=0)[1]}")
    print(f'dice class: {test_dice_list}')
if __name__ == '__main__':

    config = get_parser()
    test(config)

