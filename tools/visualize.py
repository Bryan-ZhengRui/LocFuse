import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.datasets_oxford import *
from models2 import * 
import numpy as np
import cv2



def hook_vis(rgb_pathname, bev_pathname, net = FusionNet(), Dir=None, weights=None):
    net.eval()
    load_path = os.path.join(Dir,weights)
    data_rgb = load_png(rgb_pathname)         
    data_rgb = torch.tensor(data_rgb, dtype=torch.float32) 
    data_rgb = data_rgb.unsqueeze(0)
    data_bev = load_png(bev_pathname)         
    data_bev = torch.tensor(data_bev, dtype=torch.float32) 
    data_bev = data_bev.unsqueeze(0)
    net.load_state_dict(torch.load(load_path), False)
    net.eval()
    features = []
    def hook(module, input, output):
        features.append(output[1].clone().detach())
    rgb, bev = data_rgb, data_bev 
    handle = net.gd_rgb.mutihead2.multihead_attn.register_forward_hook(hook)
    embd, _, _, wrgb, wbev = net(rgb, bev)
    print(features[0].size())
    print(features[0])
    handle.remove()
    y1 = features[0].detach().numpy()
    print(y1.shape)
    A = y1[0]
    # print(A.shape)
    plt.imsave('vis/atmap.png',A)
    mask = np.zeros((32,32))
    for patch_id in range(1024):
        heat_map = np.reshape(A[patch_id],(32,32))
        mask += heat_map
    plt.imsave('vis/mask.png',mask, cmap ='jet')
    mask = cv2.imread('vis/mask.png')
    pic = cv2.imread(rgb_pathname)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    mask = cv2.resize(mask,(320,240))
    img_mix = cv2.addWeighted(pic, 0.3, mask, 0.7, 0) 
    plt.imsave('vis/heat_map.png', img_mix)

    return img_mix


if __name__ == "__main__":
    hook_vis('data2exam/1447410692204567.png', 'data2exam/1447410692204567_bev.png', net = FusionNet())