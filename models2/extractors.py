# build a simple equivariant model using a SequentialModule
import escnn
import torch
from escnn import gspaces
from escnn.nn import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn

        
class E2CNN_extractor (nn.Module):
    def __init__(self):
        super(E2CNN_extractor, self).__init__()
        self.s = gspaces.rot2dOnR2(4)
        self.c_in = escnn.nn.FieldType(self.s, [self.s.trivial_repr]*3)
        self.c_hid1 = escnn.nn.FieldType(self.s, [self.s.regular_repr]*4)
        self.c_hid2 = escnn.nn.FieldType(self.s, [self.s.regular_repr]*8)
        self.c_hid3 = escnn.nn.FieldType(self.s, [self.s.regular_repr]*16)
        self.c_hid4 = escnn.nn.FieldType(self.s, [self.s.regular_repr]*24)
        self.c_hid5 = escnn.nn.FieldType(self.s, [self.s.regular_repr]*36)
        self.c_hid6 = escnn.nn.FieldType(self.s, [self.s.regular_repr]*48)
        self.c_hid7 = escnn.nn.FieldType(self.s, [self.s.regular_repr]*48)
        self.c_out = escnn.nn.FieldType(self.s, [self.s.regular_repr]*64)
        self.extractor_bev = SequentialModule(
            R2Conv(self.c_in, self.c_hid1, 7, stride=1, bias=False),            
            ELU(self.c_hid1, inplace=True),
            InnerBatchNorm(self.c_hid1),
            PointwiseMaxPool(self.c_hid1, kernel_size=2, stride=2),
            R2Conv(self.c_hid1, self.c_hid2, 3, stride=1, padding=1, bias=False), 
            ELU(self.c_hid2, inplace=True),
            InnerBatchNorm(self.c_hid2),
            PointwiseMaxPool(self.c_hid2, kernel_size=2, stride=2),
            R2Conv(self.c_hid2, self.c_hid3, 3, stride=1, bias=False), 
            ELU(self.c_hid3, inplace=True),
            InnerBatchNorm(self.c_hid3),
            PointwiseMaxPool(self.c_hid3, kernel_size=2, stride=2),
            R2Conv(self.c_hid3, self.c_hid4, 3, stride=1 , bias=False), 
            ELU(self.c_hid4, inplace=True),
            InnerBatchNorm(self.c_hid4),
            R2Conv(self.c_hid4, self.c_hid5, 3, stride=1, bias=False),        
            ELU(self.c_hid5, inplace=True),
            InnerBatchNorm(self.c_hid5),
            R2Conv(self.c_hid5, self.c_hid6, 3, stride=1, bias=False),           
            ELU(self.c_hid6, inplace=True),
            InnerBatchNorm(self.c_hid6),
            R2Conv(self.c_hid6, self.c_hid7, 3, stride=1, bias=False), 
            ELU(self.c_hid7, inplace=True),
            InnerBatchNorm(self.c_hid7),
            R2Conv(self.c_hid7, self.c_out, 3, stride=1, bias=False), 
            ELU(self.c_out, inplace=True),
            InnerBatchNorm(self.c_out),
            GroupPooling(self.c_out)
        )
        self.extractor_bev_exported = self.extractor_bev.export()
        
    def forward(self, x):
        x = self.extractor_bev_exported(x)
        return x     


class CNN_extractor(nn.Module):
    def __init__(self):
        super(CNN_extractor, self).__init__()      
        self.extractor_rgb = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1, bias=False),       
            nn.ELU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, stride=2),  
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),                      
            nn.ELU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2,2), stride=(2,2)),
            nn.Conv2d(16, 24, kernel_size=3, stride=1,padding=1, bias=False),           
            nn.ELU(inplace=True),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1, bias=False),           
            nn.ELU(inplace=True),   
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, bias=False),          
            nn.ELU(inplace=True),  
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, bias=False),   
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64)
        )
        
    def forward(self, x):
        # [N, 64, 21, 36]
        x = self.extractor_rgb(x)
        return x
        




if __name__ == "__main__":
# export the model
    net = E2CNN_extractor()
    # net = CNN_extractor()
    print(net)
    # check that the two models are equivalent
    img_cv = cv2.imread('data2exam/1418132627623605_bev.png')
    img_cv = np.transpose(img_cv, [2, 0, 1])
    # img_cv = cv2.imread('1418721848251526.png')
    # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # 转torch.Tensor
    x1 = torch.tensor(img_cv)
    print(x1.shape)
    x1 = x1.unsqueeze(0)
    # x1 = x1.permute(0,3,1,2)
    print(x1.shape)
    # x1 = torch.randn(1, c_in.size, 350, 350)
    x2 = torch.rot90(x1, k=1, dims =[2, 3])
    x3 = torch.rot90(x1, k=2, dims =[2, 3])
    x4 = torch.rot90(x1, k=3, dims =[2, 3])
    y1 = net(x1.to(torch.float32))
    y3 = net(x3.to(torch.float32))
    y4 = net(x4.to(torch.float32))
    y2 = net(x2.to(torch.float32))
    y1 = y1.detach().numpy()
    y2 = y2.detach().numpy()
    y3 = y3.detach().numpy()
    y4 = y4.detach().numpy()
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    #         print('weights:',param)
    print(y1.shape)
    A = y1[0,-1,:]
    B = y2[0,-1,:]
    C = y3[0,-1,:]
    D = y4[0,-1,:]
    print(A.shape)
    plt.imsave('./01.png',A, cmap='jet')
    plt.imsave('./02.png',B, cmap='jet')
    plt.imsave('./03.png',C, cmap='jet')
    plt.imsave('./04.png',D, cmap='jet')