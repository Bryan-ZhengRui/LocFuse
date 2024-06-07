import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from .attention_blocks import *


class Generate_Descriptor_RGB(nn.Module):
    def __init__(self):
        super(Generate_Descriptor_RGB, self).__init__()
        self.mutihead1 = Multihead_Attention_RGB(num_heads=4, embed_dim=64)
        self.mutihead2 = Multihead_Attention_RGB(num_heads=4, embed_dim=64)
        self.mutihead3 = Multihead_Attention_RGB(num_heads=4, embed_dim=64)
        # 
        self.laynorm11 = nn.LayerNorm([64, 21, 36])
        self.laynorm12 = nn.LayerNorm([64, 21, 36])
        self.laynorm13 = nn.LayerNorm([64, 21, 36])
     

        self.vlad_rgb = NetVLAD(num_clusters=2, dim = 64)
        
         
         
        
    def forward(self, x):
        #x->[N, C, H, W] 
        x_raw = x
        x = self.laynorm11(x_raw + self.mutihead1(x))
        x = self.laynorm12(x_raw + self.mutihead2(x))
        x = self.laynorm13(x_raw + self.mutihead3(x))
        #x->[N, K]
        x = self.vlad_rgb(x)
        # x = F.normalize(x , p=2, dim=1)
        return x
   

class Generate_Descriptor_BEV(nn.Module):
    def __init__(self):
        super(Generate_Descriptor_BEV, self).__init__()
        self.mutihead1 = Multihead_Attention_BEV(num_heads=4, embed_dim=64)
        self.mutihead2 = Multihead_Attention_BEV(num_heads=4, embed_dim=64)
        self.mutihead3 = Multihead_Attention_BEV(num_heads=4, embed_dim=64)

        
        self.laynorm11 = nn.LayerNorm([64,32,32])
        self.laynorm12 = nn.LayerNorm([64,32,32])
        self.laynorm13 = nn.LayerNorm([64,32,32])
        
     
        self.vlad_bev = NetVLAD(num_clusters=2, dim = 64)

    
    def forward(self, x):
        x_raw = x
        x = self.laynorm11(x_raw + self.mutihead1(x))
        x = self.laynorm12(x_raw + self.mutihead2(x))
        x = self.laynorm13(x_raw + self.mutihead3(x))

      
        #x->[N, K]
        x = self.vlad_bev(x) 
        # x = F.normalize(x, p=2, dim=1)
        return x


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, 
                 normalize_input=True, vladv2=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.softmax = nn.Softmax(dim = 1)
        # self.pooling = nn.AvgPool1d(6, stride=6)

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = self.softmax(soft_assign)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        # vlad = self.pooling(vlad)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalized

        return vlad