import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


# class Channel_Attention_Block(nn.Module):
#     def __init__(self):
#         super(Channel_Attention_Block, self).__init__()      
#         self.wk_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)      
#         self.softmax = nn.Softmax(dim = 2)
        
#     def forward(self, x):
#         N, C, H, W = x.size()
#         x = torch.reshape(x, (N, C, H * W))
#         k = x
#         k = torch.reshape(x, (N, 1, C, H * W))
#         k = self.wk_conv(k)
#         k = torch.reshape(x, (N, C, H * W))
#         q = x
#         qk = torch.matmul(q, k.transpose(1, 2))
#         attention_map = self.softmax(qk)
        
#         return attention_map
    
class Channel_Attention_Block(nn.Module):
    def __init__(self):
        super(Channel_Attention_Block, self).__init__()      
        self.multihead_attn = nn.MultiheadAttention(8*8, 1)
        
    def forward(self, x):
        N, C, H, W = x.size()
        x = torch.reshape(x, (N, C, H * W))
        #q,k,v->[C, N, H*W]
        k = x.permute(1,0,2)
        q = k
        v = k
        _, attention_map = self.multihead_attn(q, k, v)   
        #attn_output_rgb->[N, C, C]
        return attention_map
    

class Multihead_Attention_RGB(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8):
        super(Multihead_Attention_RGB, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        N, C, H, W = x.size()
        query = torch.reshape(x,(N, C, H*W))
        #query->[H*W, N, C]
        query = query.permute(2,0,1)
        key = query
        value = query
        attn_output_rgb, weights = self.multihead_attn(query, key, value)   
        #attn_output_rgb->[N, C, H*W]
        attn_output_rgb = attn_output_rgb.permute(1,2,0)
        attn_output_rgb = torch.reshape(attn_output_rgb,(N, C, H, W))
        return attn_output_rgb


class Multihead_Attention_BEV(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8):
        super(Multihead_Attention_BEV, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        N, C, H, W = x.size()
        query = torch.reshape(x,(N, C, H*W))
        #query->[H*W, N, C]
        query = query.permute(2,0,1)
        key = query
        value = query
        attn_output_bev, weights = self.multihead_attn(query, key, value)   
        attn_output_bev = attn_output_bev.permute(1,2,0)
        attn_output_bev = torch.reshape(attn_output_bev,(N, C, H, W))
        return attn_output_bev
    
    
    
    
    
    
    
    
    
    