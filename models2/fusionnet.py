import torch
import torch.nn as nn
from .extractors import *
from .attention_blocks import *
from .generate_des import *



class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.bev_extractor = E2CNN_extractor()
        self.rgb_extractor = CNN_extractor()
        self.concat = Concat_Channel()
        self.adaptation = Adaptation_Layer()
        self.gd_rgb = Generate_Descriptor_RGB()
        self.gd_bev = Generate_Descriptor_BEV()
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.1)
               
    def forward(self, x_rgb, x_bev):
        
        x_merged = self.concat(x_rgb, x_bev)
        wrgb, wbev = self.adaptation(x_merged)
        
        x_rgb = self.rgb_extractor(x_rgb)
        x_bev = self.bev_extractor(x_bev)
        
        #Generate Descriptors x_** -> [N, C*K]
        x_rgb = self.gd_rgb(x_rgb)
        x_bev = self.gd_bev(x_bev)
        x = torch.cat((wbev*x_bev, wrgb*x_rgb), dim = -1)
        x = F.normalize(x, p=2, dim=1)
        return x, x_rgb, x_bev, wrgb, wbev

        
        
    


class Adaptation_Layer(nn.Module):
    def __init__(self):
        super(Adaptation_Layer, self).__init__()
        self.channel_attention = Channel_Attention_Block()
        self.fc_layers1 = nn.Linear(16*16, 2)
        # self.fc_layers2 = nn.Linear(256, 2)
        self.s = 1.0
    
 
    def forward(self, x):
        attention_map = self.channel_attention(x)
        # attention_map -> [N, h, w], x -> [N, h*w]
        x = torch.flatten(attention_map, 1, -1)
        x = self.fc_layers1(x)
        # x = self.fc_layers2(x)
        # weights -> [N, 2]
        weights = 2*torch.sigmoid(self.s*x)
        wrgb, wbev = torch.split(weights, 1, dim = 1)
                
        return wrgb, wbev
    

    

class Concat_Channel(nn.Module):
    def __init__(self):
        super(Concat_Channel, self).__init__()
        self.conv1_rgb = nn.Conv2d(3, 6, kernel_size=3, padding=1, stride=2)
        self.conv1_bev = nn.Conv2d(3, 6, kernel_size=3, padding=1, stride=2)
        self.conv2_rgb = nn.Conv2d(6, 8, kernel_size=3, padding=1, stride=2)
        self.conv2_bev = nn.Conv2d(6, 8, kernel_size=3, padding=1, stride=2)
        self.conv3_rgb = nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1)
        self.conv3_bev = nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1)
        self.bn1_rgb = nn.BatchNorm2d(6)
        self.bn1_bev = nn.BatchNorm2d(6)
        self.bn2_rgb = nn.BatchNorm2d(8)
        self.bn2_bev = nn.BatchNorm2d(8)
        self.bn3_rgb = nn.BatchNorm2d(8)
        self.bn3_bev = nn.BatchNorm2d(8)
        # self.resize_rgb = nn.UpsamplingBilinear2d(size=(64,64))
        # self.resize_bev = nn.UpsamplingBilinear2d(size=(64,64))
       
        
    def forward(self, x_rgb, x_bev):
        
        # x_rgb = self.resize_rgb(x_rgb)
        # x_bev = self.resize_bev(x_bev)
        target_size = (32, 32)
        x_rgb = F.interpolate(x_rgb, size=target_size, mode='bilinear', align_corners=False)
        x_bev = F.interpolate(x_bev, size=target_size, mode='bilinear', align_corners=False)
        x_rgb = self.conv1_rgb(x_rgb)
        x_bev = self.conv1_bev(x_bev)
        x_rgb = self.bn1_rgb(x_rgb)
        x_bev = self.bn1_bev(x_bev)
                    
        x_rgb = self.conv2_rgb(x_rgb)
        x_bev = self.conv2_bev(x_bev)
        x_rgb = self.bn2_rgb(x_rgb)
        x_bev = self.bn2_bev(x_bev) 
        
        x_rgb = self.conv3_rgb(x_rgb)
        x_bev = self.conv3_bev(x_bev)
        x_rgb = self.bn3_rgb(x_rgb)
        x_bev = self.bn3_bev(x_bev) 
               
        x = torch.cat((x_rgb, x_bev), dim = 1)
        return x


