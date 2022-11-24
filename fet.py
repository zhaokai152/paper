# =============================================================================
import re 
import os, glob, datetime, time   
import numpy as np 
import torch  
 
from torch.nn.modules.loss import _Loss  
import torch.nn.init as init  
from torch.utils.data import DataLoader  
import torch.optim as optim    
from torch.optim.lr_scheduler import MultiStepLR  
import data_generator as dg 
from data_generator import DenoisingDataset  

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__() 
        kernel_size = 3  #卷积核的大小  3*3
        padding = 1  
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()  
 
 