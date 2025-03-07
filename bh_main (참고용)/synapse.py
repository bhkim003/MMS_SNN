import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import time

from snntorch import spikegen
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

from tqdm import tqdm
import math


from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *
from modules.ae_network import *




############## BPTT Synapse ##################################################
############## BPTT Synapse ##################################################
############## BPTT Synapse ##################################################

class SYNAPSE_CONV_BPTT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_CONV_BPTT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2
        # self.weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, requires_grad=True)
        # self.bias = torch.randn(self.out_channels, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.randn(self.out_channels))


        # # Kaiming 초기화
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

        # nda 초기화
        # n = self.kernel_size * self.kernel_size * self.out_channels
        # self.weight.data.normal_(0, math.sqrt(2. / n))
        # self.bias.data.zero_()

        self.TIME = TIME

        # self.ann_module = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding)
        
    
    def forward(self, spike):
        # spike: [Time, Batch, Channel, Height, Width]   
        # print('spike.shape', spike.shape)
        Time = spike.shape[0]
        assert Time == self.TIME, f'Time {Time} dimension should be same as TIME {self.TIME}'
        Batch = spike.shape[1] 
        Channel = self.out_channels
        Height = (spike.shape[3] + self.padding*2 - self.kernel_size) // self.stride + 1
        Width = (spike.shape[4] + self.padding*2 - self.kernel_size) // self.stride + 1

        # current
        # for t in range(Time):
        #     # print(f'time:{t}', torch.sum(spike_detach[t]/ torch.numel(spike_detach[t])))
        #     output_current.append(F.conv2d(spike[t], self.weight, bias=self.bias, stride=self.stride, padding=self.padding))
        #     # print(f'time:{t}', torch.sum(output_current[t]/ torch.numel(output_current[t])))
        # output_current = torch.stack(output_current, dim=0)
        
        T, B, *spatial_dims = spike.shape
        spike = F.conv2d(spike.reshape(T * B, *spatial_dims), self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        TB, *spatial_dims = spike.shape
        spike = spike.view(T , B, *spatial_dims).contiguous() 
        output_current = spike

        return output_current
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class SYNAPSE_FC_BPTT(nn.Module):
    def __init__(self, in_features, out_features, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_FC_BPTT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.randn(self.out_features))

        # # Kaiming 초기화
        # nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)
        
        # # Xavier 초기화
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.constant_(self.bias, 0)

        # # Kaiming 초기화 (PyTorch의 기본 초기화 방식)
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(self.bias, -bound, bound)

        # Xavier 균등 분포 초기화
        nn.init.xavier_uniform_(self.weight)
        # 바이어스는 0으로 초기화
        nn.init.constant_(self.bias, 0)

        # nn.init.normal_(m.weight, 0, 0.01)
        # nn.init.constant_(m.bias, 0)

        self.TIME = TIME

    def forward(self, spike):
        # # spike: [Time, Batch, Features]   
        # Time = spike.shape[0]
        # assert Time == self.TIME, 'Time dimension should be same as TIME'
        # Batch = spike.shape[1] 

        # output_current = []

        # for t in range(Time):
        #     output_current.append(F.linear(spike[t], weight = self.weight, bias= self.bias))

        # output_current = torch.stack(output_current, dim=0)


        
        T, B, *spatial_dims = spike.shape
        assert T == self.TIME, 'Time dimension should be same as TIME'
        spike = spike.reshape(T * B, *spatial_dims)

        spike = F.linear(spike, weight = self.weight, bias= self.bias)

        TB, *spatial_dims = spike.shape
        spike = spike.view(T , B, *spatial_dims).contiguous() 
        output_current = spike

        return output_current 
############## BPTT Synapse ##################################################
############## BPTT Synapse ##################################################
############## BPTT Synapse ##################################################




############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
    
class SYNAPSE_SEPARABLE_CONV_BPTT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_SEPARABLE_CONV_BPTT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        self.conv_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.conv_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
        # self._initialize_weights()

        # nda 초기화
        m = self.conv_depthwise
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        m = self.conv_pointwise
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()


    def _initialize_weights(self):
        # Xavier initialization for conv_depthwise weights
        nn.init.kaiming_uniform_(self.conv_depthwise.weight)
        if self.conv_depthwise.bias is not None:
            nn.init.constant_(self.conv_depthwise.bias, 0)
        
        # Xavier initialization for conv_pointwise weights
        nn.init.kaiming_uniform_(self.conv_pointwise.weight)
        if self.conv_pointwise.bias is not None:
            nn.init.constant_(self.conv_pointwise.bias, 0)
            

    def forward(self, x):
        T, B, *spatial_dims = x.shape
        x = x.reshape(T * B, *spatial_dims)

        x = self.conv_depthwise(x)
        x = self.conv_pointwise(x)

        TB, *spatial_dims = x.shape
        x = x.view(T , B, *spatial_dims).contiguous() 
        return x
    



class SYNAPSE_DEPTHWISE_CONV_BPTT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_DEPTHWISE_CONV_BPTT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        assert in_channels == out_channels, 'in_channels should be same as out_channels for depthwise conv'
        self.conv_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        
        
        # self._initialize_weights()
        
        # nda 초기화
        m = self.conv_depthwise
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()

    def _initialize_weights(self):
        # Xavier initialization for conv_depthwise weights
        nn.init.kaiming_uniform_(self.conv_depthwise.weight)
        if self.conv_depthwise.bias is not None:
            nn.init.constant_(self.conv_depthwise.bias, 0)

    def forward(self, x):
        T, B, *spatial_dims = x.shape
        x = x.reshape(T * B, *spatial_dims)

        x = self.conv_depthwise(x)

        TB, *spatial_dims = x.shape
        x = x.view(T , B, *spatial_dims).contiguous() 
        return x
############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
############## Separable Conv Synapse #######################################
    





############## OTTT Conv trace #######################################
############## OTTT Conv trace #######################################
############## OTTT Conv trace #######################################
class SYNAPSE_CONV_trace(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8, OTTT_sWS_on = False, first_conv = False):
        super(SYNAPSE_CONV_trace, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2
        # self.weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, requires_grad=True)
        # self.bias = torch.randn(self.out_channels, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.randn(self.out_channels))
        # Kaiming 초기화
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)
        
        self.TIME = TIME

        self.OTTT_sWS_on = OTTT_sWS_on
        # self.first_conv = first_conv # no using in this module

        if (self.OTTT_sWS_on == True):
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))

    def forward(self, spike):
        # spike: [Time, Batch, Channel, Height, Width]   
        # print('spike.shape', spike.shape)
        Time = spike.shape[0]
        assert Time == self.TIME, f'Time dimension {Time} should be same as TIME {self.TIME}'
        Batch = spike.shape[1] 
        Channel = self.out_channels
        Height = (spike.shape[3] + self.padding*2 - self.kernel_size) // self.stride + 1
        Width = (spike.shape[4] + self.padding*2 - self.kernel_size) // self.stride + 1

        if (self.OTTT_sWS_on == True):
            weight = self.get_weight()
        else:
            weight = self.weight

        T, B, *spatial_dims = spike.shape
        spike = F.conv2d(spike.reshape(T * B, *spatial_dims), weight, bias=self.bias, stride=self.stride, padding=self.padding)
        TB, *spatial_dims = spike.shape
        spike = spike.view(T , B, *spatial_dims).contiguous() 
        output_current = spike

        return output_current

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + 1e-4) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        # print(self.gain.size(), self.gain.sum())
        return weight
    

class SYNAPSE_FC_trace(nn.Module):
    def __init__(self, in_features, out_features, trace_const1=1, trace_const2=0.7, TIME=8):
        super(SYNAPSE_FC_trace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        # self.weight = torch.randn(self.out_features, self.in_features, requires_grad=True)
        # self.bias = torch.randn(self.out_features, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.randn(self.out_features))
        # Xavier 초기화
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

        # ottt
        # nn.init.normal_(self.weight, 0, 0.01)
        # nn.init.constant_(self.bias, 0)

        self.TIME = TIME

    def forward(self, spike):
        # spike: [Time, Batch, Features]   
        Time = spike.shape[0]
        assert Time == self.TIME, f'Time({Time}) dimension should be same as TIME({self.TIME})'

        T, B, *spatial_dims = spike.shape
        assert T == self.TIME, 'Time dimension should be same as TIME'

        spike = spike.reshape(T * B, *spatial_dims)
        spike = F.linear(spike, self.weight, self.bias)
        TB, *spatial_dims = spike.shape
        spike = spike.view(T , B, *spatial_dims).contiguous() 
        output_current = spike

        return output_current 
    
############## OTTT Conv trace #######################################
############## OTTT Conv trace #######################################
############## OTTT Conv trace #######################################



############## OTTT Conv trace sstep #######################################
############## OTTT Conv trace sstep #######################################
############## OTTT Conv trace sstep #######################################
class SYNAPSE_CONV_trace_sstep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, trace_const1=1, trace_const2=0.7, TIME=8, OTTT_sWS_on = False, first_conv = False):
        super(SYNAPSE_CONV_trace_sstep, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2
        # self.weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, requires_grad=True)
        # self.bias = torch.randn(self.out_channels, requires_grad=True)
        self.weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.randn(self.out_channels))
        # Kaiming 초기화
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)
        
        self.TIME = TIME

        self.OTTT_sWS_on = OTTT_sWS_on
        # self.first_conv = first_conv # no using in this module

        if (self.OTTT_sWS_on == True):
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))

    def forward(self, spike):
        if (self.OTTT_sWS_on == True):
            weight = self.get_weight()
        else:
            weight = self.weight

        spike = F.conv2d(spike, weight, bias=self.bias, stride=self.stride, padding=self.padding)
        output_current = spike
        return output_current

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + 1e-4) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight
    

    
class SYNAPSE_FC_trace_sstep(nn.Module):
    def __init__(self, in_features, out_features, trace_const1=1, trace_const2=0.7, TIME=8, OTTT_sWS_on = False):
        super(SYNAPSE_FC_trace_sstep, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.randn(self.out_features))
        # Xavier 균등 분포 초기화
        nn.init.xavier_uniform_(self.weight)
        # 바이어스는 0으로 초기화
        nn.init.constant_(self.bias, 0)

        # # Xavier 정규 분포 초기화
        # nn.init.xavier_normal_(self.weight)
        # # 바이어스는 0으로 초기화
        # nn.init.constant_(self.bias, 0)

        # # ottt
        # nn.init.normal_(self.weight, 0, 0.01)
        # nn.init.constant_(self.bias, 0)

        self.TIME = TIME

        self.OTTT_sWS_on = OTTT_sWS_on

        if self.OTTT_sWS_on == True:
            self.gain = nn.Parameter(torch.ones(self.out_features, 1))

    def forward(self, spike):
        # print('spike sparsity', self.in_features, self.out_features, torch.count_nonzero(spike.detach()) / (spike.detach()).numel())

        weight = self.weight if self.OTTT_sWS_on == False else self.get_weight()
        output_current = F.linear(spike, weight, self.bias)
        return output_current 
    

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1], keepdims=True)
        var = torch.var(self.weight, axis=[1], keepdims=True)

        # # 계산된 평균과 분산으로 weight 정규화
        # mean = torch.mean(self.weight, dim=1, keepdim=True)
        # var = torch.var(self.weight, dim=1, keepdim=True)
        # fan_in = self.in_features  # Fully connected layer의 경우 fan_in은 in_features와 동일
        
        # 정규화
        weight = (self.weight - mean) / ((var * fan_in + 1e-4) ** 0.5)
        
        if self.gain is not None:
            weight = weight * self.gain
        
        return weight
        
    
############## OTTT Conv trace sstep #######################################
############## OTTT Conv trace sstep #######################################
############## OTTT Conv trace sstep #######################################

