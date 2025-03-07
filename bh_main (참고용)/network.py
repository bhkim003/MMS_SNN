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

from torch import Tensor
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t, _ratio_any_t
from typing import Optional, List, Tuple, Union
from typing import Callable

from apex.parallel import DistributedDataParallel as DDP

from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *
from modules.old_fashioned import *
from modules.ae_network import *




######## make_layers for Conv ############################################
######## make_layers for Conv ############################################
######## make_layers for Conv ############################################
class MY_SNN_CONV(nn.Module):
    def __init__(self, cfg, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_conv_trace_const1, 
                    synapse_conv_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    synapse_fc_out_features, synapse_fc_trace_const1, synapse_fc_trace_const2,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    OTTT_sWS_on,
                    DFA_on,
                    drop_rate,
                    UDA_on,
                    alpha_uda):
        super(MY_SNN_CONV, self).__init__()
        self.params = {
            'cfg': cfg,
            'in_c': in_c,
            'IMAGE_SIZE': IMAGE_SIZE,
            'synapse_conv_kernel_size': synapse_conv_kernel_size,
            'synapse_conv_stride': synapse_conv_stride,
            'synapse_conv_padding': synapse_conv_padding,
            'synapse_conv_trace_const1': synapse_conv_trace_const1,
            'synapse_conv_trace_const2': synapse_conv_trace_const2,
            'lif_layer_v_init': lif_layer_v_init,
            'lif_layer_v_decay': lif_layer_v_decay,
            'lif_layer_v_threshold': lif_layer_v_threshold,
            'lif_layer_v_reset': lif_layer_v_reset,
            'lif_layer_sg_width': lif_layer_sg_width,
            'synapse_fc_out_features': synapse_fc_out_features,
            'synapse_fc_trace_const1': synapse_fc_trace_const1,
            'synapse_fc_trace_const2': synapse_fc_trace_const2,
            'tdBN_on': tdBN_on,
            'BN_on': BN_on,
            'TIME': TIME,
            'surrogate': surrogate,
            'BPTT_on': BPTT_on,
            'OTTT_sWS_on': OTTT_sWS_on,
            'DFA_on': DFA_on,
            'drop_rate': drop_rate,
            'UDA_on': UDA_on,
            'alpha_uda': alpha_uda,
        }

        self.layers = make_layers_conv(cfg, in_c, IMAGE_SIZE,
                                    synapse_conv_kernel_size, synapse_conv_stride, 
                                    synapse_conv_padding, synapse_conv_trace_const1, 
                                    synapse_conv_trace_const2, 
                                    lif_layer_v_init, lif_layer_v_decay, 
                                    lif_layer_v_threshold, lif_layer_v_reset,
                                    lif_layer_sg_width,
                                    tdBN_on,
                                    BN_on, TIME,
                                    surrogate,
                                    BPTT_on,
                                    synapse_fc_out_features,
                                    OTTT_sWS_on,
                                    DFA_on,
                                    drop_rate,)
        
        self.just_shell = True

    def forward(self, spike_input):
        # inputs: [Batch, Time, Channel, Height, Width]   
        spike_input = spike_input.permute(1, 0, 2, 3, 4)
        # inputs: [Time, Batch, Channel, Height, Width]   
        spike_input = self.layers(spike_input)

        # spike_input = spike_input.sum(axis=0)
        spike_input = spike_input.mean(axis=0)
        return spike_input
    

def make_layers_conv(cfg, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_conv_trace_const1, 
                    synapse_conv_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    synapse_fc_out_features,
                    OTTT_sWS_on,
                    DFA_on,
                    drop_rate,):
    
    layers = []
    in_channels = in_c
    img_size_var = IMAGE_SIZE
    classifier_making = False
    first_conv = True
    for which in cfg:
        if (classifier_making == False):
            if type(which) == list:
                # residual block 
                layer = ResidualBlock_conv(which, in_channels, img_size_var,
                        synapse_conv_kernel_size, synapse_conv_stride, 
                        synapse_conv_padding, synapse_conv_trace_const1, 
                        synapse_conv_trace_const2, 
                        lif_layer_v_init, lif_layer_v_decay, 
                        lif_layer_v_threshold, lif_layer_v_reset,
                        lif_layer_sg_width,
                        tdBN_on,
                        BN_on, TIME,
                        surrogate,
                        BPTT_on,
                        synapse_fc_out_features,
                        OTTT_sWS_on,
                        first_conv,
                        DFA_on,
                        drop_rate,)
                assert in_channels == layer.in_channels, 'pre-residu, post-residu channel should be same'
                in_channels = layer.in_channels
                # print('\n\n\nimg_size_var !!!', img_size_var, 'layer.img_size_var', layer.img_size_var, 'which', which,'\n\n\n')
                assert img_size_var == layer.img_size_var, 'pre-residu, post-residu img_size_var should be same'
                img_size_var = layer.img_size_var
                layers.append( layer)
                first_conv = False
            elif which == 'P':
                layers += [DimChanger_for_pooling(nn.AvgPool2d(kernel_size=2, stride=2))]
                img_size_var = img_size_var // 2
            elif which == 'M':
                layers += [DimChanger_for_pooling(nn.MaxPool2d(kernel_size=2, stride=2))]
                img_size_var = img_size_var // 2
            elif which == 'D':
                layers += [DimChanger_for_pooling(nn.AdaptiveAvgPool2d((1, 1)))]
                img_size_var = 1
            elif which == 'L':
                classifier_making = True
                layers += [DimChanger_for_FC()]
                in_channels = in_channels*img_size_var*img_size_var
            else:
                if (which >= 10000 and which < 20000):
                    out_channels = which - 10000
                    layers += [SYNAPSE_SEPARABLE_CONV_BPTT(
                                            in_channels=in_channels,
                                            out_channels=out_channels, 
                                            kernel_size=synapse_conv_kernel_size, 
                                            stride=synapse_conv_stride, 
                                            padding=synapse_conv_padding, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME)]
                elif (which >= 20000 and which < 30000):
                    out_channels = which - 20000
                    layers += [SYNAPSE_DEPTHWISE_CONV_BPTT(
                                            in_channels=in_channels,
                                            out_channels=out_channels, 
                                            kernel_size=synapse_conv_kernel_size, 
                                            stride=synapse_conv_stride, 
                                            padding=synapse_conv_padding, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME)]
                else:
                    out_channels = which
                    if (BPTT_on == False):
                        # layers += [SYNAPSE_CONV(in_channels=in_channels,
                        #                         out_channels=out_channels, 
                        #                         kernel_size=synapse_conv_kernel_size, 
                        #                         stride=synapse_conv_stride, 
                        #                         padding=synapse_conv_padding, 
                        #                         trace_const1=synapse_conv_trace_const1, 
                        #                         trace_const2=synapse_conv_trace_const2,
                        #                         TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=first_conv)]
                        layers += [SYNAPSE_CONV_trace(in_channels=in_channels,
                                                out_channels=out_channels, 
                                                kernel_size=synapse_conv_kernel_size, 
                                                stride=synapse_conv_stride, 
                                                padding=synapse_conv_padding, 
                                                trace_const1=synapse_conv_trace_const1, 
                                                trace_const2=synapse_conv_trace_const2,
                                                TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=first_conv)]
                        
                    else:
                        layers += [SYNAPSE_CONV_BPTT(in_channels=in_channels,
                                                out_channels=out_channels, 
                                                kernel_size=synapse_conv_kernel_size, 
                                                stride=synapse_conv_stride, 
                                                padding=synapse_conv_padding, 
                                                trace_const1=synapse_conv_trace_const1, 
                                                trace_const2=synapse_conv_trace_const2,
                                                TIME=TIME)]
                        # layers += [SpikeModule(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))]
                first_conv = False
                
                img_size_var = (img_size_var - synapse_conv_kernel_size + 2*synapse_conv_padding)//synapse_conv_stride + 1
            
                in_channels = out_channels
                

                # batchnorm or tdBN 추가 ##########################
                if (tdBN_on == True):
                    layers += [tdBatchNorm(in_channels)] # 여기서 in_channel이 out_channel임

                if (BN_on == True):
                    layers += [BatchNorm(in_channels, TIME)]
                #################################################


                # LIF 뉴런 추가 ##################################
                if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                    if (BPTT_on == False):
                        # layers += [LIF_layer(v_init=lif_layer_v_init, 
                        #                         v_decay=lif_layer_v_decay, 
                        #                         v_threshold=lif_layer_v_threshold, 
                        #                         v_reset=lif_layer_v_reset, 
                        #                         sg_width=lif_layer_sg_width,
                        #                         surrogate=surrogate,
                        #                         BPTT_on=BPTT_on)]
                        layers += [LIF_layer_trace(v_init=lif_layer_v_init, 
                                                v_decay=lif_layer_v_decay, 
                                                v_threshold=lif_layer_v_threshold, 
                                                v_reset=lif_layer_v_reset, 
                                                sg_width=lif_layer_sg_width,
                                                surrogate=surrogate,
                                                BPTT_on=BPTT_on, 
                                                trace_const1=synapse_conv_trace_const1, 
                                                trace_const2=synapse_conv_trace_const2)]
                    else:
                        layers += [LIF_layer(v_init=lif_layer_v_init, 
                                                v_decay=lif_layer_v_decay, 
                                                v_threshold=lif_layer_v_threshold, 
                                                v_reset=lif_layer_v_reset, 
                                                sg_width=lif_layer_sg_width,
                                                surrogate=surrogate,
                                                BPTT_on=BPTT_on)]
                elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                    # NDA의 LIF 뉴런 쓰고 싶을 때 
                    lif_layer_v_threshold -= 10000
                    layers += [DimChanger_for_change_0_1()]
                    layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                                lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
                    layers += [DimChanger_for_change_0_1()]
                    lif_layer_v_threshold += 10000

                ## OTTT sWS하면 스케일링해줘야됨
                if OTTT_sWS_on == True:
                    layers += [Scale(2.74)]
                
                if drop_rate > 0:
                    assert False, 'not implemented yet'
                    # layers += [Dropout_sstep(drop_rate, TIME)]

                if DFA_on == True:
                    layers += [Feedback_Receiver(synapse_fc_out_features)]
                #################################################
                

        else: # classifier_making
            if (BPTT_on == False):
                layers += [SYNAPSE_FC_trace(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                                out_features=which, 
                                                trace_const1=synapse_conv_trace_const1, 
                                                trace_const2=synapse_conv_trace_const2,
                                                TIME=TIME)]
            else:
                layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                                out_features=which, 
                                                trace_const1=synapse_conv_trace_const1, 
                                                trace_const2=synapse_conv_trace_const2,
                                                TIME=TIME)]
            in_channels = which

            
            # LIF 뉴런 추가 ##################################
            if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                if (BPTT_on == False):
                    # layers += [LIF_layer(v_init=lif_layer_v_init, 
                    #                         v_decay=lif_layer_v_decay, 
                    #                         v_threshold=lif_layer_v_threshold, 
                    #                         v_reset=lif_layer_v_reset, 
                    #                         sg_width=lif_layer_sg_width,
                    #                         surrogate=surrogate,
                    #                         BPTT_on=BPTT_on)]
                    layers += [LIF_layer_trace(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2)]
                else:
                    layers += [LIF_layer(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on)]
            elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                # NDA의 LIF 뉴런 쓰고 싶을 때 
                lif_layer_v_threshold -= 10000
                layers += [DimChanger_for_change_0_1()]
                layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                            lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
                layers += [DimChanger_for_change_0_1()]
                lif_layer_v_threshold += 10000
            
            if drop_rate > 0:
                assert False, 'not implemented yet'
                # layers += [Dropout_sstep(drop_rate, TIME)]

            if DFA_on == True:
                layers += [Feedback_Receiver(synapse_fc_out_features)]
            #################################################
                

    if classifier_making == False: # cfg에 'L'한번도 없을때
        layers += [DimChanger_for_FC()]
        in_channels = in_channels*img_size_var*img_size_var
        
    if (BPTT_on == False):
        # layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
        #                                 out_features=synapse_fc_out_features, 
        #                                 trace_const1=synapse_conv_trace_const1, 
        #                                 trace_const2=synapse_conv_trace_const2,
        #                                 TIME=TIME)]
        layers += [SYNAPSE_FC_trace(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                        out_features=synapse_fc_out_features, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
    else:
        layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                        out_features=synapse_fc_out_features, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]

    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=synapse_fc_out_features)



class ResidualBlock_conv(nn.Module):
    def __init__(self, layers, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_conv_trace_const1, 
                    synapse_conv_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    synapse_fc_out_features,
                    OTTT_sWS_on,
                    first_conv,
                    DFA_on,
                    drop_rate,):
        super(ResidualBlock_conv, self).__init__()
        assert DFA_on == False, 'not implemented yet DFA & residual block'
        self.layers, self.in_channels, self.img_size_var= make_layers_conv_residual(layers, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_conv_trace_const1, 
                    synapse_conv_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    synapse_fc_out_features,
                    OTTT_sWS_on,
                    first_conv,
                    DFA_on,
                    drop_rate,)
        
        self.just_shell = True
    def forward(self, x):
        return self.layers(x)
     

def make_layers_conv_residual(cfg, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_conv_trace_const1, 
                    synapse_conv_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    synapse_fc_out_features,
                    OTTT_sWS_on,
                    first_conv,
                    DFA_on,
                    drop_rate,):
    assert DFA_on == False, 'not implemented yet DFA & residual block'
    
    layers = []
    in_channels = in_c
    img_size_var = IMAGE_SIZE
    for which in cfg:
        if which == 'P':
            layers += [DimChanger_for_pooling(nn.AvgPool2d(kernel_size=2, stride=2))]
            img_size_var = img_size_var // 2
        elif which == 'M':
            layers += [DimChanger_for_pooling(nn.MaxPool2d(kernel_size=2, stride=2))]
            img_size_var = img_size_var // 2
        elif which == 'D':
            layers += [DimChanger_for_pooling(nn.AdaptiveAvgPool2d((1, 1)))]
            img_size_var = 1
        else:
            if (which >= 10000 and which < 20000):
                out_channels = which - 10000
                layers += [SYNAPSE_SEPARABLE_CONV_BPTT(
                                        in_channels=in_channels,
                                        out_channels=out_channels, 
                                        kernel_size=synapse_conv_kernel_size, 
                                        stride=synapse_conv_stride, 
                                        padding=synapse_conv_padding, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
                
            elif (which >= 20000 and which < 30000):
                out_channels = which - 20000
                layers += [SYNAPSE_DEPTHWISE_CONV_BPTT(
                                        in_channels=in_channels,
                                        out_channels=out_channels, 
                                        kernel_size=synapse_conv_kernel_size, 
                                        stride=synapse_conv_stride, 
                                        padding=synapse_conv_padding, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
            
            
            else: 
                out_channels = which
                if (BPTT_on == False):
                    # layers += [SYNAPSE_CONV(in_channels=in_channels,
                    #                             out_channels=out_channels, 
                    #                             kernel_size=synapse_conv_kernel_size, 
                    #                             stride=synapse_conv_stride, 
                    #                             padding=synapse_conv_padding, 
                    #                             trace_const1=synapse_conv_trace_const1, 
                    #                             trace_const2=synapse_conv_trace_const2,
                    #                             TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=False)]
                    layers += [SYNAPSE_CONV_trace(in_channels=in_channels,
                                            out_channels=out_channels, 
                                            kernel_size=synapse_conv_kernel_size, 
                                            stride=synapse_conv_stride, 
                                            padding=synapse_conv_padding, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=first_conv)]
                else:
                    layers += [SYNAPSE_CONV_BPTT(in_channels=in_channels,
                                            out_channels=out_channels, 
                                            kernel_size=synapse_conv_kernel_size, 
                                            stride=synapse_conv_stride, 
                                            padding=synapse_conv_padding, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME)]
            first_conv = False
            img_size_var = (img_size_var - synapse_conv_kernel_size + 2*synapse_conv_padding)//synapse_conv_stride + 1
            in_channels = out_channels
            
            if (tdBN_on == True):
                layers += [tdBatchNorm(in_channels)] # 여기서 in_channel이 out_channel임

            if (BN_on == True):
                layers += [BatchNorm(in_channels, TIME)]

            # LIF 뉴런 추가 ##################################
            if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                if BPTT_on == False:
                    # layers += [LIF_layer(v_init=lif_layer_v_init, 
                    #                         v_decay=lif_layer_v_decay, 
                    #                         v_threshold=lif_layer_v_threshold, 
                    #                         v_reset=lif_layer_v_reset, 
                    #                         sg_width=lif_layer_sg_width,
                    #                         surrogate=surrogate,
                    #                         BPTT_on=BPTT_on)]
                    layers += [LIF_layer_trace(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2)]
                else:
                    layers += [LIF_layer(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on)]
            elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                # NDA의 LIF 뉴런 쓰고 싶을 때 
                lif_layer_v_threshold -= 10000
                layers += [DimChanger_for_change_0_1()]
                layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                            lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
                layers += [DimChanger_for_change_0_1()]
                lif_layer_v_threshold += 10000
                
            if drop_rate > 0:
                assert False, 'not implemented yet'
                # layers += [Dropout_sstep(drop_rate, TIME)]
            #################################################

    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=synapse_fc_out_features), in_channels, img_size_var
######## make_layers for Conv ############################################
######## make_layers for Conv ############################################
######## make_layers for Conv ############################################






######## make_layers for FC ############################################
######## make_layers for FC ############################################
######## make_layers for FC ############################################
class MY_SNN_FC(nn.Module):
    def __init__(self, cfg, in_c, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    drop_rate,
                    UDA_on,
                    alpha_uda):
        super(MY_SNN_FC, self).__init__()

        self.params = {
            'cfg': cfg,
            'in_c': in_c,
            'IMAGE_SIZE': IMAGE_SIZE,
            'out_c': out_c,
            'synapse_fc_trace_const1': synapse_fc_trace_const1,
            'synapse_fc_trace_const2': synapse_fc_trace_const2,
            'lif_layer_v_init': lif_layer_v_init,
            'lif_layer_v_decay': lif_layer_v_decay,
            'lif_layer_v_threshold': lif_layer_v_threshold,
            'lif_layer_v_reset': lif_layer_v_reset,
            'lif_layer_sg_width': lif_layer_sg_width,
            'tdBN_on': tdBN_on,
            'BN_on': BN_on,
            'TIME': TIME,
            'surrogate': surrogate,
            'BPTT_on': BPTT_on,
            'DFA_on': DFA_on,
            'drop_rate': drop_rate,
            'UDA_on': UDA_on,
            'alpha_uda': alpha_uda,
        }


        self.layers = make_layers_fc(cfg, in_c, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    drop_rate,)

        self.just_shell = True
    def forward(self, spike_input):
        # inputs: [Batch, Time, Channel, Height, Width]   
        spike_input = spike_input.permute(1, 0, 2, 3, 4)
        # inputs: [Time, Batch, Channel, Height, Width]   
        # spike_input = spike_input.view(spike_input.size(0), spike_input.size(1), -1)
        
        spike_input = self.layers(spike_input)

        spike_input = spike_input.mean(axis=0)
        # spike_input = spike_input.sum(axis=0)

        return spike_input
    

def make_layers_fc(cfg, in_c, IMAGE_SIZE, out_c,
                        synapse_fc_trace_const1, synapse_fc_trace_const2, 
                        lif_layer_v_init, lif_layer_v_decay, 
                        lif_layer_v_threshold, lif_layer_v_reset,
                        lif_layer_sg_width,
                        tdBN_on,
                        BN_on, TIME,
                        surrogate,
                        BPTT_on,
                        DFA_on,
                        drop_rate,):

    layers = []
    img_size = IMAGE_SIZE
    in_channels = in_c * img_size * img_size
    class_num = out_c
    pre_pooling_done = False
    for which in cfg:
        if type(which) == list:
            if (pre_pooling_done == False):
                layers += [DimChanger_for_FC()]
                pre_pooling_done = True
            # residual block 
            layer = ResidualBlock_fc(which, in_channels, IMAGE_SIZE, out_c,
                        synapse_fc_trace_const1, synapse_fc_trace_const2, 
                        lif_layer_v_init, lif_layer_v_decay, 
                        lif_layer_v_threshold, lif_layer_v_reset,
                        lif_layer_sg_width,
                        tdBN_on,
                        BN_on, TIME,
                        surrogate,
                        BPTT_on,
                        DFA_on)
            
            assert in_channels == layer.in_channels, 'pre-residu, post-residu channel should be same'
            in_channels = layer.in_channels
            layers.append( layer)
        elif which == 'P':
            assert pre_pooling_done == False, 'you must not do pooling after FC'
            layers += [DimChanger_for_pooling(nn.AvgPool2d(kernel_size=2, stride=2))]
            img_size = img_size // 2
            in_channels = in_c * img_size * img_size
        elif which == 'M':
            assert pre_pooling_done == False, 'you must not do pooling after FC'
            layers += [DimChanger_for_pooling(nn.MaxPool2d(kernel_size=2, stride=2))]
            img_size = img_size // 2
            in_channels = in_c * img_size * img_size
        else:
            if (pre_pooling_done == False):
                layers += [DimChanger_for_FC()]
                pre_pooling_done = True
            out_channels = which
            if(BPTT_on == False):
                # layers += [SYNAPSE_FC(in_features=in_channels, 
                #                             out_features=out_channels, 
                #                             trace_const1=synapse_fc_trace_const1, 
                #                             trace_const2=synapse_fc_trace_const2,
                #                             TIME=TIME)]
                layers += [SYNAPSE_FC_trace(in_features=in_channels, 
                                            out_features=out_channels, 
                                            trace_const1=synapse_fc_trace_const1, 
                                            trace_const2=synapse_fc_trace_const2,
                                            TIME=TIME)]
            else:
                layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                            out_features=out_channels, 
                                            trace_const1=synapse_fc_trace_const1, 
                                            trace_const2=synapse_fc_trace_const2,
                                            TIME=TIME)]
            in_channels = which
        
            if (tdBN_on == True):
                layers += [tdBatchNorm_FC(in_channels)] # 여기서 in_channel이 out_channel임

            if (BN_on == True):
                layers += [BatchNorm_FC(in_channels, TIME)]


            # LIF 뉴런 추가 ##################################
            if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                if(BPTT_on == False):
                    # layers += [LIF_layer(v_init=lif_layer_v_init, 
                    #                         v_decay=lif_layer_v_decay, 
                    #                         v_threshold=lif_layer_v_threshold, 
                    #                         v_reset=lif_layer_v_reset, 
                    #                         sg_width=lif_layer_sg_width,
                    #                         surrogate=surrogate,
                    #                         BPTT_on=BPTT_on)]
                    layers += [LIF_layer_trace(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on, 
                                            trace_const1=synapse_fc_trace_const1, 
                                            trace_const2=synapse_fc_trace_const2)]
                else:
                    layers += [LIF_layer(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on)]
            elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                # NDA의 LIF 뉴런 쓰고 싶을 때 
                lif_layer_v_threshold -= 10000
                layers += [DimChanger_for_change_0_1()]
                layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                            lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
                layers += [DimChanger_for_change_0_1()]
                lif_layer_v_threshold += 10000
                
            if drop_rate > 0:
                assert False, 'not implemented yet'
                # layers += [Dropout_sstep(drop_rate, TIME)]

            if DFA_on == True:
                layers += [Feedback_Receiver(class_num)]
            #################################################

    
    out_channels = class_num
    if(BPTT_on == False):
        # layers += [SYNAPSE_FC(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
        #                             out_features=out_channels, 
        #                             trace_const1=synapse_fc_trace_const1, 
        #                             trace_const2=synapse_fc_trace_const2,
        #                             TIME=TIME)]
        layers += [SYNAPSE_FC_trace(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                    out_features=out_channels, 
                                    trace_const1=synapse_fc_trace_const1, 
                                    trace_const2=synapse_fc_trace_const2,
                                    TIME=TIME)]
    else:
        layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                    out_features=out_channels, 
                                    trace_const1=synapse_fc_trace_const1, 
                                    trace_const2=synapse_fc_trace_const2,
                                    TIME=TIME)]

    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=class_num)



class ResidualBlock_fc(nn.Module):
    def __init__(self, layers, in_channels, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    drop_rate,):
        super(ResidualBlock_fc, self).__init__()
        assert DFA_on == False, 'not implemented yet DFA & residual block'
        self.layers, self.in_channels = make_layers_fc_residual(layers, in_channels, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    drop_rate,)
    
        self.just_shell = True
    def forward(self, x):
        return self.layers(x)
    


def make_layers_fc_residual(cfg, in_c, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    drop_rate,):
    assert DFA_on == False, 'not implemented yet DFA & residual block'

    layers = []
    img_size = IMAGE_SIZE
    in_channels = in_c
    class_num = out_c
    for which in cfg:

        out_channels = which

        if(BPTT_on == False):
            # layers += [SYNAPSE_FC(in_features=in_channels, 
            #                             out_features=out_channels, 
            #                             trace_const1=synapse_fc_trace_const1, 
            #                             trace_const2=synapse_fc_trace_const2,
            #                             TIME=TIME)]
            layers += [SYNAPSE_FC_trace(in_features=in_channels,  
                                        out_features=out_channels, 
                                        trace_const1=synapse_fc_trace_const1, 
                                        trace_const2=synapse_fc_trace_const2,
                                        TIME=TIME)]
        else:
            layers += [SYNAPSE_FC_BPTT(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                        out_features=out_channels, 
                                        trace_const1=synapse_fc_trace_const1, 
                                        trace_const2=synapse_fc_trace_const2,
                                        TIME=TIME)]
            


        in_channels = which
        
        if (tdBN_on == True):
            layers += [tdBatchNorm_FC(in_channels)] # 여기서 in_channel이 out_channel임

        if (BN_on == True):
            layers += [BatchNorm_FC(in_channels, TIME)]

        

        # LIF 뉴런 추가 ##################################
        if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
            if(BPTT_on == False):
                # layers += [LIF_layer(v_init=lif_layer_v_init, 
                #                         v_decay=lif_layer_v_decay, 
                #                         v_threshold=lif_layer_v_threshold, 
                #                         v_reset=lif_layer_v_reset, 
                #                         sg_width=lif_layer_sg_width,
                #                         surrogate=surrogate,
                #                         BPTT_on=BPTT_on)]
                layers += [LIF_layer_trace(v_init=lif_layer_v_init, 
                                        v_decay=lif_layer_v_decay, 
                                        v_threshold=lif_layer_v_threshold, 
                                        v_reset=lif_layer_v_reset, 
                                        sg_width=lif_layer_sg_width,
                                        surrogate=surrogate,
                                        BPTT_on=BPTT_on, 
                                        trace_const1=synapse_fc_trace_const1, 
                                        trace_const2=synapse_fc_trace_const2)]
            else:
                layers += [LIF_layer(v_init=lif_layer_v_init, 
                                        v_decay=lif_layer_v_decay, 
                                        v_threshold=lif_layer_v_threshold, 
                                        v_reset=lif_layer_v_reset, 
                                        sg_width=lif_layer_sg_width,
                                        surrogate=surrogate,
                                        BPTT_on=BPTT_on)]
        elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
            # NDA의 LIF 뉴런 쓰고 싶을 때 
            lif_layer_v_threshold -= 10000
            layers += [DimChanger_for_change_0_1()]
            layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                        lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
            layers += [DimChanger_for_change_0_1()]
            lif_layer_v_threshold += 10000
                    
        if drop_rate > 0:
            assert False, 'not implemented yet'
            # layers += [Dropout_sstep(drop_rate, TIME)]
        #################################################

    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=class_num), in_channels
######## make_layers for FC ############################################
######## make_layers for FC ############################################
######## make_layers for FC ############################################






####### make_layers for ottt conv single step ############################################
####### make_layers for ottt conv single step ############################################
####### make_layers for ottt conv single step ############################################
class MY_SNN_CONV_sstep(nn.Module):
    def __init__(self, cfg, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_conv_trace_const1, 
                    synapse_conv_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    synapse_fc_out_features, synapse_fc_trace_const1, synapse_fc_trace_const2,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    OTTT_sWS_on,
                    DFA_on,
                    drop_rate,
                    UDA_on,
                    alpha_uda,):
        super(MY_SNN_CONV_sstep, self).__init__()        
        self.params = {
            'cfg': cfg,
            'in_c': in_c,
            'IMAGE_SIZE': IMAGE_SIZE,
            'synapse_conv_kernel_size': synapse_conv_kernel_size,
            'synapse_conv_stride': synapse_conv_stride,
            'synapse_conv_padding': synapse_conv_padding,
            'synapse_conv_trace_const1': synapse_conv_trace_const1,
            'synapse_conv_trace_const2': synapse_conv_trace_const2,
            'lif_layer_v_init': lif_layer_v_init,
            'lif_layer_v_decay': lif_layer_v_decay,
            'lif_layer_v_threshold': lif_layer_v_threshold,
            'lif_layer_v_reset': lif_layer_v_reset,
            'lif_layer_sg_width': lif_layer_sg_width,
            'synapse_fc_out_features': synapse_fc_out_features,
            'synapse_fc_trace_const1': synapse_fc_trace_const1,
            'synapse_fc_trace_const2': synapse_fc_trace_const2,
            'tdBN_on': tdBN_on,
            'BN_on': BN_on,
            'TIME': TIME,
            'surrogate': surrogate,
            'BPTT_on': BPTT_on,
            'OTTT_sWS_on': OTTT_sWS_on,
            'DFA_on': DFA_on,
            'drop_rate': drop_rate,
            'UDA_on': UDA_on,
            'alpha_uda': alpha_uda,
        }

        if (self.params['UDA_on'] == True):
            self.UDA_classifier_on = True
            self.UDA_adapter_on = True
            # cfg == ([200,200],[200],[200],[1]) 이런식으로 들어옴 맨 앞이 feature, 그 다음 게 classifier, 그 다음 게 adapter, 마지막이 도메인 개수 
            self.UDA_feature_layers, feature_size = make_layers_conv_sstep_UDA_feature(cfg[0], in_c, IMAGE_SIZE,
                                        synapse_conv_kernel_size, synapse_conv_stride, 
                                        synapse_conv_padding, synapse_conv_trace_const1, 
                                        synapse_conv_trace_const2, 
                                        lif_layer_v_init, lif_layer_v_decay, 
                                        lif_layer_v_threshold, lif_layer_v_reset,
                                        lif_layer_sg_width,
                                        tdBN_on,
                                        BN_on, TIME,
                                        surrogate,
                                        BPTT_on,
                                        synapse_fc_out_features,
                                        OTTT_sWS_on,
                                        DFA_on,
                                        drop_rate,)
            self.UDA_classifier_layers = make_layers_fc_sstep_UDA_classifier(cfg[1], feature_size, IMAGE_SIZE, synapse_fc_out_features,
                        synapse_fc_trace_const1, synapse_fc_trace_const2, 
                        lif_layer_v_init, lif_layer_v_decay, 
                        lif_layer_v_threshold, lif_layer_v_reset,
                        lif_layer_sg_width,
                        tdBN_on,
                        BN_on, TIME,
                        surrogate,
                        BPTT_on,
                        DFA_on,
                        OTTT_sWS_on,
                        drop_rate,)
            self.UDA_adapter_layers = make_layers_fc_sstep_UDA_adapter(cfg[2], feature_size, IMAGE_SIZE, cfg[3][0],
                        synapse_fc_trace_const1, synapse_fc_trace_const2, 
                        lif_layer_v_init, lif_layer_v_decay, 
                        lif_layer_v_threshold, lif_layer_v_reset,
                        lif_layer_sg_width,
                        tdBN_on,
                        BN_on, TIME,
                        surrogate,
                        BPTT_on,
                        DFA_on,
                        OTTT_sWS_on,
                        drop_rate,
                        alpha_uda,)
        else:
            self.layers = make_layers_conv_sstep(cfg, in_c, IMAGE_SIZE,
                                        synapse_conv_kernel_size, synapse_conv_stride, 
                                        synapse_conv_padding, synapse_conv_trace_const1, 
                                        synapse_conv_trace_const2, 
                                        lif_layer_v_init, lif_layer_v_decay, 
                                        lif_layer_v_threshold, lif_layer_v_reset,
                                        lif_layer_sg_width,
                                        tdBN_on,
                                        BN_on, TIME,
                                        surrogate,
                                        BPTT_on,
                                        synapse_fc_out_features,
                                        OTTT_sWS_on,
                                        DFA_on,
                                        drop_rate,)

        self.just_shell = True
    def forward(self, spike_input):
        # inputs: [Batch, Channel, Height, Width]   
        if (self.params['UDA_on'] == True):
            UDA_feature = self.UDA_feature_layers(spike_input)

            if self.UDA_adapter_on == True:
                UDA_adapter_out = self.UDA_adapter_layers(UDA_feature)
            else:
                assert False, 'UDA_adapter_on should be True'
            
            if self.UDA_classifier_on == True or self.UDA_classifier_on == False :
                UDA_classifier_out = self.UDA_classifier_layers(UDA_feature)
                return UDA_classifier_out, UDA_adapter_out
            else:
                # 일단 둘 다 classifier는 다 내보내게 하자. 밖에서 .backward만 안해주면 되지 않나?
                pass
        else:
            spike_input = self.layers(spike_input)
            # spike_input = spike_input.sum(axis=0)
            # spike_input = spike_input.mean(axis=0)
            return spike_input
    

def make_layers_conv_sstep(cfg, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_conv_trace_const1, 
                    synapse_conv_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    synapse_fc_out_features,
                    OTTT_sWS_on,
                    DFA_on,
                    drop_rate,):
    assert BPTT_on == False, 'BPTT_on should be False'
    layers = []
    in_channels = in_c
    img_size_var = IMAGE_SIZE
    classifier_making = False
    first_conv = True
    for which in cfg:
        if (classifier_making == False):
            if type(which) == list:
                # residual block 
                layer = ResidualBlock_conv_sstep(which, in_channels, img_size_var,
                        synapse_conv_kernel_size, synapse_conv_stride, 
                        synapse_conv_padding, synapse_conv_trace_const1, 
                        synapse_conv_trace_const2, 
                        lif_layer_v_init, lif_layer_v_decay, 
                        lif_layer_v_threshold, lif_layer_v_reset,
                        lif_layer_sg_width,
                        tdBN_on,
                        BN_on, TIME,
                        surrogate,
                        BPTT_on,
                        synapse_fc_out_features,
                        OTTT_sWS_on,
                        first_conv,
                        DFA_on,
                        drop_rate,)
                assert in_channels == layer.in_channels, 'pre-residu, post-residu channel should be same'
                in_channels = layer.in_channels
                # print('\n\n\nimg_size_var !!!', img_size_var, 'layer.img_size_var', layer.img_size_var, 'which', which,'\n\n\n')
                assert img_size_var == layer.img_size_var, 'pre-residu, post-residu img_size_var should be same'
                img_size_var = layer.img_size_var
                layers.append( layer)
            elif which == 'P':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                img_size_var = img_size_var // 2
            elif which == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                img_size_var = img_size_var // 2
            elif which == 'D':
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
                img_size_var = 1
            elif which == 'L':
                classifier_making = True
                layers += [DimChanger_for_FC_sstep()]
                in_channels = in_channels*img_size_var*img_size_var
            else:
                if (which >= 10000 and which < 20000):
                    assert False, 'not implemented'
                elif (which >= 20000 and which < 30000):
                    assert False, 'not implemented'
                else:
                    out_channels = which
                    if (layers == []):
                        first_conv = True
                    else:
                        first_conv = False
                    layers += [SYNAPSE_CONV_trace_sstep(in_channels=in_channels,
                                            out_channels=out_channels, 
                                            kernel_size=synapse_conv_kernel_size, 
                                            stride=synapse_conv_stride, 
                                            padding=synapse_conv_padding, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=first_conv)]
                
                img_size_var = (img_size_var - synapse_conv_kernel_size + 2*synapse_conv_padding)//synapse_conv_stride + 1
            
                in_channels = out_channels
                

                # batchnorm or tdBN 추가 ##########################
                if (tdBN_on == True):
                    assert False, 'impossible in single step mode'

                if (BN_on == True):
                    layers += nn.BatchNorm2d(in_channels)
                #################################################


                # LIF 뉴런 추가 ##################################
                if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                    layers += [LIF_layer_trace_sstep(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME)]
                elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                    assert False
                else:
                    assert False
                
                ## OTTT sWS하면 스케일링해줘야됨
                if OTTT_sWS_on == True:
                    layers += [Scale(2.74)]

                if drop_rate > 0:
                    layers += [Dropout_sstep(drop_rate, TIME)]

                if DFA_on == True:
                    layers += [Feedback_Receiver(synapse_fc_out_features)]
                #################################################
                

        else: # classifier_making
            layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  
                                            out_features=which, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME,
                                            OTTT_sWS_on=OTTT_sWS_on,)]
            in_channels = which

            # LIF 뉴런 추가 ##################################
            if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                if (BPTT_on == False):
                    # layers += [LIF_layer(v_init=lif_layer_v_init, 
                    #                         v_decay=lif_layer_v_decay, 
                    #                         v_threshold=lif_layer_v_threshold, 
                    #                         v_reset=lif_layer_v_reset, 
                    #                         sg_width=lif_layer_sg_width,
                    #                         surrogate=surrogate,
                    #                         BPTT_on=BPTT_on)]
                    layers += [LIF_layer_trace(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2)]
                else:
                    layers += [LIF_layer(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on)]
            elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                # NDA의 LIF 뉴런 쓰고 싶을 때 
                lif_layer_v_threshold -= 10000
                layers += [DimChanger_for_change_0_1()]
                layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                            lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
                layers += [DimChanger_for_change_0_1()]
                lif_layer_v_threshold += 10000

            ## OTTT sWS하면 스케일링해줘야됨
            if OTTT_sWS_on == True:
                layers += [Scale(2.74)]

            if drop_rate > 0:
                layers += [Dropout_sstep(drop_rate, TIME)]
                
            if DFA_on == True:
                layers += [Feedback_Receiver(synapse_fc_out_features)]
            #################################################

    if classifier_making == False: # cfg에 'L'한번도 없을때
        layers += [DimChanger_for_FC_sstep()]
        in_channels = in_channels*img_size_var*img_size_var
        
    
    layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  # 마지막CONV의 OUT_CHANNEL * H * W
                                    out_features=synapse_fc_out_features, 
                                    trace_const1=synapse_conv_trace_const1, 
                                    trace_const2=synapse_conv_trace_const2,
                                    TIME=TIME,
                                    OTTT_sWS_on=False,)]
    
    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=synapse_fc_out_features)

    
def make_layers_conv_sstep_UDA_feature(cfg, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_conv_trace_const1, 
                    synapse_conv_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    synapse_fc_out_features,
                    OTTT_sWS_on,
                    DFA_on,
                    drop_rate,):
    assert BPTT_on == False, 'BPTT_on should be False'
    layers = []
    in_channels = in_c
    img_size_var = IMAGE_SIZE
    classifier_making = False
    first_conv = True
    for which in cfg:
        if (classifier_making == False):
            if type(which) == list:
                # residual block 
                assert False, 'not implemented yet'
            elif which == 'P':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                img_size_var = img_size_var // 2
            elif which == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                img_size_var = img_size_var // 2
            elif which == 'D':
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
                img_size_var = 1
            elif which == 'L':
                classifier_making = True
                layers += [DimChanger_for_FC_sstep()]
                in_channels = in_channels*img_size_var*img_size_var
            else:
                if (which >= 10000 and which < 20000):
                    assert False, 'not implemented'
                elif (which >= 20000 and which < 30000):
                    assert False, 'not implemented'
                else:
                    out_channels = which
                    if (layers == []):
                        first_conv = True
                    else:
                        first_conv = False
                    layers += [SYNAPSE_CONV_trace_sstep(in_channels=in_channels,
                                            out_channels=out_channels, 
                                            kernel_size=synapse_conv_kernel_size, 
                                            stride=synapse_conv_stride, 
                                            padding=synapse_conv_padding, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=first_conv)]
                
                img_size_var = (img_size_var - synapse_conv_kernel_size + 2*synapse_conv_padding)//synapse_conv_stride + 1
            
                in_channels = out_channels
                

                # batchnorm or tdBN 추가 ##########################
                if (tdBN_on == True):
                    assert False, 'impossible in single step mode'

                if (BN_on == True):
                    layers += nn.BatchNorm2d(in_channels)
                #################################################


                # LIF 뉴런 추가 ##################################
                if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                    layers += [LIF_layer_trace_sstep(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME)]
                elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                    assert False
                else:
                    assert False
                
                ## OTTT sWS하면 스케일링해줘야됨
                if OTTT_sWS_on == True:
                    layers += [Scale(2.74)]

                if drop_rate > 0:
                    layers += [Dropout_sstep(drop_rate, TIME)]

                if DFA_on == True:
                    layers += [Feedback_Receiver(synapse_fc_out_features)]
                #################################################
                

        else: # classifier_making
            layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  
                                            out_features=which, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2,
                                            TIME=TIME,
                                            OTTT_sWS_on=OTTT_sWS_on,)]
            in_channels = which

            # LIF 뉴런 추가 ##################################
            if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                if (BPTT_on == False):
                    # layers += [LIF_layer(v_init=lif_layer_v_init, 
                    #                         v_decay=lif_layer_v_decay, 
                    #                         v_threshold=lif_layer_v_threshold, 
                    #                         v_reset=lif_layer_v_reset, 
                    #                         sg_width=lif_layer_sg_width,
                    #                         surrogate=surrogate,
                    #                         BPTT_on=BPTT_on)]
                    layers += [LIF_layer_trace(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on, 
                                            trace_const1=synapse_conv_trace_const1, 
                                            trace_const2=synapse_conv_trace_const2)]
                else:
                    layers += [LIF_layer(v_init=lif_layer_v_init, 
                                            v_decay=lif_layer_v_decay, 
                                            v_threshold=lif_layer_v_threshold, 
                                            v_reset=lif_layer_v_reset, 
                                            sg_width=lif_layer_sg_width,
                                            surrogate=surrogate,
                                            BPTT_on=BPTT_on)]
            elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                # NDA의 LIF 뉴런 쓰고 싶을 때 
                lif_layer_v_threshold -= 10000
                layers += [DimChanger_for_change_0_1()]
                layers += [LIFSpike(lif_layer_v_threshold = lif_layer_v_threshold, 
                            lif_layer_v_decay = lif_layer_v_decay, lif_layer_sg_width = lif_layer_sg_width)] # 이거 걍 **lif_parameters에 아무것도 없어도 default값으로 알아서 됨.
                layers += [DimChanger_for_change_0_1()]
                lif_layer_v_threshold += 10000

            ## OTTT sWS하면 스케일링해줘야됨
            if OTTT_sWS_on == True:
                layers += [Scale(2.74)]

            if drop_rate > 0:
                layers += [Dropout_sstep(drop_rate, TIME)]
                
            if DFA_on == True:
                layers += [Feedback_Receiver(synapse_fc_out_features)]
            #################################################

    if classifier_making == False: # cfg에 'L'한번도 없을때
        layers += [DimChanger_for_FC_sstep()]
        in_channels = in_channels*img_size_var*img_size_var
        feature_size = in_channels
    
    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=synapse_fc_out_features), feature_size

    

class ResidualBlock_conv_sstep(nn.Module):
    def __init__(self, layers, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_conv_trace_const1, 
                    synapse_conv_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    synapse_fc_out_features,
                    OTTT_sWS_on,
                    first_conv,
                    DFA_on,
                    drop_rate,):
        super(ResidualBlock_conv_sstep, self).__init__()
        assert DFA_on == False, 'not implemented yet DFA & residual block'
        self.layers, self.in_channels, self.img_size_var= make_layers_conv_residual_sstep(layers, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_conv_trace_const1, 
                    synapse_conv_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    synapse_fc_out_features,
                    OTTT_sWS_on,
                    first_conv,
                    DFA_on,
                    drop_rate,)
    
        self.just_shell = True
    def forward(self, x):
        return self.layers(x)
     

def make_layers_conv_residual_sstep(cfg, in_c, IMAGE_SIZE,
                    synapse_conv_kernel_size, synapse_conv_stride, 
                    synapse_conv_padding, synapse_conv_trace_const1, 
                    synapse_conv_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    synapse_fc_out_features,
                    OTTT_sWS_on,
                    first_conv,
                    DFA_on,
                    drop_rate,):
    assert DFA_on == False, 'not implemented yet DFA & residual block'
    assert BPTT_on == False, 'BPTT_on should be False'
    layers = []
    in_channels = in_c
    img_size_var = IMAGE_SIZE
    for which in cfg:
        if which == 'P':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            img_size_var = img_size_var // 2
        elif which == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            img_size_var = img_size_var // 2
        elif which == 'D':
            layers += [nn.AdaptiveAvgPool2d((1, 1))]
            img_size_var = 1
        else:
            if (which >= 10000 and which < 20000):
                assert False, 'not implemented'
            elif (which >= 20000 and which < 30000):
                assert False, 'not implemented'
            else:
                out_channels = which
                if (layers == []):
                    first_conv = True
                else:
                    first_conv = False
                layers += [SYNAPSE_CONV_trace_sstep(in_channels=in_channels,
                                        out_channels=out_channels, 
                                        kernel_size=synapse_conv_kernel_size, 
                                        stride=synapse_conv_stride, 
                                        padding=synapse_conv_padding, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME, OTTT_sWS_on=OTTT_sWS_on, first_conv=first_conv)]
            
            img_size_var = (img_size_var - synapse_conv_kernel_size + 2*synapse_conv_padding)//synapse_conv_stride + 1
        
            in_channels = out_channels
            

            # batchnorm or tdBN 추가 ##########################
            if (tdBN_on == True):
                assert False, 'impossible in single step mode'

            if (BN_on == True):
                layers += nn.BatchNorm2d(in_channels)
            #################################################


            # LIF 뉴런 추가 ##################################
            if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                layers += [LIF_layer_trace_sstep(v_init=lif_layer_v_init, 
                                        v_decay=lif_layer_v_decay, 
                                        v_threshold=lif_layer_v_threshold, 
                                        v_reset=lif_layer_v_reset, 
                                        sg_width=lif_layer_sg_width,
                                        surrogate=surrogate,
                                        BPTT_on=BPTT_on, 
                                        trace_const1=synapse_conv_trace_const1, 
                                        trace_const2=synapse_conv_trace_const2,
                                        TIME=TIME)]
            elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                assert False
            else:
                assert False
            #################################################
            
            ## OTTT sWS하면 스케일링해줘야됨
            if OTTT_sWS_on == True:
                layers += [Scale(2.74)]

            if drop_rate > 0:
                layers += [Dropout_sstep(drop_rate, TIME)]
    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=synapse_fc_out_features), in_channels, img_size_var
####### make_layers for ottt conv single step ############################################
####### make_layers for ottt conv single step ############################################
####### make_layers for ottt conv single step ############################################



####### make_layers for ottt fc single step ############################################
####### make_layers for ottt fc single step ############################################
####### make_layers for ottt fc single step ############################################
class MY_SNN_FC_sstep(nn.Module):
    def __init__(self, cfg, in_c, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    OTTT_sWS_on,
                    drop_rate,
                    UDA_on,
                    alpha_uda,):
        super(MY_SNN_FC_sstep, self).__init__()
        self.params = {
            'cfg': cfg,
            'in_c': in_c,
            'IMAGE_SIZE': IMAGE_SIZE,
            'out_c': out_c,
            'synapse_fc_trace_const1': synapse_fc_trace_const1,
            'synapse_fc_trace_const2': synapse_fc_trace_const2,
            'lif_layer_v_init': lif_layer_v_init,
            'lif_layer_v_decay': lif_layer_v_decay,
            'lif_layer_v_threshold': lif_layer_v_threshold,
            'lif_layer_v_reset': lif_layer_v_reset,
            'lif_layer_sg_width': lif_layer_sg_width,
            'tdBN_on': tdBN_on,
            'BN_on': BN_on,
            'TIME': TIME,
            'surrogate': surrogate,
            'BPTT_on': BPTT_on,
            'DFA_on': DFA_on,
            'OTTT_sWS_on': OTTT_sWS_on,
            'drop_rate': drop_rate,
            'UDA_on': UDA_on,
            'alpha_uda': alpha_uda,
        }

        if (self.params['UDA_on'] == True):
            self.UDA_classifier_on = True
            self.UDA_adapter_on = True
            # cfg == ([200,200],[200],[200],[2]) 이런식으로 들어옴 맨 앞이 feature, 그 다음 게 classifier, 그 다음 게 adapter, 마지막이 도메인 개수 
            self.UDA_feature_layers = make_layers_fc_sstep_UDA_feature(cfg[0], in_c, IMAGE_SIZE, cfg[1][0],
                        synapse_fc_trace_const1, synapse_fc_trace_const2, 
                        lif_layer_v_init, lif_layer_v_decay, 
                        lif_layer_v_threshold, lif_layer_v_reset,
                        lif_layer_sg_width,
                        tdBN_on,
                        BN_on, TIME,
                        surrogate,
                        BPTT_on,
                        DFA_on,
                        OTTT_sWS_on,
                        drop_rate,)
            self.UDA_classifier_layers = make_layers_fc_sstep_UDA_classifier(cfg[1], cfg[0][-1], IMAGE_SIZE, out_c,
                        synapse_fc_trace_const1, synapse_fc_trace_const2, 
                        lif_layer_v_init, lif_layer_v_decay, 
                        lif_layer_v_threshold, lif_layer_v_reset,
                        lif_layer_sg_width,
                        tdBN_on,
                        BN_on, TIME,
                        surrogate,
                        BPTT_on,
                        DFA_on,
                        OTTT_sWS_on,
                        drop_rate,)
            self.UDA_adapter_layers = make_layers_fc_sstep_UDA_adapter(cfg[2], cfg[0][-1], IMAGE_SIZE, cfg[3][0],
                        synapse_fc_trace_const1, synapse_fc_trace_const2, 
                        lif_layer_v_init, lif_layer_v_decay, 
                        lif_layer_v_threshold, lif_layer_v_reset,
                        lif_layer_sg_width,
                        tdBN_on,
                        BN_on, TIME,
                        surrogate,
                        BPTT_on,
                        DFA_on,
                        OTTT_sWS_on,
                        drop_rate,
                        alpha_uda,)
        else:
            self.layers = make_layers_fc_sstep(cfg, in_c, IMAGE_SIZE, out_c,
                        synapse_fc_trace_const1, synapse_fc_trace_const2, 
                        lif_layer_v_init, lif_layer_v_decay, 
                        lif_layer_v_threshold, lif_layer_v_reset,
                        lif_layer_sg_width,
                        tdBN_on,
                        BN_on, TIME,
                        surrogate,
                        BPTT_on,
                        DFA_on,
                        OTTT_sWS_on,
                        drop_rate,)

        self.just_shell = True
    def forward(self, spike_input):
        # inputs: [Batch, Channel, Height, Width]   
        # spike_input = spike_input.permute(1, 0, 2, 3, 4)
        # spike_input = spike_input.view(spike_input.size(0), -1)


        if (self.params['UDA_on'] == True):
            UDA_feature = self.UDA_feature_layers(spike_input)

            if self.UDA_adapter_on == True or self.UDA_adapter_on == False:
                UDA_adapter_out = self.UDA_adapter_layers(UDA_feature)
            else:
                # 일단 둘 다 결과는 다 내보내게 하자. 밖에서 .backward만 안해주면 됨.
                assert False, 'UDA_adapter_on should be True'
            
            if self.UDA_classifier_on == True or self.UDA_classifier_on == False :
                UDA_classifier_out = self.UDA_classifier_layers(UDA_feature)
                return UDA_classifier_out, UDA_adapter_out
            else:
                # 일단 둘 다 classifier는 다 내보내게 하자. 밖에서 .backward만 안해주면 됨.
                pass

        else:
            spike_input = self.layers(spike_input)
            # spike_input = spike_input.mean(axis=0)
            # spike_input = spike_input.sum(axis=0)
            return spike_input
    
def make_layers_fc_sstep(cfg, in_c, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    OTTT_sWS_on,
                    drop_rate,):
    assert BPTT_on == False, 'BPTT_on should be False'
    layers = []
    img_size = IMAGE_SIZE
    in_channels = in_c * img_size * img_size
    class_num = out_c
    pre_pooling_done = False
    for which in cfg:
        if type(which) == list:
            if (pre_pooling_done == False):
                layers += [DimChanger_for_FC_sstep()]
                pre_pooling_done = True
            # residual block 
            layer = ResidualBlock_fc_sstep(which, in_channels, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    OTTT_sWS_on,
                    drop_rate,)
            assert in_channels == layer.in_channels, 'pre-residu, post-residu channel should be same'
            in_channels = layer.in_channels
            layers.append( layer)
        elif which == 'P':
            assert pre_pooling_done == False, 'you must not do pooling after FC'
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            img_size = img_size // 2
            in_channels = in_c * img_size * img_size
        elif which == 'M':
            assert pre_pooling_done == False, 'you must not do pooling after FC'
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            img_size = img_size // 2
            in_channels = in_c * img_size * img_size
        else:
            if (pre_pooling_done == False):
                layers += [DimChanger_for_FC_sstep()]
                pre_pooling_done = True
            out_channels = which
            layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  
                                            out_features=out_channels, 
                                            trace_const1=synapse_fc_trace_const1, 
                                            trace_const2=synapse_fc_trace_const2,
                                            TIME=TIME,
                                            OTTT_sWS_on=OTTT_sWS_on,)]

            in_channels = which

            if (tdBN_on == True):
                assert False, 'impossible in single step mode'
            if (BN_on == True):
                layers += [nn.BatchNorm1d(in_channels)]

            # LIF 뉴런 추가 ##################################
            if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                layers += [LIF_layer_trace_sstep(v_init=lif_layer_v_init, 
                                        v_decay=lif_layer_v_decay, 
                                        v_threshold=lif_layer_v_threshold, 
                                        v_reset=lif_layer_v_reset, 
                                        sg_width=lif_layer_sg_width,
                                        surrogate=surrogate,
                                        BPTT_on=BPTT_on, 
                                        trace_const1=synapse_fc_trace_const1, 
                                        trace_const2=synapse_fc_trace_const2,
                                        TIME=TIME)]
            elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                assert False
            else:
                assert False

            ## OTTT sWS하면 스케일링해줘야됨
            if OTTT_sWS_on == True:
                layers += [Scale(2.74)]

            if drop_rate > 0:
                layers += [Dropout_sstep(drop_rate, TIME)]
                
            if DFA_on == True:
                layers += [Feedback_Receiver(class_num)]
            #################################################

    
    out_channels = class_num
    layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  
                                    out_features=out_channels, 
                                    trace_const1=synapse_fc_trace_const1, 
                                    trace_const2=synapse_fc_trace_const2,
                                    TIME=TIME,
                                    OTTT_sWS_on=False,)]
    
    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=class_num)

        
def make_layers_fc_sstep_UDA_feature(cfg, in_c, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    OTTT_sWS_on,
                    drop_rate,):
    assert BPTT_on == False, 'BPTT_on should be False'
    layers = []
    img_size = IMAGE_SIZE
    in_channels = in_c * img_size * img_size
    class_num = out_c
    pre_pooling_done = False
    for which in cfg:
        if type(which) == list:
            assert False, 'not implemented yet'
        elif which == 'P':
            assert pre_pooling_done == False, 'you must not do pooling after FC'
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            img_size = img_size // 2
            in_channels = in_c * img_size * img_size
        elif which == 'M':
            assert pre_pooling_done == False, 'you must not do pooling after FC'
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            img_size = img_size // 2
            in_channels = in_c * img_size * img_size
        else:
            if (pre_pooling_done == False):
                layers += [DimChanger_for_FC_sstep()]
                pre_pooling_done = True
            out_channels = which
            layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  
                                            out_features=out_channels, 
                                            trace_const1=synapse_fc_trace_const1, 
                                            trace_const2=synapse_fc_trace_const2,
                                            TIME=TIME,
                                            OTTT_sWS_on=OTTT_sWS_on,)]

            in_channels = which

            if (tdBN_on == True):
                assert False, 'impossible in single step mode'
            if (BN_on == True):
                layers += [nn.BatchNorm1d(in_channels)]

            # LIF 뉴런 추가 ##################################
            if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                layers += [LIF_layer_trace_sstep(v_init=lif_layer_v_init, 
                                        v_decay=lif_layer_v_decay, 
                                        v_threshold=lif_layer_v_threshold, 
                                        v_reset=lif_layer_v_reset, 
                                        sg_width=lif_layer_sg_width,
                                        surrogate=surrogate,
                                        BPTT_on=BPTT_on, 
                                        trace_const1=synapse_fc_trace_const1, 
                                        trace_const2=synapse_fc_trace_const2,
                                        TIME=TIME)]
            elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                assert False
            else:
                assert False

            ## OTTT sWS하면 스케일링해줘야됨
            if OTTT_sWS_on == True:
                layers += [Scale(2.74)]

            if drop_rate > 0:
                layers += [Dropout_sstep(drop_rate, TIME)]
                
            if DFA_on == True:
                layers += [Feedback_Receiver(class_num)]
            #################################################
    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=class_num)
        
def make_layers_fc_sstep_UDA_classifier(cfg, in_c, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    OTTT_sWS_on,
                    drop_rate,):
    assert BPTT_on == False, 'BPTT_on should be False'
    layers = []
    # img_size = IMAGE_SIZE
    in_channels = in_c
    class_num = out_c
    for which in cfg:
        if type(which) == list:
            assert False, 'not implemented yet'
        elif which == 'P':
            assert False, 'no pooling in UDA classifier'
        elif which == 'M':
            assert False, 'no pooling in UDA classifier'
        else:
            out_channels = which
            layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  
                                            out_features=out_channels, 
                                            trace_const1=synapse_fc_trace_const1, 
                                            trace_const2=synapse_fc_trace_const2,
                                            TIME=TIME,
                                            OTTT_sWS_on=OTTT_sWS_on,)]

            in_channels = which

            if (tdBN_on == True):
                assert False, 'impossible in single step mode'
            if (BN_on == True):
                layers += [nn.BatchNorm1d(in_channels)]

            # LIF 뉴런 추가 ##################################
            if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                layers += [LIF_layer_trace_sstep(v_init=lif_layer_v_init, 
                                        v_decay=lif_layer_v_decay, 
                                        v_threshold=lif_layer_v_threshold, 
                                        v_reset=lif_layer_v_reset, 
                                        sg_width=lif_layer_sg_width,
                                        surrogate=surrogate,
                                        BPTT_on=BPTT_on, 
                                        trace_const1=synapse_fc_trace_const1, 
                                        trace_const2=synapse_fc_trace_const2,
                                        TIME=TIME)]
            elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                assert False
            else:
                assert False

            ## OTTT sWS하면 스케일링해줘야됨
            if OTTT_sWS_on == True:
                layers += [Scale(2.74)]

            if drop_rate > 0:
                layers += [Dropout_sstep(drop_rate, TIME)]
                
            if DFA_on == True:
                layers += [Feedback_Receiver(class_num)]
            #################################################

    out_channels = class_num
    layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  
                                    out_features=out_channels, 
                                    trace_const1=synapse_fc_trace_const1, 
                                    trace_const2=synapse_fc_trace_const2,
                                    TIME=TIME,
                                    OTTT_sWS_on=False,)]
    
    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=class_num)
        
def make_layers_fc_sstep_UDA_adapter(cfg, in_c, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    OTTT_sWS_on,
                    drop_rate,
                    alpha_uda,):
    assert BPTT_on == False, 'BPTT_on should be False'
    layers = []
    # img_size = IMAGE_SIZE
    in_channels = in_c
    class_num = out_c # maybe 2
    layers += [Gradient_Reversal_Layer(alpha = alpha_uda)]
    for which in cfg:
        if type(which) == list:
            assert False, 'not implemented yet'
        elif which == 'P':
            assert False, 'no pooling in UDA classifier'
        elif which == 'M':
            assert False, 'no pooling in UDA classifier'
        else:
            out_channels = which
            layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  
                                            out_features=out_channels, 
                                            trace_const1=synapse_fc_trace_const1, 
                                            trace_const2=synapse_fc_trace_const2,
                                            TIME=TIME,
                                            OTTT_sWS_on=OTTT_sWS_on,)]

            in_channels = which

            if (tdBN_on == True):
                assert False, 'impossible in single step mode'
            if (BN_on == True):
                layers += [nn.BatchNorm1d(in_channels)]

            # LIF 뉴런 추가 ##################################
            if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
                layers += [LIF_layer_trace_sstep(v_init=lif_layer_v_init, 
                                        v_decay=lif_layer_v_decay, 
                                        v_threshold=lif_layer_v_threshold, 
                                        v_reset=lif_layer_v_reset, 
                                        sg_width=lif_layer_sg_width,
                                        surrogate=surrogate,
                                        BPTT_on=BPTT_on, 
                                        trace_const1=synapse_fc_trace_const1, 
                                        trace_const2=synapse_fc_trace_const2,
                                        TIME=TIME)]
            elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
                assert False
            else:
                assert False

            ## OTTT sWS하면 스케일링해줘야됨
            if OTTT_sWS_on == True:
                layers += [Scale(2.74)]

            if drop_rate > 0:
                layers += [Dropout_sstep(drop_rate, TIME)]
                
            if DFA_on == True:
                layers += [Feedback_Receiver(class_num)]
            #################################################

    out_channels = class_num
    layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  
                                    out_features=out_channels, 
                                    trace_const1=synapse_fc_trace_const1, 
                                    trace_const2=synapse_fc_trace_const2,
                                    TIME=TIME,
                                    OTTT_sWS_on=False,)]
    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=class_num)

class ResidualBlock_fc_sstep(nn.Module):
    def __init__(self, layers, in_channels, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    OTTT_sWS_on,
                    drop_rate,):
        super(ResidualBlock_fc_sstep, self).__init__()
        assert DFA_on == False, 'not implemented yet DFA & residual block'
        self.layers, self.in_channels = make_layers_fc_residual_sstep(layers, in_channels, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    OTTT_sWS_on,
                    drop_rate,)
    
        self.just_shell = True
    def forward(self, x):
        x = self.layers(x)
        return x
    
    


def make_layers_fc_residual_sstep(cfg, in_c, IMAGE_SIZE, out_c,
                    synapse_fc_trace_const1, synapse_fc_trace_const2, 
                    lif_layer_v_init, lif_layer_v_decay, 
                    lif_layer_v_threshold, lif_layer_v_reset,
                    lif_layer_sg_width,
                    tdBN_on,
                    BN_on, TIME,
                    surrogate,
                    BPTT_on,
                    DFA_on,
                    OTTT_sWS_on,
                    drop_rate,):
    assert DFA_on == False, 'not implemented yet DFA & residual block'
    layers = []
    img_size = IMAGE_SIZE
    in_channels = in_c
    class_num = out_c
    for which in cfg:

        out_channels = which

        layers += [SYNAPSE_FC_trace_sstep(in_features=in_channels,  
                                        out_features=out_channels, 
                                        trace_const1=synapse_fc_trace_const1, 
                                        trace_const2=synapse_fc_trace_const2,
                                        TIME=TIME,
                                        OTTT_sWS_on=OTTT_sWS_on,)]

        in_channels = which
        
        if (tdBN_on == True):
            assert False, 'impossible in single step mode'

        if (BN_on == True):
            layers += [nn.BatchNorm1d(in_channels)]


        # LIF 뉴런 추가 ##################################
        if (lif_layer_v_threshold >= 0 and lif_layer_v_threshold < 10000):
            layers += [LIF_layer_trace_sstep(v_init=lif_layer_v_init, 
                                    v_decay=lif_layer_v_decay, 
                                    v_threshold=lif_layer_v_threshold, 
                                    v_reset=lif_layer_v_reset, 
                                    sg_width=lif_layer_sg_width,
                                    surrogate=surrogate,
                                    BPTT_on=BPTT_on, 
                                    trace_const1=synapse_fc_trace_const1, 
                                    trace_const2=synapse_fc_trace_const2,
                                    TIME=TIME)]
        elif (lif_layer_v_threshold >= 10000 and lif_layer_v_threshold < 20000):
            assert False
        else:
            assert False

        ## OTTT sWS하면 스케일링해줘야됨
        if OTTT_sWS_on == True:
            layers += [Scale(2.74)]

        if drop_rate > 0:
            layers += [Dropout_sstep(drop_rate, TIME)]
        #################################################
            
    return MY_Sequential(*layers, BPTT_on=BPTT_on, DFA_on=DFA_on, class_num=class_num), in_channels
####### make_layers for ottt fc single step ############################################
####### make_layers for ottt fc single step ############################################
####### make_layers for ottt fc single step ############################################






##### Several functions for model implementation #########################
##### Several functions for model implementation #########################
##### Several functions for model implementation #########################
class DimChanger_for_pooling(nn.Module):
    def __init__(self, module):
        super(DimChanger_for_pooling, self).__init__()
        self.ann_module = module

    def forward(self, x):
        timestep, batch_size, *dim = x.shape
        output = self.ann_module(x.reshape(timestep * batch_size, *dim))
        _, *dim = output.shape
        output = output.view(timestep, batch_size, *dim).contiguous()
        return output


class DimChanger_for_FC(nn.Module):
    def __init__(self):
        super(DimChanger_for_FC, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        return x
class DimChanger_for_FC_sstep(nn.Module):
    def __init__(self):
        super(DimChanger_for_FC_sstep, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
    
    
class DimChanger_for_change_0_1(nn.Module):
    def __init__(self):
        super(DimChanger_for_change_0_1, self).__init__()

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
        return x

class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, channel):
        super(tdBatchNorm, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        # self.weight.data.mul_(0.5)

    def forward(self, x):
        T, B, *spatial_dims = x.shape
        out = super().forward(x.reshape(T * B, *spatial_dims))
        TB, *spatial_dims = out.shape
        out = out.view(T, B, *spatial_dims).contiguous()
        return out
    
    
class tdBatchNorm_FC(nn.BatchNorm1d):
    def __init__(self, channel):
        super(tdBatchNorm_FC, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        # self.weight.data.mul_(0.5)

    def forward(self, x):
        T, B, *spatial_dims = x.shape
        out = super().forward(x.reshape(T * B, *spatial_dims))
        TB, *spatial_dims = out.shape
        out = out.view(T, B, *spatial_dims).contiguous()
        return out


class BatchNorm(nn.Module):
    def __init__(self, out_channels, TIME):
        super(BatchNorm, self).__init__()
        self.out_channels = out_channels
        self.TIME = TIME
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(self.out_channels) for _ in range(self.TIME)])

    def forward(self, x):
        # out = torch.zeros_like(x, device=x.device) #Time, Batch, Channel, Height, Width
        out = [] #Time, Batch, Channel, Height, Width
        for t in range(self.TIME):
            out.append(self.bn_layers[t](x[t]))
        out = torch.stack(out, dim=0)
        return out
    
class BatchNorm_FC(nn.Module):
    def __init__(self, out_channels, TIME):
        super(BatchNorm_FC, self).__init__()
        self.out_channels = out_channels
        self.TIME = TIME
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(self.out_channels) for _ in range(self.TIME)])

    def forward(self, x):
        # out = torch.zeros_like(x, device=x.device) #Time, Batch, Channel, Height, Width
        out = [] #Time, Batch, Channel, Height, Width
        for t in range(self.TIME):
            out.append(self.bn_layers[t](x[t]))
        out = torch.stack(out, dim=0)
        return out
##### Several functions for model implementation #########################
##### Several functions for model implementation #########################
##### Several functions for model implementation #########################



######### OTTT & spikingjelly functions #################################################################################
######### OTTT & spikingjelly functions #################################################################################
######### OTTT & spikingjelly functions #################################################################################
class Scale(nn.Module):
    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale
    
class MY_Sequential(nn.Sequential):
    def __init__(self, *args, BPTT_on, DFA_on, class_num):
        super().__init__(*args)
        self.BPTT_on = BPTT_on
        self.DFA_on = DFA_on
        self.class_num = class_num
        if (self.DFA_on == True):
            self.DFA_top = Top_Gradient()

        self.just_shell = True
    def forward(self, input):
        if self.DFA_on == True:
            dummies = []

        for module in self:
            if self.BPTT_on == True:
                if isinstance(module, Feedback_Receiver):
                    output, dummy = module(input)
                    dummies.append(dummy)
                else:
                    output = module(input)
                if isinstance(module, ResidualBlock_fc_sstep) or isinstance(module, ResidualBlock_conv_sstep) or isinstance(module, ResidualBlock_fc) or isinstance(module, ResidualBlock_conv): # e.g., ResidualBlock_fc_sstep
                    output = output + input

            elif self.BPTT_on == False: #ottt with trace
                
                if not isinstance(input, list): # input: only spike
                    output = module(input)
                    if isinstance(module, ResidualBlock_fc_sstep) or isinstance(module, ResidualBlock_conv_sstep) or isinstance(module, ResidualBlock_fc) or isinstance(module, ResidualBlock_conv): # e.g., ResidualBlock_fc_sstep
                        assert isinstance(input, list) == isinstance(output, list), 'residual input and output should have same type'
                        output = output + input
                else: # input: [spike, trace]
                    residual = False
                    if isinstance(module, SYNAPSE_CONV_trace) or isinstance(module, SYNAPSE_FC_trace) or isinstance(module, SYNAPSE_CONV_trace_sstep) or isinstance(module, SYNAPSE_FC_trace_sstep): # e.g., Conv2d, Linear, etc.
                    # if len(list(module.parameters())) > 0: # e.g., Conv2d, Linear, etc.
                        module = GradwithTrace(module)
                    elif isinstance(module, ResidualBlock_fc_sstep) or isinstance(module, ResidualBlock_conv_sstep) or isinstance(module, ResidualBlock_fc) or isinstance(module, ResidualBlock_conv): # e.g., ResidualBlock_fc_sstep
                        residual = True
                    elif isinstance(module, Feedback_Receiver):
                        assert self.DFA_on == True, 'Feedback_Receiver should be used only when DFA_on is True'
                    else: # e.g., Dropout, AvgPool, etc.
                        module = SpikeTraceOp(module)

                    if isinstance(module, Feedback_Receiver):
                        output, dummy = module(input)
                        dummies.append(dummy)
                    else:
                        output = module(input)

                    if residual == True: # e.g., ResidualBlock_fc_sstep
                        assert isinstance(input, list) == isinstance(output, list) and len(output) == len(input) and len(input) == 2, 'residual input and output should have same type'
                        output = [a + b for a, b in zip(output, input)] #output = output + input
            input = output

        if self.DFA_on == True:
            if isinstance(input, list) == True:
                spike, trace = input[0], input[1]
                output = self.DFA_top(spike, *dummies)
                output = [output, trace]
            else:
                output = self.DFA_top(input, *dummies)
        return output

class SpikeTraceOp(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]
            
        if isinstance(self.module, Dropout_sstep):
            spike = self.module(spike, trace_input = False)
        else:
            spike = self.module(spike)
        

        with torch.no_grad():
            if isinstance(self.module, Dropout_sstep):
                trace = self.module(trace, trace_input = True)
            else:
                trace = self.module(trace)
        x = [spike, trace]

        return x
    
class GradwithTrace(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]
        
        with torch.no_grad():
            out = self.module(spike).detach()

        in_for_grad = ReplaceforGrad.apply(spike, trace)
        out_for_grad = self.module(in_for_grad)

        x = ReplaceforGrad.apply(out_for_grad, out)

        return x

class ReplaceforGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_r):
        return x_r

    @staticmethod
    def backward(ctx, grad):
        return (grad, grad)
######### OTTT & spikingjelly functions #################################################################################
######### OTTT & spikingjelly functions #################################################################################
######### OTTT & spikingjelly functions #################################################################################




######### ASAP DFA CODE ################################################################################################
######### ASAP DFA CODE ################################################################################################
######### ASAP DFA CODE ################################################################################################
class feedback_receiver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_fb):
        output = input.clone()
        if (len(input.size()) == 3 or len(input.size()) == 5): # multi time processing
            dummy = torch.Tensor(input.size()[0],input.size()[1],weight_fb.size()[0]).zero_().to(input.device)
        else:
            dummy = torch.Tensor(input.size()[0],weight_fb.size()[0]).zero_().to(input.device)
        ctx.save_for_backward(weight_fb,)
        ctx.shape = input.shape
        return output, dummy
    
    @staticmethod
    def backward(ctx, grad_output, grad_dummy):
        weight_fb, = ctx.saved_tensors
        input_size = ctx.shape
        if (len(input_size) == 3 or len(input_size) == 5): # multi time processing
            grad_input = torch.mm(grad_dummy.view(grad_dummy.size()[0] * grad_dummy.size()[1],-1), weight_fb).view(input_size)
        else:
            grad_input = torch.mm(grad_dummy.view(grad_dummy.size()[0],-1), weight_fb).view(input_size)
        
        grad_weight_fb = None
        return grad_input, grad_weight_fb


class Feedback_Receiver(nn.Module):
    def __init__(self, connect_features):
        super(Feedback_Receiver, self).__init__()
        self.connect_features = connect_features
        self.weight_fb = None
    
    def forward(self, input):
        if isinstance(input, list) == True:
            spike, trace = input[0], input[1]
        else:
            spike, trace = input, input

        if self.weight_fb is None:
            if (len(spike.size()) == 3 or len(spike.size()) == 5): # multi time processing
                self.weight_fb = nn.Parameter(torch.Tensor(self.connect_features, *spike.size()[2:]).view(self.connect_features, -1)).to(spike.device)
            else:
                self.weight_fb = nn.Parameter(torch.Tensor(self.connect_features, *spike.size()[1:]).view(self.connect_features, -1)).to(spike.device)
            nn.init.normal_(self.weight_fb, std = math.sqrt(1./self.connect_features))
        if isinstance(input, list) == True:
            output, dummy = feedback_receiver.apply(spike, self.weight_fb)
            output = [output, trace]
        else:
            output, dummy = feedback_receiver.apply(spike, self.weight_fb)
        return output, dummy

class top_gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, *dummies):
        output = input.clone()
        ctx.save_for_backward(output ,*dummies)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, *dummies = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_dummies = [grad_output.clone() for dummy in dummies]
        return tuple([grad_input, *grad_dummies])

class Top_Gradient(nn.Module):
    def __init__(self):
        super(Top_Gradient, self).__init__()
    
    def forward(self, input, *dummies):
        return top_gradient.apply(input, *dummies)
######### ASAP DFA CODE ################################################################################################
######### ASAP DFA CODE ################################################################################################
######### ASAP DFA CODE ################################################################################################
    




######### BP_DFA_SWAP ################################################################################################
######### BP_DFA_SWAP ################################################################################################
######### BP_DFA_SWAP ###############################################################################################
def BP_DFA_SWAP(net, convTrue_fcFalse, single_step, ddp_on, args_gpu):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(net, nn.DataParallel) == True:
        net_old = net.module
    else:
        net_old = net

    if net_old.params['DFA_on'] == True:
        print('\n==================== e-transport DFA --> BP ===============================================\n==================== e-transport DFA --> BP ===============================================\n\n')
    else:
        print('\n==================== e-transport BP --> DFA ===============================================\n==================== e-transport BP --> DFA ===============================================\n\n')

    net_old.params['DFA_on'] = not net_old.params['DFA_on'] # conversion

    if (convTrue_fcFalse == False):
        if (single_step == False):
            net_new = MY_SNN_FC(**(net_old.params)).to(device)
        else:
            net_new = MY_SNN_FC_sstep(**(net_old.params)).to(device)
    else:
        if (single_step == False):
            net_new = MY_SNN_CONV(**(net_old.params)).to(device)
        else:
            net_new = MY_SNN_CONV_sstep(**(net_old.params)).to(device)

    net_old_layer_list = list(net_old.named_modules())
    old_index = 0
    net_new_layer_list = list(net_new.named_modules())
    new_index = 0
    while (True):
        while (True):
            old_name, old_current_layer = net_old_layer_list[old_index]
            old_index = old_index + 1
            if any(p.requires_grad for p in old_current_layer.parameters()) and not hasattr(old_current_layer, 'just_shell'):
                break
        while (True):
            new_name, new_current_layer = net_new_layer_list[new_index]
            new_index = new_index + 1
            if any(p.requires_grad for p in new_current_layer.parameters()) and not hasattr(new_current_layer, 'just_shell'):
                break

        assert type(old_current_layer) == type(new_current_layer), 'layer type should be same'

        new_current_layer.weight = old_current_layer.weight
        if hasattr(new_current_layer, 'bias'):
            new_current_layer.bias = old_current_layer.bias

        if old_index >= len(net_old_layer_list) or new_index >= len(net_new_layer_list):
            break
            
    if isinstance(net, nn.DataParallel) == True:
        assert ddp_on == False
        net_new = torch.nn.DataParallel(net_new) 
    
    if isinstance(net, nn.DataParallel) == False:
        assert ddp_on == True
        device = args_gpu
        net_new = net_new.to(args_gpu)
        net_new = DDP(net_new, delay_allreduce=True)

    net_new = net_new.to(device)
    # if isinstance(net, nn.DataParallel) == True or torch.distributed.get_rank() == 0:
    #     print(net_new)  

    return net_new
######### BP_DFA_SWAP ################################################################################################
######### BP_DFA_SWAP ################################################################################################
######### BP_DFA_SWAP ################################################################################################
    

######### Dropout ################################################################################################
######### Dropout ################################################################################################
######### Dropout ################################################################################################
class Dropout_sstep(nn.Module):
    def __init__(self, p, TIME):
        super().__init__()
        assert 0.0 < p < 1.0
        self.p = p
        self.TIME = TIME
        self.time_count = 0

    def forward(self, x, trace_input = False):
        if self.training == True:
            if trace_input == False:
                self.time_count = self.time_count + 1
                if self.time_count == 1:
                    self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)
                    # self.mask = F.dropout2d(torch.ones_like(x.data), self.p, training=True)
            x = x * self.mask
            if trace_input == False:
                if (self.time_count == self.TIME):
                    self.time_count = 0
        return x
######### Dropout ################################################################################################
######### Dropout ################################################################################################
######### Dropout ################################################################################################
    


######### UDA GRL ################################################################################################
######### UDA GRL ################################################################################################
######### UDA GRL ################################################################################################
class Gradient_Reversal_Layer(nn.Module):
    def __init__(self, alpha = 1.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return GRL_METHOD.apply(x, self.alpha)
    
class GRL_METHOD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        x, alpha = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
            
        return grad_input, None
######### UDA GRL ################################################################################################
######### UDA GRL ################################################################################################
######### UDA GRL ################################################################################################
    
