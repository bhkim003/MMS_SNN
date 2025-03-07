import sys
import os
from cv2 import NONE_POLISHER
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


from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *
from modules.old_fashioned import *
from modules.ae_network import *

######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
# class LIF_layer(nn.Module):
#     def __init__ (self, v_init , v_decay , v_threshold , v_reset , sg_width, surrogate, BPTT_on):
#         super(LIF_layer, self).__init__()
#         self.v_init = v_init
#         self.v_decay = v_decay
#         self.v_threshold = v_threshold
#         self.v_reset = v_reset
#         self.sg_width = sg_width
#         self.surrogate = surrogate
#         self.BPTT_on = BPTT_on

#     def forward(self, input_current):
#         v = torch.full_like(input_current, fill_value = self.v_init, dtype = torch.float, requires_grad=False) # v (membrane potential) init
#         v_start = torch.full_like(input_current[0], fill_value = self.v_init, dtype = torch.float, requires_grad=False)
#         post_spike = torch.full_like(input_current, fill_value = self.v_init, device=input_current.device, dtype = torch.float, requires_grad=False) 
#         # i와 v와 post_spike size는 여기서 다 같음: [Time, Batch, Channel, Height, Width] 
#         Time = v.shape[0]
#         for t in range(Time):
#             # leaky하고 input_current 더하고 fire하고 reset까지 (backward직접처리)

#             # post_spike[t], v[t] = LIF_METHOD.apply(input_current[t], v[t], 
#             #                             self.v_decay, self.v_threshold, self.v_reset, self.sg_width, self.surrogate, self.BPTT_on) 

#             if t == 0:
#                 post_spike[t], v[t] = LIF_METHOD.apply(input_current[t], v_start, 
#                                             self.v_decay, self.v_threshold, self.v_reset, self.sg_width, self.surrogate, self.BPTT_on) 
#             else:
#                 post_spike[t], v[t] = LIF_METHOD.apply(input_current[t], v[t-1], 
#                                             self.v_decay, self.v_threshold, self.v_reset, self.sg_width, self.surrogate, self.BPTT_on) 
            
            
            
#             # if t != 0:
#             #     print(torch.equal(v[t], v[t-1]))
#             #     if torch.equal(v[t], v[t-1]):
#             #         print('v',v[t])
#             #     print(torch.equal(post_spike[t], post_spike[t-1]))
#             #     if torch.equal(post_spike[t], post_spike[t-1]):
#             #         print('post_spike',post_spike[t])
        
#         return post_spike 



# class LIF_METHOD(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_current_one_time, v_one_time, v_decay, v_threshold, v_reset, sg_width, surrogate, BPTT_on):
#         v_one_time = v_one_time * v_decay + input_current_one_time # leak + pre-synaptic current integrate
#         spike = (v_one_time >= v_threshold).float() #fire
#         if surrogate == 'sigmoid':
#             surrogate = 1
#         elif surrogate == 'rectangle':
#             surrogate = 2
#         elif surrogate == 'rough_rectangle':
#             surrogate = 3
#         elif surrogate == 'hard_sigmoid':
#             surrogate = 4
#         else:
#             pass

#         if (BPTT_on == True):
#             BPTT_on = 1
#         else:
#             BPTT_on = 0

#         ctx.save_for_backward(v_one_time, torch.tensor([v_decay], requires_grad=False), 
#                             torch.tensor([v_threshold], requires_grad=False), 
#                             torch.tensor([v_reset], requires_grad=False), 
#                             torch.tensor([sg_width], requires_grad=False),
#                             torch.tensor([surrogate], requires_grad=False),
#                             torch.tensor([BPTT_on], requires_grad=False)) # save before reset
        
#         if (v_reset >= 0 and v_reset < 10000): # soft reset
#             v_one_time = (v_one_time - spike * v_threshold) # reset
#             # v_one_time = (v_one_time - spike * v_threshold).clamp_min(0) # reset # 0미만으로는 안 가게 하려면 이 줄 on
#         elif (v_reset >= 10000 and v_reset < 20000): # hard reset 
#             v_reset -= 10000 
#             v_one_time = v_one_time * (1 - spike) + v_reset * spike # reset
#             v_reset += 10000
#         return spike, v_one_time

#     @staticmethod
#     def backward(ctx, grad_output_spike, grad_output_v):
#         v_one_time, v_decay, v_threshold, v_reset, sg_width, surrogate, BPTT_on = ctx.saved_tensors
#         v_decay=v_decay.item()
#         v_threshold=v_threshold.item()
#         v_reset=v_reset.item()
#         sg_width=sg_width.item()
#         surrogate=surrogate.item()
#         BPTT_on=BPTT_on.item()

#         grad_input_current = grad_output_spike.clone()
#         if BPTT_on == 1:
#             grad_input_v = grad_output_v.clone() 
#             # print('grad_output_v',grad_output_v)
#         ################ select one of the following surrogate gradient functions ################
#         if (surrogate == 1):
#             #===========surrogate gradient function (sigmoid)
#             alpha = sg_width 
#             sig = torch.sigmoid(alpha*(v_one_time - v_threshold))
#             grad_input_current *= alpha*sig*(1-sig)
#             # grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha

#         elif (surrogate == 2):
#             # ===========surrogate gradient function (rectangle)
#             grad_input_current *= ((v_one_time - v_threshold).abs() < sg_width/2).float() / sg_width

#         elif (surrogate == 3):
#             #===========surrogate gradient function (rough rectangle)
#             grad_input_current[(v_one_time - v_threshold).abs() > sg_width/2] = 0
#             grad_input_current = grad_input_current / sg_width
#         elif (surrogate == 4):
#             #===========surrogate gradient function (hard sigmoid)
#             alpha = sg_width 
#             sig = torch.clamp(alpha*(v_one_time - v_threshold) * 0.2 + 0.5, min=0, max=1)
#             grad_input_current = alpha*sig*(1-sig)*grad_input_current
#         else: 
#             assert False, 'surrogate doesn\'t exist'
#         ###########################################################################################

#         ## if BPTT_on == 1, then second return value is not None
#         if (BPTT_on == 1):
#             grad_input_v = v_decay * grad_input_v 
#         else:
#             grad_input_v = None 
#         return grad_input_current, grad_input_v, None, None, None, None, None, None
######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
######## LIF Neuron #####################################################
    
    
# deprecated!!!!!!!!
######## LIF Neuron trace #####################################################
######## LIF Neuron trace #####################################################
######## LIF Neuron trace #####################################################
# deprecated!!!!!!!!
# class LIF_layer_trace(nn.Module):
#     def __init__ (self, v_init , v_decay , v_threshold , v_reset , sg_width, surrogate, BPTT_on, trace_const1=1, trace_const2=0.7):
#         super(LIF_layer_trace, self).__init__()
#         self.v_init = v_init
#         self.v_decay = v_decay
#         self.v_threshold = v_threshold
#         self.v_reset = v_reset
#         self.sg_width = sg_width
#         self.surrogate = surrogate
#         self.BPTT_on = BPTT_on
#         self.trace_const1 = trace_const1
#         self.trace_const2 = trace_const2

#     def forward(self, input_current):
#         v = torch.full_like(input_current, fill_value = self.v_init, dtype = torch.float, requires_grad=False) # v (membrane potential) init
#         post_spike = torch.full_like(input_current, fill_value = self.v_init, device=input_current.device, dtype = torch.float, requires_grad=False) 
#         # i와 v와 post_spike size는 여기서 다 같음: [Time, Batch, Channel, Height, Width] 
#         Time = v.shape[0]
#         trace = torch.full_like(input_current, fill_value = self.v_init, device=input_current.device, dtype = torch.float, requires_grad=False) 
#         trace_past = torch.full_like(input_current[0], fill_value = self.v_init, device=input_current.device, dtype = torch.float, requires_grad=False) 
#         for t in range(Time):
#             # leaky하고 input_current 더하고 fire하고 reset까지 (backward직접처리)
#             post_spike[t], v[t] = LIF_METHOD.apply(input_current[t], v[t-1], 
#                                             self.v_decay, self.v_threshold, self.v_reset, self.sg_width, self.surrogate, self.BPTT_on) 
#             trace[t] = self.trace_const1*((post_spike[t]).detach()) + self.trace_const2*trace_past
#             trace_past = trace[t]
#         return [post_spike, trace] 
######## LIF Neuron trace #####################################################
######## LIF Neuron trace #####################################################
######## LIF Neuron trace #####################################################



    
class LIF_layer(nn.Module):
    def __init__ (self, v_init , v_decay , v_threshold , v_reset , sg_width, surrogate, BPTT_on):
        super(LIF_layer, self).__init__()
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on

    def forward(self, input_current):
        Time = input_current.shape[0]
        v = torch.full_like(input_current[0], fill_value = self.v_init, dtype = torch.float, requires_grad=False)
        post_spike = torch.full_like(input_current, fill_value = 0.0, device=input_current.device, dtype = torch.float, requires_grad=False) 
        
        for t in range(Time):
            # print('input_current', input_current.shape, input_current[t][0])
            # print('membrane potential', v.shape, v[0])
            if (self.BPTT_on == True):
                v = v * self.v_decay + input_current[t]
                # v = V_DECAY.apply(v, self.v_decay, self.BPTT_on) + input_current[t]
            else:
                v = v.detach() * self.v_decay + input_current[t]
                # v = V_DECAY.apply(v, self.v_decay, self.BPTT_on) + input_current[t]

            post_spike[t] = FIRE.apply(v - self.v_threshold, self.surrogate, self.sg_width) 
            # print(f"v 평균값: {v.mean()}, v 분산: {v.var()}, post_spike 평균값: {post_spike[t].mean()}, post_spike 분산: {post_spike[t].var()}")

            if (self.v_reset >= 0 and self.v_reset < 10000): # soft reset
                v = v - post_spike[t].detach() * self.v_threshold
            elif (self.v_reset >= 10000 and self.v_reset < 20000): # hard reset 
                v = v*(1-post_spike[t].detach()) + (self.v_reset - 10000)*post_spike[t].detach()

            # print(f"v 평균값: {v.mean()}, v 분산: {v.var()}, post_spike 평균값: {post_spike[t].mean()}, post_spike 분산: {post_spike[t].var()}")
        # print('input_current', input_current.shape)
        # print('post_spike', post_spike.shape)
        # print('sparsity', (post_spike == 0.0).sum().item() / post_spike.numel())
        return post_spike
    

class V_DECAY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_decay, BPTT_on):
        ctx.save_for_backward(torch.tensor([v_decay], requires_grad=False),
                              torch.tensor([BPTT_on], requires_grad=False)) # save before reset
        return v*v_decay

    @staticmethod
    def backward(ctx, grad_output):
        v_decay, BPTT_on = ctx.saved_tensors
        v_decay=v_decay.item()
        BPTT_on=BPTT_on.item()
        
        v_decay = v_decay if BPTT_on else 0.0
        grad_input = grad_output * v_decay

        return grad_input, None, None
    

class FIRE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_threshold, surrogate, sg_width):
        if surrogate == 'sigmoid':
            surrogate = 1
        elif surrogate == 'rectangle':
            surrogate = 2
        elif surrogate == 'rough_rectangle':
            surrogate = 3
        elif surrogate == 'hard_sigmoid':
            surrogate = 4
        else:
            assert False, 'surrogate doesn\'t exist'
        ctx.save_for_backward(v_minus_threshold,
                            torch.tensor([surrogate], requires_grad=False),
                            torch.tensor([sg_width], requires_grad=False)) # save before reset
        return (v_minus_threshold >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        v_minus_threshold, surrogate, sg_width = ctx.saved_tensors
        # v_minus_threshold=v_minus_threshold.item() #ValueError: only one element tensors can be converted to Python scalars
        surrogate=surrogate.item()
        sg_width=sg_width.item()

        if (surrogate == 1):
            #===========surrogate gradient function (sigmoid)
            alpha = sg_width 
            sig = torch.sigmoid(alpha*v_minus_threshold)
            grad_input = alpha*sig*(1-sig)*grad_output
        elif (surrogate == 2):
            # ===========surrogate gradient function (rectangle)
            grad_input = grad_output * (v_minus_threshold.abs() < sg_width/2).float() / sg_width
        elif (surrogate == 3):
            #===========surrogate gradient function (rough rectangle)
            grad_input[v_minus_threshold.abs() > sg_width/2] = 0
            grad_input = grad_output / sg_width
        elif (surrogate == 4):
            #===========surrogate gradient function (hard sigmoid)
            alpha = sg_width 
            sig = torch.clamp(alpha*v_minus_threshold * 0.2 + 0.5, min=0, max=1)
            grad_input = alpha*sig*(1-sig)*grad_output
        return grad_input, None, None
    


######## LIF Neuron trace single step #####################################################
######## LIF Neuron trace single step #####################################################
######## LIF Neuron trace single step #####################################################
class LIF_layer_trace_sstep(nn.Module):
    def __init__ (self, v_init , v_decay , v_threshold , v_reset , sg_width, surrogate, BPTT_on, trace_const1=1, trace_const2=0.7, TIME=6):
        super(LIF_layer_trace_sstep, self).__init__()
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.trace_const1 = trace_const1
        self.trace_const2 = trace_const2

        self.TIME = TIME
        self.time_count = 0

    def forward(self, input_current):
        self.time_count = self.time_count + 1

        if self.time_count == 1:
            self.trace = torch.full_like(input_current, fill_value = 0.0, dtype = torch.float, requires_grad=False) # v (membrane potential) init
            self.v = torch.full_like(input_current, fzll_value = self.v_init, dtype = torch.float, requires_grad=False) # v (membrane potential) init

        self.v = self.v.detach() * self.v_decay + input_current 
        post_spike = FIRE.apply(self.v - self.v_threshold, self.surrogate, self.sg_width) 
        if (self.v_reset >= 0 and self.v_reset < 10000): # soft reset
            self.v = self.v - post_spike.detach() * self.v_threshold
        elif (self.v_reset >= 10000 and self.v_reset < 20000): # hard reset 
            self.v = self.v*(1-post_spike.detach()) + (self.v_reset - 10000)*post_spike.detach()
        out_trace = self.trace*self.trace_const2 + post_spike*self.trace_const1

        if (self.time_count == self.TIME):
            self.v = self.v.detach()
            self.trace = self.trace.detach()
            self.time_count = 0
        
        return [post_spike, out_trace] 
######## LIF Neuron trace single step #####################################################
######## LIF Neuron trace single step #####################################################
######## LIF Neuron trace single step #####################################################
    







######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################

class QCFS_GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
class QCFS_IF(nn.Module):
    def __init__(self,  L=8, thresh=8.0):
        super(QCFS_IF, self).__init__()
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.L = L
        self.shift = 0.5
        print('thresh', self.thresh.item(), 'L', self.L)

    def forward(self, x):
        x = x / self.thresh
        x = torch.clamp(x, 0, 1) # 이게 한 줄 밑에 가야되는 거 아님? 논문상으론 그런데??? 나중에 실험해보자
        x = QCFS_GradFloor.apply(x*self.L+self.shift)/self.L
        x = x * self.thresh
        return x
######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################
######## QCFS Neuron ######################################################################
