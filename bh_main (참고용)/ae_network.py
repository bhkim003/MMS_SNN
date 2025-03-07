
from sklearn.decomposition import TruncatedSVD
from sympy import Rel
import torch
import torch.nn as nn

from modules.data_loader import *
from modules.network import *
from modules.neuron import *
from modules.synapse import *
from modules.old_fashioned import *
from modules.ae_network import *


import modules.neuron as neuron # 밑에서neuron.LIF_layer 이렇게 import해라.

class SSBH_DimChanger_for_fc(nn.Module):
    def __init__(self):
        super(SSBH_DimChanger_for_fc, self).__init__()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return x
class SSBH_DimChanger_for_conv1(nn.Module):
    def __init__(self, out_channels):
        super(SSBH_DimChanger_for_conv1, self).__init__()
        self.out_channels = out_channels
    def forward(self, x):
        x = x.reshape(x.size(0), self.out_channels, -1)
        return x
class SSBH_DimChanger_for_unsuqeeze(nn.Module):
    def __init__(self, dim=1):
        super(SSBH_DimChanger_for_unsuqeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = x.unsqueeze(self.dim)
        return x
class SSBH_DimChanger_for_suqeeze(nn.Module):
    def __init__(self, dim=1):
        super(SSBH_DimChanger_for_suqeeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        x = x.squeeze(self.dim)
        return x
class SSBH_size_detector(nn.Module):
    def __init__(self):
        super(SSBH_size_detector, self).__init__()

    def forward(self, x):
        print(x.size())
        # if len(x.shape) == 4:
        #     print(x[0][0])
        # else:
        #     print(x[0])
        return x
class SSBH_L2NormLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(SSBH_L2NormLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        # return F.normalize(x, p=2, dim=self.dim, eps=self.eps)
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        x = x / (norm + self.eps)
        return x
    
class SSBH_DimChanger_one_two(nn.Module):
    def __init__(self):
        super(SSBH_DimChanger_one_two, self).__init__()

    def forward(self, x):
        dims = list(range(x.dim()))  # 전체 차원의 인덱스 리스트
        dims[0], dims[1] = dims[1], dims[0]  # 첫 번째와 두 번째 차원 교체
        return x.permute(*dims)
    
class SSBH_DimChanger_for_one_two_coupling(nn.Module):
    def __init__(self, TIME):
        super(SSBH_DimChanger_for_one_two_coupling, self).__init__()
        self.TIME = TIME

    def forward(self, x):
        T, B, *spatial_dims = x.shape
        # assert T == self.TIME, f'x.shape {x.shape},  self.TIME {self.TIME}'
        x = x.reshape(T * B, *spatial_dims)
        return x   
class SSBH_DimChanger_for_one_two_decoupling(nn.Module): #항상TIME이 앞에 오도록 decoupling
    def __init__(self, TIME):
        super(SSBH_DimChanger_for_one_two_decoupling, self).__init__()
        self.TIME = TIME

    def forward(self, x):
        TB, *spatial_dims = x.shape
        T = self.TIME 
        B = TB // T
        assert TB == T*B
        x = x.reshape(T, B, *spatial_dims)
        return x
class SSBH_activation_watcher(nn.Module):
    def __init__(self):
        super(SSBH_activation_watcher, self).__init__()

    def forward(self, x):
        print(x.size())
        print(x)
        return x
class SSBH_mean(nn.Module):
    def __init__(self, dim=0):
        super(SSBH_mean, self).__init__()
        self.dim = dim
    def forward(self, x):
        x = x.mean(dim=self.dim)
        return x
class SSBH_repeat(nn.Module):
    def __init__(self, TIME):
        super(SSBH_repeat, self).__init__()
        self.TIME = TIME
    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.TIME, 1, 1) # (batch, time, feature)로 변환
        return x
class SSBH_activation_collector(nn.Module):
    def __init__(self):
        super(SSBH_activation_collector, self).__init__()
        self.activation = []
    def forward(self, x):
        self.activation += [x]
        return x
class SSBH_mul_vth(nn.Module):
    def __init__(self, vth):
        super(SSBH_mul_vth, self).__init__()
        self.vth = vth
    def forward(self, x):
        x = x*self.vth
        return x
class SSBH_rate_coding(nn.Module):
    def __init__(self, TIME):
        super(SSBH_rate_coding, self).__init__()
        self.TIME = TIME
    def forward(self, x):
        x = spikegen.rate(x, num_steps=self.TIME).transpose(0, 1)
        return x
class SSBH_repeat_coding(nn.Module):
    def __init__(self, TIME):
        super(SSBH_repeat_coding, self).__init__()
        self.TIME = TIME
    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.TIME, *([1] * x.dim())).transpose(0, 1)# 첫 번째 차원만 반복
        return x
class SSBH_SAE_batchnorm1d(nn.Module):
    def __init__(self, TIME, output_num):
        super(SSBH_SAE_batchnorm1d, self).__init__()
        self.TIME = TIME
        self.output_num = output_num
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(output_num) for _ in range(self.TIME)])

    def forward(self, x):
        for i in range(self.TIME):
            x[i] = self.batch_norm[i](x[i].clone().to(x.device))
        return x
class SSBH_DimChanger_for_two_three_coupling(nn.Module):
    def __init__(self):
        super(SSBH_DimChanger_for_two_three_coupling, self).__init__()

    def forward(self, x):
        assert x.dim() == 3
        B, T, F = x.shape
        x = x.reshape(B, T * F)
        return x   
class SSBH_DimChanger_for_two_three_decoupling(nn.Module): #항상TIME이 앞에 오도록 decoupling
    def __init__(self, TIME):
        super(SSBH_DimChanger_for_two_three_decoupling, self).__init__()
        self.TIME = TIME

    def forward(self, x):
        assert x.dim() == 2
        B, TF = x.shape
        x = x.reshape(B, self.TIME, -1)
        return x   
class SSBH_MultiLinearLayer(nn.Module):
    def __init__(self, time, feature):
        super(SSBH_MultiLinearLayer, self).__init__()
        self.time = time
        self.feature = feature
        self.linears = nn.ModuleList([nn.Linear(time, 1) for _ in range(feature)])

    def forward(self, x):
        # Feature 차원별로 개별적으로 Linear 적용
        outputs = [self.linears[i](x[:, :, i]) for i in range(self.feature)]  
        
        # (batch, out_dim, feature) 형태로 변환 후 batch 차원에서 concat
        outputs = torch.cat(outputs, dim=-1)  # (batch, feature)
        assert outputs.dim() == 2 and outputs.shape[0] == x.shape[0] and outputs.shape[1] == self.feature

        return outputs
class SSBH_DimSum(nn.Module):
    def __init__(self, dim=0):
        super(SSBH_DimSum, self).__init__()
        self.dim = dim
    def forward(self, x):
        x = x.sum(dim=self.dim)
        return x
class SSBH_LeakyIntegrate(nn.Module):
    def __init__(self, decay):
        super(SSBH_LeakyIntegrate, self).__init__()
        self.decay = decay
    def forward(self, x):
        for i in range(x.shape[0]):
            x[x.shape[0] - i - 1] = x[x.shape[0] - i - 1] * (self.decay ** i)
        return x



# Autoencoder 모델 정의
class SAE_fc_only(nn.Module):
    def __init__(self, encoder_ch=[96, 64, 32, 4], decoder_ch=[32,64,96,50], in_channels=1, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False, batch_norm_on=False, sae_relu_on=False):
        super(SAE_fc_only, self).__init__()
        self.encoder_ch = encoder_ch
        self.decoder_ch = decoder_ch
        self.in_channels = in_channels
        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.need_bias = need_bias
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last
        self.batch_norm_on = batch_norm_on
        assert self.decoder_ch == self.encoder_ch[:-1][::-1]+[self.in_channels]

        # self.activation_function = nn.ReLU()
        self.activation_function = neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)
        
        self.encoder = []
        self.decoder = []

        past_channel = self.in_channels

        # self.encoder.append(SSBH_size_detector())
        self.encoder += [SSBH_DimChanger_one_two()]
        if self.lif_add_at_first:
            self.encoder += [self.activation_function]
        # self.encoder.append(SSBH_size_detector())
        for en_i in range(len(self.encoder_ch)):
            # self.encoder += [SSBH_size_detector()]
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder += [nn.Linear(past_channel, self.encoder_ch[en_i], bias = self.need_bias)]
            # self.encoder += [SYNAPSE_FC_BPTT(in_features=past_channel,  # 마지막CONV의 OUT_CHANNEL * H * W
            #                 out_features=self.encoder_ch[en_i], 
            #                 trace_const1=self.synapse_fc_trace_const1,  #BPTT에선 안 씀
            #                 trace_const2=self.synapse_fc_trace_const2, #BPTT에선 안 씀
            #                 TIME=self.TIME)]
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
            if en_i != len(self.encoder_ch)-1:
                if self.batch_norm_on:
                    self.encoder += [SSBH_SAE_batchnorm1d(self.TIME, self.encoder_ch[en_i])]
                    # self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
                    # self.encoder += [nn.BatchNorm1d(self.encoder_ch[en_i])]
                    # self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
                self.encoder += [self.activation_function]
            # self.encoder.append(SSBH_size_detector())
            past_channel = self.encoder_ch[en_i]

        if sae_lif_bridge:
            if sae_relu_on:
                self.encoder += [nn.ReLU()]
            else:
                self.encoder += [self.activation_function]
            
        
        if sae_l2_norm_bridge:
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder += [SSBH_L2NormLayer()] 
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

        # self.encoder.append(SSBH_size_detector())

        self.encoder += [SSBH_DimChanger_one_two()]
        self.encoder = nn.Sequential(*self.encoder)


        self.decoder += [SSBH_DimChanger_one_two()]
        # self.decoder.append(SSBH_size_detector())
        for de_i in range(len(self.decoder_ch)):
            # self.decoder += [SSBH_size_detector()]
            self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.decoder += [nn.Linear(past_channel, self.decoder_ch[de_i], bias = self.need_bias)]
            # self.decoder += [SYNAPSE_FC_BPTT(in_features=past_channel,  # 마지막CONV의 OUT_CHANNEL * H * W
            #                 out_features=self.decoder_ch[de_i], 
            #                 trace_const1=self.synapse_fc_trace_const1,  #BPTT에선 안 씀
            #                 trace_const2=self.synapse_fc_trace_const2, #BPTT에선 안 씀
            #                 TIME=self.TIME)]
            self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

            if self.lif_add_at_last == True:
                if self.batch_norm_on:
                    self.decoder += [SSBH_SAE_batchnorm1d(self.TIME, self.decoder_ch[de_i])]
                    # self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
                    # self.decoder += [nn.BatchNorm1d(self.decoder_ch[de_i])]
                    # self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
                self.decoder += [self.activation_function]
            else:
                if de_i != len(self.decoder_ch)-1:
                    if self.batch_norm_on:
                        self.decoder += [SSBH_SAE_batchnorm1d(self.TIME, self.decoder_ch[de_i])]
                        # self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
                        # self.decoder += [nn.BatchNorm1d(self.decoder_ch[de_i])]
                        # self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
                    self.decoder += [self.activation_function]
                    
            # self.decoder.append(SSBH_size_detector())
            past_channel = self.decoder_ch[de_i]
        # self.decoder.append(SSBH_size_detector())
        self.decoder += [SSBH_DimChanger_one_two()]
        # self.decoder.append(SSBH_size_detector())
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    











# Autoencoder 모델 정의
class SAE_conv1(nn.Module):
    def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False, batch_norm_on=False, sae_relu_on=False):
        super(SAE_conv1, self).__init__()
        self.encoder_ch = encoder_ch
        self.fc_dim = fc_dim
        self.decoder_ch = self.encoder_ch[::-1]
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_padding = 0
        self.need_bias = need_bias
        self.encoder = []
        self.decoder = []
        self.current_length = input_length
        self.init_type_conv = 'kaiming_uniform'
        self.init_type_fc = "uniform"
        self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)

        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last
        self.batch_norm_on = batch_norm_on

        # self.activation_function = nn.ReLU()
        self.activation_function = neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)
        

        self.encoder += [SSBH_DimChanger_one_two()]


        # self.encoder.append(SSBH_size_detector())

        # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 2))
        if self.lif_add_at_first:
            self.encoder += [self.activation_function]
        # self.encoder.append(SSBH_size_detector())
        past_channel = self.input_channels
        for en_i in range(len(self.encoder_ch)):
            # self.encoder.append(SSBH_size_detector())
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
            if self.batch_norm_on:
                self.encoder += [SSBH_SAE_batchnorm1d(self.TIME, self.encoder_ch[en_i])]
            # self.encoder.append(SSBH_size_detector())
            self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
            past_channel = self.encoder_ch[en_i]
            self.length_save.append(self.current_length)
            self.encoder += [self.activation_function]
            # self.encoder.append(SSBH_size_detector())
            past_channel = self.encoder_ch[en_i]

        # self.encoder.append(SSBH_size_detector())
        # self.encoder += [SSBH_activation_watcher()]
        self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.encoder.append(SSBH_DimChanger_for_fc())
        fc_length = self.current_length * self.encoder_ch[-1]

        self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias))

        # self.encoder += [SSBH_size_detector()] 
        self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        if sae_lif_bridge:
            if sae_relu_on:
                self.encoder += [nn.ReLU()]
            else:
                self.encoder += [self.activation_function]
            
        if sae_l2_norm_bridge:
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder += [SSBH_L2NormLayer()]
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

        # self.encoder += [SSBH_activation_watcher()]

        self.encoder += [SSBH_DimChanger_one_two()]
        # self.encoder.append(SSBH_size_detector())
        self.encoder = nn.Sequential(*self.encoder)

        print('conv length',self.length_save)

        self.length_save = self.length_save[::-1]


        # self.decoder.append(SSBH_size_detector())
        self.decoder += [SSBH_DimChanger_one_two()]
        
        # self.decoder.append(SSBH_size_detector())
        self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.decoder.append(nn.Linear(self.fc_dim, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        # self.decoder.append(SSBH_size_detector())
        self.decoder += [self.activation_function]
        
        # self.decoder.append(SSBH_size_detector())
        self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))
        self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        # self.decoder.append(SSBH_size_detector())
        
        for de_i in range(len(self.decoder_ch)):
            if de_i != len(self.decoder_ch)-1:
                out_channel = self.decoder_ch[de_i + 1]
            else: 
                out_channel = self.input_channels # 1

            # self.decoder.append(SSBH_size_detector())
            output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )

            self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))
            # self.decoder.append(SSBH_size_detector())
            self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

            if self.lif_add_at_last == True:
                if self.batch_norm_on:
                    self.decoder += [SSBH_SAE_batchnorm1d(self.TIME, out_channel)]
                self.decoder += [self.activation_function]
            else: 
                if de_i != len(self.decoder_ch)-1:
                    if self.batch_norm_on:
                        self.decoder += [SSBH_SAE_batchnorm1d(self.TIME, out_channel)]
                    self.decoder += [self.activation_function]
            # self.decoder.append(SSBH_size_detector())
        # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=2)) 안 씀. 밖에서 그냥 채널로 받아버림.
        
        self.decoder += [SSBH_DimChanger_one_two()]
        # self.decoder.append(SSBH_size_detector())
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    











# Autoencoder 모델 정의
class SAE_conv1_DR(nn.Module): #이거 fusion net은 아니고, bridge부분만 fc로 dimension reduction한 거임. 그래서 DR이라는 postfix가 붙는다. 
    def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False, batch_norm_on=False):
        super(SAE_conv1_DR, self).__init__()
        self.encoder_ch = encoder_ch
        self.fc_dim = fc_dim
        self.decoder_ch = self.encoder_ch[::-1]
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_padding = 0
        self.need_bias = need_bias
        self.encoder = []
        self.decoder = []
        self.current_length = input_length
        self.init_type_conv = 'kaiming_uniform'
        self.init_type_fc = "uniform"
        self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)

        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last
        self.batch_norm_on = batch_norm_on

        # self.activation_function = nn.ReLU()
        self.activation_function = neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)
        

        self.encoder += [SSBH_DimChanger_one_two()]


        # self.encoder.append(SSBH_size_detector())

        # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 2))
        if self.lif_add_at_first:
            self.encoder += [self.activation_function]
        # self.encoder.append(SSBH_size_detector())
        past_channel = self.input_channels
        for en_i in range(len(self.encoder_ch)):
            # self.encoder.append(SSBH_size_detector())
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
            if self.batch_norm_on:
                self.encoder += [SSBH_SAE_batchnorm1d(self.TIME, self.encoder_ch[en_i])]
            # self.encoder.append(SSBH_size_detector())
            self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
            past_channel = self.encoder_ch[en_i]
            self.length_save.append(self.current_length)
            self.encoder += [self.activation_function]
            # self.encoder.append(SSBH_size_detector())
            past_channel = self.encoder_ch[en_i]

        # self.encoder.append(SSBH_size_detector())
        # self.encoder += [SSBH_activation_watcher()]
            
        # self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            
        self.encoder += [SSBH_DimChanger_one_two()] # Batch time channel feature
        self.encoder.append(SSBH_DimChanger_for_fc()) # Batch time*channel*feature
        fc_length = self.current_length * self.encoder_ch[-1]
        self.encoder.append(nn.Linear(fc_length * self.TIME, self.fc_dim, bias=self.need_bias))
        


        # self.encoder += [SSBH_size_detector()] 
        # self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        if sae_lif_bridge:
            self.encoder += [self.activation_function]
            # self.encoder += [nn.ReLU()]
            
        if sae_l2_norm_bridge:
            # self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder += [SSBH_L2NormLayer()]
            # self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

        # self.encoder += [SSBH_activation_watcher()]

        # self.encoder += [SSBH_DimChanger_one_two()]
        # self.encoder.append(SSBH_size_detector())
        self.encoder = nn.Sequential(*self.encoder)

        print('conv length',self.length_save)

        self.length_save = self.length_save[::-1]


        # self.decoder.append(SSBH_size_detector())
        # self.decoder += [SSBH_DimChanger_one_two()]
        
        # self.decoder.append(SSBH_size_detector())
        # self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.decoder.append(nn.Linear(self.fc_dim, self.TIME * self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        # self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        self.decoder += [SSBH_DimChanger_for_two_three_decoupling(self.TIME)] # B T F
        self.decoder += [SSBH_DimChanger_one_two()] # time batch feature


        
        

        # self.decoder.append(SSBH_size_detector())
        self.decoder += [self.activation_function]
        
        # self.decoder.append(SSBH_size_detector())
        self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))
        self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        # self.decoder.append(SSBH_size_detector())
        
        for de_i in range(len(self.decoder_ch)):
            if de_i != len(self.decoder_ch)-1:
                out_channel = self.decoder_ch[de_i + 1]
            else: 
                out_channel = self.input_channels # 1

            # self.decoder.append(SSBH_size_detector())
            output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )

            self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))
            # self.decoder.append(SSBH_size_detector())
            self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

            if self.lif_add_at_last == True:
                if self.batch_norm_on:
                    self.decoder += [SSBH_SAE_batchnorm1d(self.TIME, out_channel)]
                self.decoder += [self.activation_function]
            else: 
                if de_i != len(self.decoder_ch)-1:
                    if self.batch_norm_on:
                        self.decoder += [SSBH_SAE_batchnorm1d(self.TIME, out_channel)]
                    self.decoder += [self.activation_function]
            # self.decoder.append(SSBH_size_detector())
        # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=2)) 안 씀. 밖에서 그냥 채널로 받아버림.
        
        self.decoder += [SSBH_DimChanger_one_two()]
        # self.decoder.append(SSBH_size_detector())
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    











# Autoencoder 모델 정의
class Autoencoder_only_FC(nn.Module):
    def __init__(self, encoder_ch=[96, 64, 32, 4], decoder_ch=[32,64,96,50], n_sample=50, need_bias=False,
                 l2norm_bridge=True, relu_bridge=False, activation_collector_on=False, batch_norm_on=False, QCFS_neuron_on=False, ):
        super(Autoencoder_only_FC, self).__init__()
        self.encoder_ch = encoder_ch
        self.decoder_ch = decoder_ch
        self.n_sample = n_sample
        self.need_bias = need_bias
        self.l2norm_bridge = l2norm_bridge
        self.relu_bridge = relu_bridge
        self.activation_collector_on = activation_collector_on

        self.batch_norm_on = batch_norm_on
        self.QCFS_neuron_on = QCFS_neuron_on # True False

        level = 8
        if self.QCFS_neuron_on:
            self.activation_function = neuron.QCFS_IF(L=level, thresh=8.0)
        else:
            self.activation_function = nn.ReLU()

        assert self.decoder_ch == self.encoder_ch[:-1][::-1]+[self.n_sample]
        
        
        self.encoder = []
        past_channel = self.n_sample
        for en_i in range(len(self.encoder_ch)):
            self.encoder += [nn.Linear(past_channel, self.encoder_ch[en_i], bias = self.need_bias)]
            if self.batch_norm_on:
                self.encoder.append(nn.BatchNorm1d(self.encoder_ch[en_i]))
            if en_i != len(self.encoder_ch)-1:
                self.encoder += [self.activation_function]
                if self.activation_collector_on:
                    self.encoder += [SSBH_activation_collector()]
            past_channel = self.encoder_ch[en_i]
        
        if self.relu_bridge:
            self.encoder.append(self.activation_function)
            if self.activation_collector_on:
                self.encoder += [SSBH_activation_collector()]
        if self.l2norm_bridge:
            self.encoder.append(SSBH_L2NormLayer())

        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        past_channel = self.encoder_ch[-1]
        for de_i in range(len(self.decoder_ch)):
            self.decoder += [nn.Linear(past_channel, self.decoder_ch[de_i], bias = self.need_bias)]
            if de_i != len(self.decoder_ch)-1:
                if self.batch_norm_on:
                    self.decoder.append(nn.BatchNorm1d(self.decoder_ch[de_i]))
                self.decoder += [self.activation_function]
                if self.activation_collector_on:
                    self.decoder += [SSBH_activation_collector()]
            past_channel = self.decoder_ch[de_i]

        self.decoder = nn.Sequential(*self.decoder)

        # Xavier 초기화 적용
        self.init_type_fc = "uniform"
        # self._initialize_weights() # 일단 이거 주석처리.

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def _initialize_weights(self):
        """
        다양한 초기화 방법을 선택할 수 있도록 옵션 제공
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type_fc == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'uniform':
                    torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                else:
                    raise ValueError(f"Unknown initialization type: {self.init_type_fc}")







# Autoencoder 모델 정의 (1D Convolution 기반)
class Autoencoder_conv1(nn.Module):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
    def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, need_bias = False,
                l2norm_bridge=True, relu_bridge=False, activation_collector_on = False, batch_norm_on=False, QCFS_neuron_on=False):
        super(Autoencoder_conv1, self).__init__()
        self.encoder_ch = encoder_ch
        self.fc_dim = fc_dim
        self.decoder_ch = self.encoder_ch[::-1]
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_padding = 0
        self.need_bias = need_bias
        self.l2norm_bridge = l2norm_bridge
        self.relu_bridge = relu_bridge
        self.encoder = []
        self.decoder = []
        self.current_length = input_length
        self.init_type_conv = 'kaiming_uniform'
        self.init_type_fc = "uniform"
        self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)
        
        self.batch_norm_on = batch_norm_on # True False
        self.QCFS_neuron_on = QCFS_neuron_on # True False

        max_act = [0.53, 0.24, 0.08, 0.09, 0.88, 0.68, 0.35]
        max_act_index = 0
        level = 8
        
        if self.QCFS_neuron_on:
            self.activation_function = neuron.QCFS_IF
            # self.activation_function = neuron.QCFS_IF(L=level, thresh=8.0)
        else:
            self.activation_function = nn.ReLU()

        # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 1))
        past_channel = self.input_channels
        for en_i in range(len(self.encoder_ch)):
            # self.encoder.append(SSBH_size_detector())
            self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
            if self.batch_norm_on:
                self.encoder.append(nn.BatchNorm1d(self.encoder_ch[en_i]))
            self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
            past_channel = self.encoder_ch[en_i]
            self.length_save.append(self.current_length)
            if self.QCFS_neuron_on:
                self.encoder.append(self.activation_function(L=level, thresh=max_act[max_act_index]))
            else:
                self.encoder.append(self.activation_function)
            max_act_index += 1
            if activation_collector_on:
                self.encoder += [SSBH_activation_collector()]

        self.encoder.append(SSBH_DimChanger_for_fc())
        fc_length = self.current_length * self.encoder_ch[-1]
        self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias))
        # if self.batch_norm_on:
        #     self.encoder.append(nn.BatchNorm1d(self.fc_dim))

        # 노말라이즈 안 할 거면 빼
        if self.relu_bridge:
            if self.QCFS_neuron_on:
                self.encoder.append(self.activation_function(L=level, thresh=max_act[max_act_index]))
            else:
                self.encoder.append(self.activation_function)
            max_act_index += 1
            if activation_collector_on:
                self.encoder += [SSBH_activation_collector()]
        if self.l2norm_bridge:
            self.encoder.append(SSBH_L2NormLayer())


        self.encoder = nn.Sequential(*self.encoder)
        
        print('ae conv lenght', self.length_save)
        self.length_save = self.length_save[::-1]
        # Decoder
        # if self.l2norm_bridge==False:
        #     self.decoder.append(SSBH_L2NormLayer())

        self.decoder.append(nn.Linear(self.fc_dim, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        
        # big decoder!!!!
        # self.decoder.append(nn.Linear(self.fc_dim, 200, bias=self.need_bias))
        # self.decoder.append(self.activation_function)
        # self.decoder.append(nn.Linear(200, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        
        # if self.batch_norm_on:
        #     self.decoder.append(nn.BatchNorm1d(self.length_save[0]*self.decoder_ch[0]))
        if self.QCFS_neuron_on:
            self.decoder.append(self.activation_function(L=level, thresh=max_act[max_act_index]))
        else:
            self.decoder.append(self.activation_function)
        max_act_index += 1
        if activation_collector_on:
            self.decoder += [SSBH_activation_collector()]
        self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))

        for de_i in range(len(self.decoder_ch)):
            if de_i != len(self.decoder_ch)-1:
                out_channel = self.decoder_ch[de_i + 1]
            else: 
                out_channel = self.input_channels

            output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )

            self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))


            if de_i != len(self.decoder_ch)-1:
                if self.batch_norm_on:
                    self.decoder.append(nn.BatchNorm1d(out_channel))
                if self.QCFS_neuron_on:
                    self.decoder.append(self.activation_function(L=level, thresh=max_act[max_act_index]))
                else:
                    self.decoder.append(self.activation_function)
                max_act_index += 1
                if activation_collector_on:
                    self.decoder += [SSBH_activation_collector()]
            
        
        # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=1)) 안 씀. 밖에서 그냥 채널로 받아버림.
        self.decoder = nn.Sequential(*self.decoder)
        
        # Xavier 초기화 적용
        # self._initialize_weights()  # 일단 이거 주석처리.

    def forward(self, x):
        # Encoder
        x = self.encoder(x)  # Conv1d를 통해 압축
        # Decoder
        x = self.decoder(x)  # Transposed Conv1d를 통해 복원
        return x


    def _initialize_weights(self):
        """
        다양한 초기화 방법을 선택할 수 있도록 옵션 제공
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                if self.init_type_conv == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_conv == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_conv == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_conv == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_conv == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_conv == 'uniform':
                    torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                else:
                    raise ValueError(f"Unknown initialization type: {self.init_type_conv}")
            elif isinstance(m, nn.Linear):
                if self.init_type_fc == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'xavier_normal':
                    torch.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif self.init_type_fc == 'uniform':
                    torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                else:
                    raise ValueError(f"Unknown initialization type: {self.init_type_fc}")



class SAE_converted_fc(nn.Module):
    def __init__(self, encoder_ch=[96, 64, 32, 4], decoder_ch=[32,64,96,50], in_channels=1, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False, vth_mul_on = False, batch_norm_on = False):
        super(SAE_converted_fc, self).__init__()
        self.encoder_ch = encoder_ch
        self.decoder_ch = decoder_ch
        self.in_channels = in_channels
        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.need_bias = need_bias
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last
        self.vth_mul_on = vth_mul_on
        self.batch_norm_on = batch_norm_on

        assert self.decoder_ch == self.encoder_ch[:-1][::-1]+[self.in_channels]

        self.encoder = []
        self.decoder = []

        past_channel = self.in_channels

        # self.encoder.append(SSBH_size_detector())
        self.encoder += [SSBH_DimChanger_one_two()]
        if self.lif_add_at_first:
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
            if self.vth_mul_on:
                self.encoder += [SSBH_mul_vth(self.v_threshold)]
        # self.encoder.append(SSBH_size_detector())
        for en_i in range(len(self.encoder_ch)):
            # self.encoder += [SSBH_size_detector()]
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder += [nn.Linear(past_channel, self.encoder_ch[en_i], bias = self.need_bias)]
            if self.batch_norm_on:
                self.encoder.append(nn.BatchNorm1d(self.encoder_ch[en_i]))
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
            if en_i != len(self.encoder_ch)-1:
                self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                                v_decay=self.v_decay, 
                                                v_threshold=self.v_threshold, 
                                                v_reset=self.v_reset, 
                                                sg_width=self.sg_width,
                                                surrogate=self.surrogate,
                                                BPTT_on=self.BPTT_on)]
                if self.vth_mul_on:
                    self.encoder += [SSBH_mul_vth(self.v_threshold)]
            # self.encoder.append(SSBH_size_detector())
            past_channel = self.encoder_ch[en_i]

        self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        
        if sae_lif_bridge:
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
            if self.vth_mul_on:
                self.encoder += [SSBH_mul_vth(self.v_threshold)]
        self.encoder += [SSBH_mean(dim=0)] # time mean

        if sae_l2_norm_bridge:
            self.encoder += [SSBH_L2NormLayer()] 
        
        # self.encoder.append(SSBH_size_detector())

        self.encoder = nn.Sequential(*self.encoder)


        self.decoder = []
        past_channel = self.encoder_ch[-1]
        for de_i in range(len(self.decoder_ch)):
            self.decoder += [nn.Linear(past_channel, self.decoder_ch[de_i], bias = self.need_bias)]
            if de_i != len(self.decoder_ch)-1:
                if self.batch_norm_on:
                    self.decoder.append(nn.BatchNorm1d(self.decoder_ch[de_i]))
                self.decoder += [nn.ReLU()]
            past_channel = self.decoder_ch[de_i]

        self.decoder = nn.Sequential(*self.decoder)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    


class SAE_converted_conv1(nn.Module):
    def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False, vth_mul_on = False, batch_norm_on = False):
        super(SAE_converted_conv1, self).__init__()
        self.encoder_ch = encoder_ch
        self.fc_dim = fc_dim
        self.decoder_ch = self.encoder_ch[::-1]
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_padding = 0
        self.need_bias = need_bias
        self.encoder = []
        self.decoder = []
        self.current_length = input_length
        self.init_type_conv = 'kaiming_uniform'
        self.init_type_fc = "uniform"
        self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)

        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last


        self.vth_mul_on = vth_mul_on
        self.batch_norm_on = batch_norm_on

        self.encoder += [SSBH_DimChanger_one_two()]


        # self.encoder.append(SSBH_size_detector())

        # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 2))
        if self.lif_add_at_first:
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
            if self.vth_mul_on:
                self.encoder += [SSBH_mul_vth(self.v_threshold)]
        # self.encoder.append(SSBH_size_detector())
        past_channel = self.input_channels
        for en_i in range(len(self.encoder_ch)):
            # self.encoder.append(SSBH_size_detector())
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
            if self.batch_norm_on:
                self.encoder.append(nn.BatchNorm1d(self.encoder_ch[en_i]))
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
            # self.encoder.append(SSBH_size_detector())
            self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
            past_channel = self.encoder_ch[en_i]
            self.length_save.append(self.current_length)
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
            if self.vth_mul_on:
                self.encoder += [SSBH_mul_vth(self.v_threshold)]
            # self.encoder.append(SSBH_size_detector())
            past_channel = self.encoder_ch[en_i]

        # self.encoder.append(SSBH_size_detector())
        # self.encoder += [SSBH_activation_watcher()]
        self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.encoder.append(SSBH_DimChanger_for_fc())
        fc_length = self.current_length * self.encoder_ch[-1]
        self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias))
        if self.batch_norm_on:
            self.encoder.append(nn.BatchNorm1d(self.fc_dim))
        self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        if sae_lif_bridge:
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
            if self.vth_mul_on:
                self.encoder += [SSBH_mul_vth(self.v_threshold)]
        self.encoder += [SSBH_mean(dim=0)] # time mean
        if sae_l2_norm_bridge:
            self.encoder += [SSBH_L2NormLayer()] 
        # self.encoder += [SSBH_activation_watcher()]

        # self.encoder.append(SSBH_size_detector())
        self.encoder = nn.Sequential(*self.encoder)

        print('conv length',self.length_save)

        self.length_save = self.length_save[::-1]


        # self.encoder.append(SSBH_size_detector())

        # Decoder
        # if self.l2norm_bridge==False:
        #     self.decoder.append(SSBH_L2NormLayer())
        self.decoder.append(nn.Linear(self.fc_dim, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        if self.batch_norm_on:
            self.decoder.append(nn.BatchNorm1d(self.length_save[0]*self.decoder_ch[0]))
        self.decoder.append(nn.ReLU())
        self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))

        for de_i in range(len(self.decoder_ch)):
            if de_i != len(self.decoder_ch)-1:
                out_channel = self.decoder_ch[de_i + 1]
            else: 
                out_channel = self.input_channels

            output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )

            self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))

            if de_i != len(self.decoder_ch)-1:
                if self.batch_norm_on:
                    self.decoder.append(nn.BatchNorm1d(out_channel))
                self.decoder.append(nn.ReLU())
            
        
        # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=1)) 안 씀. 밖에서 그냥 채널로 받아버림.
        self.decoder = nn.Sequential(*self.decoder)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    


# Autoencoder 모델 정의
class FUSION_net_conv1(nn.Module): # mean으로 줄여버림
    def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False, repeat_coding=False):
        super(FUSION_net_conv1, self).__init__()
        self.encoder_ch = encoder_ch
        self.fc_dim = fc_dim
        self.decoder_ch = self.encoder_ch[::-1]
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_padding = 0
        self.need_bias = need_bias
        self.encoder = []
        self.decoder = []
        self.current_length = input_length
        self.init_type_conv = 'kaiming_uniform'
        self.init_type_fc = "uniform"
        self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)

        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last
        self.repeat_coding = repeat_coding

        if self.repeat_coding:
            self.encoder += [SSBH_repeat_coding(self.TIME)]
        else:
            self.encoder += [SSBH_rate_coding(self.TIME)]
        self.encoder += [SSBH_DimChanger_one_two()]


        # self.encoder.append(SSBH_size_detector())

        # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 2))
        if self.lif_add_at_first:
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
        # self.encoder.append(SSBH_size_detector())
        past_channel = self.input_channels
        for en_i in range(len(self.encoder_ch)):
            # self.encoder.append(SSBH_size_detector())
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
            # self.encoder.append(SSBH_size_detector())
            self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
            past_channel = self.encoder_ch[en_i]
            self.length_save.append(self.current_length)
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
            # self.encoder.append(SSBH_size_detector())
            past_channel = self.encoder_ch[en_i]

        # self.encoder.append(SSBH_size_detector())
        # self.encoder += [SSBH_activation_watcher()]
        self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.encoder.append(SSBH_DimChanger_for_fc())
        fc_length = self.current_length * self.encoder_ch[-1]
        self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias))
        # self.encoder += [SSBH_size_detector()] 
        self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        if sae_lif_bridge:
            self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)]
            
        self.encoder += [SSBH_mean(dim=0)] # time mean
        if sae_l2_norm_bridge:
            self.encoder += [SSBH_L2NormLayer()]
        # self.encoder += [SSBH_activation_watcher()]
        # self.encoder.append(SSBH_size_detector())
        self.encoder = nn.Sequential(*self.encoder)

        print('conv length',self.length_save)

        self.length_save = self.length_save[::-1]

        # Decoder
        # self.decoder.append(SSBH_size_detector())
        self.decoder.append(nn.Linear(self.fc_dim, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        self.decoder.append(nn.ReLU())
        # self.decoder.append(SSBH_size_detector())
        self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))

        # self.decoder.append(SSBH_size_detector())
        for de_i in range(len(self.decoder_ch)):
            if de_i != len(self.decoder_ch)-1:
                out_channel = self.decoder_ch[de_i + 1]
            else: 
                out_channel = self.input_channels
            output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )
            self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))
            if de_i != len(self.decoder_ch)-1 or self.lif_add_at_last:
                self.decoder.append(nn.ReLU())
            # self.decoder.append(SSBH_size_detector())
            
        
        # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=1)) 안 씀. 밖에서 그냥 채널로 받아버림.
        self.decoder = nn.Sequential(*self.decoder)
        
        # Xavier 초기화 적용
        # self._initialize_weights()  # 일단 이거 주석처리.

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    




# Autoencoder 모델 정의
class SAE_FUSION2_net_conv1(nn.Module): # fc로 한번에 줄여버림
    def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False, batch_norm_on=False, sae_relu_on=False):
        super(SAE_FUSION2_net_conv1, self).__init__()
        self.encoder_ch = encoder_ch
        self.fc_dim = fc_dim
        self.decoder_ch = self.encoder_ch[::-1]
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_padding = 0
        self.need_bias = need_bias
        self.encoder = []
        self.decoder = []
        self.current_length = input_length
        self.init_type_conv = 'kaiming_uniform'
        self.init_type_fc = "uniform"
        self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)

        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last
        self.batch_norm_on = batch_norm_on

        # self.activation_function = nn.ReLU()
        self.activation_function = neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)
        

        self.encoder += [SSBH_DimChanger_one_two()]


        # self.encoder.append(SSBH_size_detector())

        # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 2))
        if self.lif_add_at_first:
            self.encoder += [self.activation_function]
        # self.encoder.append(SSBH_size_detector())
        past_channel = self.input_channels
        for en_i in range(len(self.encoder_ch)):
            # self.encoder.append(SSBH_size_detector())
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
            if self.batch_norm_on:
                self.encoder += [SSBH_SAE_batchnorm1d(self.TIME, self.encoder_ch[en_i])]
            # self.encoder.append(SSBH_size_detector())
            self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
            past_channel = self.encoder_ch[en_i]
            self.length_save.append(self.current_length)
            self.encoder += [self.activation_function]
            # self.encoder.append(SSBH_size_detector())
            past_channel = self.encoder_ch[en_i]


        ##### batch, 4 차원으로 줄이기 ###############################################################
        
        # 한번에 24000 --> 4
        self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)] #TB C F
        self.encoder.append(SSBH_DimChanger_for_fc()) # TB CF
        self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)] #T B CF
        self.encoder += [SSBH_DimChanger_one_two()] # B T CF
        self.encoder += [SSBH_DimChanger_for_two_three_coupling()] # B TCF
        fc_length = self.current_length * self.encoder_ch[-1]
        self.encoder.append(nn.Linear(fc_length * self.TIME, self.fc_dim, bias=self.need_bias))





        # # 24000 --> 200 --> 4
        # self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)] # TB C F
        # self.encoder.append(SSBH_DimChanger_for_fc()) # TB CF
        # fc_length = self.current_length * self.encoder_ch[-1]
        # self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias)) # TB 4
        # self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)] # T B 4
        # self.encoder += [SSBH_DimChanger_one_two()] # B T 4
        # self.encoder += [SSBH_DimChanger_for_two_three_coupling()] # B T4
        # self.encoder.append(nn.Linear(self.fc_dim * self.TIME, self.fc_dim, bias=self.need_bias)) # TB 4


        # # 24000 --> 200 ---(50-1, 50-1, 50-1, 50-1)--> 4
        # self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)] # TB C F
        # self.encoder.append(SSBH_DimChanger_for_fc()) # TB CF
        # fc_length = self.current_length * self.encoder_ch[-1]
        # self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias)) # TB 4
        # self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)] # T B 4
        # self.encoder += [self.activation_function]
        # self.encoder += [SSBH_DimChanger_one_two()] # B T 4
        # self.encoder.append(SSBH_MultiLinearLayer(time=self.TIME, feature=self.fc_dim)) # B 4

        # # time meaning
        # self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)] #TB C F
        # self.encoder.append(SSBH_DimChanger_for_fc()) # TB CF
        # fc_length = self.current_length * self.encoder_ch[-1]
        # self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias)) # TB 4
        # self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)] # T B 4
        # self.encoder += [self.activation_function]
        # self.encoder += [SSBH_mean(dim=0)] # B 4


        ##### batch, 4 차원으로 줄이기 ###############################################################

        if sae_lif_bridge:
            assert False
            # if sae_relu_on:
            #     self.encoder += [nn.ReLU()]
            # else:
            #     self.encoder += [self.activation_function]

        if sae_l2_norm_bridge:
            # self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder += [SSBH_L2NormLayer()]
            # self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

        # self.encoder += [SSBH_activation_watcher()]

        # self.encoder += [SSBH_DimChanger_one_two()]
            
        # self.encoder.append(SSBH_size_detector())
        self.encoder = nn.Sequential(*self.encoder)

        print('conv length',self.length_save)

        self.length_save = self.length_save[::-1]


        # Decoder
        # self.decoder.append(SSBH_size_detector())
        self.decoder.append(nn.Linear(self.fc_dim, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        self.decoder.append(nn.ReLU())
        # self.decoder.append(SSBH_size_detector())
        self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))

        # self.decoder.append(SSBH_size_detector())
        for de_i in range(len(self.decoder_ch)):
            if de_i != len(self.decoder_ch)-1:
                out_channel = self.decoder_ch[de_i + 1]
            else: 
                out_channel = self.input_channels
            output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )
            self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))
            if de_i != len(self.decoder_ch)-1 or self.lif_add_at_last:
                self.decoder.append(nn.ReLU())
            # self.decoder.append(SSBH_size_detector())
            
        # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=1)) 안 씀. 밖에서 그냥 채널로 받아버림.
        self.decoder = nn.Sequential(*self.decoder)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    



# Autoencoder 모델 정의
class SAE_FUSION3_net_conv1(nn.Module): # 처음부터 ann conv
    def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False, batch_norm_on=False, sae_relu_on=False):
        super(SAE_FUSION3_net_conv1, self).__init__()
        self.encoder_ch = encoder_ch
        self.fc_dim = fc_dim
        self.decoder_ch = self.encoder_ch[::-1]
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_padding = 0
        self.need_bias = need_bias
        self.encoder = []
        self.decoder = []
        self.current_length = input_length
        self.init_type_conv = 'kaiming_uniform'
        self.init_type_fc = "uniform"
        self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)

        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last
        self.batch_norm_on = batch_norm_on

        # self.activation_function = nn.ReLU()
        self.activation_function = neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)
        
        # B T C F
        self.encoder += [SSBH_DimChanger_for_suqeeze(dim=2)] # B T F

        # self.encoder += [SSBH_DimChanger_one_two()] # T B C F


        # self.encoder.append(SSBH_size_detector())

        # if self.lif_add_at_first:
        #     self.encoder += [self.activation_function]
        # self.encoder.append(SSBH_size_detector())

        # past_channel = self.input_channels
        past_channel = self.TIME
        for en_i in range(len(self.encoder_ch)):
            self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
            self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
            past_channel = self.encoder_ch[en_i]
            self.length_save.append(self.current_length)
            self.encoder.append(nn.ReLU())

        self.encoder.append(SSBH_DimChanger_for_fc())
        fc_length = self.current_length * self.encoder_ch[-1]
        self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias))

        if sae_lif_bridge:
            assert False
            # if sae_relu_on:
            #     self.encoder += [nn.ReLU()]
            # else:
            #     self.encoder += [self.activation_function]

        if sae_l2_norm_bridge:
            # self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder += [SSBH_L2NormLayer()]
            # self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

        # self.encoder += [SSBH_activation_watcher()]

        # self.encoder += [SSBH_DimChanger_one_two()]
            
        # self.encoder.append(SSBH_size_detector())
        self.encoder = nn.Sequential(*self.encoder)

        print('conv length',self.length_save)

        self.length_save = self.length_save[::-1]


        # Decoder
        # self.decoder.append(SSBH_size_detector())
        self.decoder.append(nn.Linear(self.fc_dim, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        self.decoder.append(nn.ReLU())
        # self.decoder.append(SSBH_size_detector())
        self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))

        # self.decoder.append(SSBH_size_detector())
        for de_i in range(len(self.decoder_ch)):
            if de_i != len(self.decoder_ch)-1:
                out_channel = self.decoder_ch[de_i + 1]
            else: 
                out_channel = self.input_channels
            output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )
            self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))
            if de_i != len(self.decoder_ch)-1 or self.lif_add_at_last:
                self.decoder.append(nn.ReLU())
            # self.decoder.append(SSBH_size_detector())
            
        # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=1)) 안 씀. 밖에서 그냥 채널로 받아버림.
        self.decoder = nn.Sequential(*self.decoder)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    



# Autoencoder 모델 정의
class SAE_FUSION4_net_conv1(nn.Module): # integrate해서 hidden feature만들기
    def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False, batch_norm_on=False, sae_relu_on=False):
        super(SAE_FUSION4_net_conv1, self).__init__()
        self.encoder_ch = encoder_ch
        self.fc_dim = fc_dim
        self.decoder_ch = self.encoder_ch[::-1]
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_padding = 0
        self.need_bias = need_bias
        self.encoder = []
        self.decoder = []
        self.current_length = input_length
        self.init_type_conv = 'kaiming_uniform'
        self.init_type_fc = "uniform"
        self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)

        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last
        self.batch_norm_on = batch_norm_on

        # self.activation_function = nn.ReLU()
        self.activation_function = neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)
        

        self.encoder += [SSBH_DimChanger_one_two()]


        # self.encoder.append(SSBH_size_detector())

        # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 2))
        if self.lif_add_at_first:
            self.encoder += [self.activation_function]
        # self.encoder.append(SSBH_size_detector())
        past_channel = self.input_channels
        for en_i in range(len(self.encoder_ch)):
            # self.encoder.append(SSBH_size_detector())
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
            if self.batch_norm_on:
                self.encoder += [SSBH_SAE_batchnorm1d(self.TIME, self.encoder_ch[en_i])]
            # self.encoder.append(SSBH_size_detector())
            self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
            past_channel = self.encoder_ch[en_i]
            self.length_save.append(self.current_length)
            self.encoder += [self.activation_function]
            # self.encoder.append(SSBH_size_detector())
            past_channel = self.encoder_ch[en_i]

        self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.encoder.append(SSBH_DimChanger_for_fc())
        fc_length = self.current_length * self.encoder_ch[-1]

        self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias))
        self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)] # T B F
        # self.encoder += [SSBH_LeakyIntegrate(decay = 0.95)] # T B F
        self.encoder += [SSBH_DimSum(dim=0)] # B F
        if sae_lif_bridge:
            assert False
            # if sae_relu_on:
            #     self.encoder += [nn.ReLU()]
            # else:
            #     self.encoder += [self.activation_function]

        if sae_l2_norm_bridge:
            # self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder += [SSBH_L2NormLayer()]
            # self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

        # self.encoder += [SSBH_activation_watcher()]

        # self.encoder += [SSBH_DimChanger_one_two()]
            
        # self.encoder.append(SSBH_size_detector())
        self.encoder = nn.Sequential(*self.encoder)

        print('conv length',self.length_save)

        self.length_save = self.length_save[::-1]


        # Decoder
        # self.decoder.append(SSBH_size_detector())
        self.decoder.append(nn.Linear(self.fc_dim, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        self.decoder.append(nn.ReLU())
        # self.decoder.append(SSBH_size_detector())
        self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))

        # self.decoder.append(SSBH_size_detector())
        for de_i in range(len(self.decoder_ch)):
            if de_i != len(self.decoder_ch)-1:
                out_channel = self.decoder_ch[de_i + 1]
            else: 
                out_channel = self.input_channels
            output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )
            self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))
            if de_i != len(self.decoder_ch)-1 or self.lif_add_at_last:
                self.decoder.append(nn.ReLU())
            # self.decoder.append(SSBH_size_detector())
            
        # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=1)) 안 씀. 밖에서 그냥 채널로 받아버림.
        self.decoder = nn.Sequential(*self.decoder)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    




# Autoencoder 모델 정의
class SAE_FUSION5_net_conv1(nn.Module): # fc로 한번에 줄여버림
    def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
                 sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False, batch_norm_on=False, sae_relu_on=False):
        super(SAE_FUSION5_net_conv1, self).__init__()
        self.encoder_ch = encoder_ch
        self.fc_dim = fc_dim
        self.decoder_ch = self.encoder_ch[::-1]
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_padding = 0
        self.need_bias = need_bias
        self.encoder = []
        self.decoder = []
        self.current_length = input_length
        self.init_type_conv = 'kaiming_uniform'
        self.init_type_fc = "uniform"
        self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)

        self.synapse_fc_trace_const1 = synapse_fc_trace_const1
        self.synapse_fc_trace_const2 = synapse_fc_trace_const2
        self.TIME = TIME
        self.v_init = v_init
        self.v_decay = v_decay
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.sg_width = sg_width
        self.surrogate = surrogate
        self.BPTT_on = BPTT_on
        self.lif_add_at_first = lif_add_at_first
        self.lif_add_at_last = lif_add_at_last
        self.batch_norm_on = batch_norm_on

        # self.activation_function = nn.ReLU()
        self.activation_function = neuron.LIF_layer(v_init=self.v_init, 
                                            v_decay=self.v_decay, 
                                            v_threshold=self.v_threshold, 
                                            v_reset=self.v_reset, 
                                            sg_width=self.sg_width,
                                            surrogate=self.surrogate,
                                            BPTT_on=self.BPTT_on)
        

        self.encoder += [SSBH_DimChanger_one_two()]


        # self.encoder.append(SSBH_size_detector())

        # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 2))
        if self.lif_add_at_first:
            self.encoder += [self.activation_function]
        # self.encoder.append(SSBH_size_detector())
        past_channel = self.input_channels
        for en_i in range(len(self.encoder_ch)):
            # self.encoder.append(SSBH_size_detector())
            self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
            self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
            self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
            if self.batch_norm_on:
                self.encoder += [SSBH_SAE_batchnorm1d(self.TIME, self.encoder_ch[en_i])]
            # self.encoder.append(SSBH_size_detector())
            self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
            past_channel = self.encoder_ch[en_i]
            self.length_save.append(self.current_length)
            self.encoder += [self.activation_function]
            # self.encoder.append(SSBH_size_detector())
            past_channel = self.encoder_ch[en_i]


        # self.encoder.append(SSBH_size_detector())
        # self.encoder += [SSBH_activation_watcher()]
        self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        self.encoder.append(SSBH_DimChanger_for_fc())
        fc_length = self.current_length * self.encoder_ch[-1]

        self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias))

        # self.encoder += [SSBH_size_detector()] 
        self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        # if sae_lif_bridge:
        #     if sae_relu_on:
        #         self.encoder += [nn.ReLU()]
        #     else:
        #     self.encoder += [self.activation_function]
        self.encoder += [self.activation_function]
            
        # if sae_l2_norm_bridge:
        #     self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
        #     self.encoder += [SSBH_L2NormLayer()]
        #     self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

        # self.encoder += [SSBH_activation_watcher()]

        self.encoder += [SSBH_DimChanger_one_two()]
        # self.encoder.append(SSBH_size_detector())
        self.encoder = nn.Sequential(*self.encoder)

        print('conv length',self.length_save)

        self.length_save = self.length_save[::-1]

        # B T F

        # Decoder
        self.decoder += [SSBH_DimChanger_for_two_three_coupling()] # B TF
        # self.decoder.append(SSBH_size_detector())
        self.decoder.append(nn.Linear(self.fc_dim*self.TIME, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
        self.decoder.append(nn.ReLU())
        # self.decoder.append(SSBH_size_detector())
        self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))

        # self.decoder.append(SSBH_size_detector())
        for de_i in range(len(self.decoder_ch)):
            if de_i != len(self.decoder_ch)-1:
                out_channel = self.decoder_ch[de_i + 1]
            else: 
                out_channel = self.input_channels
            output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )
            self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))
            if de_i != len(self.decoder_ch)-1 or self.lif_add_at_last:
                self.decoder.append(nn.ReLU())
            # self.decoder.append(SSBH_size_detector())
            
        # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=1)) 안 씀. 밖에서 그냥 채널로 받아버림.
        self.decoder = nn.Sequential(*self.decoder)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    






# class SAE_converted_fc_2(nn.Module):
#     def __init__(self, encoder_ch=[96, 64, 32, 4], decoder_ch=[32,64,96,50], in_channels=1, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
#                  sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False):
#         super(SAE_converted_fc_2, self).__init__()
#         self.encoder_ch = encoder_ch
#         self.decoder_ch = decoder_ch
#         self.in_channels = in_channels
#         self.synapse_fc_trace_const1 = synapse_fc_trace_const1
#         self.synapse_fc_trace_const2 = synapse_fc_trace_const2
#         self.TIME = TIME
#         self.v_init = v_init
#         self.v_decay = v_decay
#         self.v_threshold = v_threshold
#         self.v_reset = v_reset
#         self.sg_width = sg_width
#         self.surrogate = surrogate
#         self.BPTT_on = BPTT_on
#         self.need_bias = need_bias
#         self.lif_add_at_first = lif_add_at_first
#         self.lif_add_at_last = lif_add_at_last

#         assert self.decoder_ch == self.encoder_ch[:-1][::-1]+[self.in_channels]

#         self.encoder = []
#         self.decoder = []

#         past_channel = self.in_channels

#         # self.encoder.append(SSBH_size_detector())
#         self.encoder += [SSBH_DimChanger_one_two()]
#         if self.lif_add_at_first:
#             self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
#                                             v_decay=self.v_decay, 
#                                             v_threshold=self.v_threshold, 
#                                             v_reset=self.v_reset, 
#                                             sg_width=self.sg_width,
#                                             surrogate=self.surrogate,
#                                             BPTT_on=self.BPTT_on)]
#             self.encoder += [SSBH_mul_vth(self.v_threshold)]
#         # self.encoder.append(SSBH_size_detector())
#         for en_i in range(len(self.encoder_ch)):
#             # self.encoder += [SSBH_size_detector()]
#             self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
#             self.encoder += [nn.Linear(past_channel, self.encoder_ch[en_i], bias = self.need_bias)]
#             self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
#             self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
#                                             v_decay=self.v_decay, 
#                                             v_threshold=self.v_threshold, 
#                                             v_reset=self.v_reset, 
#                                             sg_width=self.sg_width,
#                                             surrogate=self.surrogate,
#                                             BPTT_on=self.BPTT_on)]
#             self.encoder += [SSBH_mul_vth(self.v_threshold)]
#             # self.encoder.append(SSBH_size_detector())
#             past_channel = self.encoder_ch[en_i]

#         self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
#         self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
        
#         if sae_lif_bridge:
#             self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
#                                             v_decay=self.v_decay, 
#                                             v_threshold=self.v_threshold, 
#                                             v_reset=self.v_reset, 
#                                             sg_width=self.sg_width,
#                                             surrogate=self.surrogate,
#                                             BPTT_on=self.BPTT_on)]
#             self.encoder += [SSBH_mul_vth(self.v_threshold)]
#         self.encoder += [SSBH_mean(dim=0)] # time mean

#         if sae_l2_norm_bridge:
#             self.encoder += [SSBH_L2NormLayer()] 
        
#         # self.encoder.append(SSBH_size_detector())

#         self.encoder = nn.Sequential(*self.encoder)

#         self.decoder += [SSBH_repeat(self.TIME)]
#         self.decoder += [SSBH_DimChanger_one_two()]
#         # self.decoder.append(SSBH_size_detector())
#         for de_i in range(len(self.decoder_ch)):
#             # self.decoder += [SSBH_size_detector()]
#             self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
#             self.decoder += [nn.Linear(past_channel, self.decoder_ch[de_i], bias = self.need_bias)]
#             # self.decoder += [SYNAPSE_FC_BPTT(in_features=past_channel,  # 마지막CONV의 OUT_CHANNEL * H * W
#             #                 out_features=self.decoder_ch[de_i], 
#             #                 trace_const1=self.synapse_fc_trace_const1,  #BPTT에선 안 씀
#             #                 trace_const2=self.synapse_fc_trace_const2, #BPTT에선 안 씀
#             #                 TIME=self.TIME)]
#             self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

#             if self.lif_add_at_last == True:
#                 self.decoder += [neuron.LIF_layer(v_init=self.v_init, 
#                                                 v_decay=self.v_decay, 
#                                                 v_threshold=self.v_threshold, 
#                                                 v_reset=self.v_reset, 
#                                                 sg_width=self.sg_width,
#                                                 surrogate=self.surrogate,
#                                                 BPTT_on=self.BPTT_on)]
#                 self.decoder += [SSBH_mul_vth(self.v_threshold)]
#             else:
#                 if de_i != len(self.decoder_ch)-1:
#                     self.decoder += [neuron.LIF_layer(v_init=self.v_init, 
#                                                     v_decay=self.v_decay, 
#                                                     v_threshold=self.v_threshold, 
#                                                     v_reset=self.v_reset, 
#                                                     sg_width=self.sg_width,
#                                                     surrogate=self.surrogate,
#                                                     BPTT_on=self.BPTT_on)]
#                     self.decoder += [SSBH_mul_vth(self.v_threshold)]
                    
#             # self.decoder.append(SSBH_size_detector())
#             past_channel = self.decoder_ch[de_i]
#         # self.decoder.append(SSBH_size_detector())
#         self.decoder += [SSBH_DimChanger_one_two()]
#         # self.decoder.append(SSBH_size_detector())
#         self.decoder = nn.Sequential(*self.decoder)



#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    


# class SAE_converted_conv1_2(nn.Module):
#     def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, synapse_fc_trace_const1=1,synapse_fc_trace_const2=0.7, TIME=10, v_init=0.0, v_decay=0.5, v_threshold=0.75, v_reset=10000.0, sg_width=4.0, surrogate='sigmoid', BPTT_on=True, need_bias=False, lif_add_at_first=True,
#                  sae_l2_norm_bridge = True, sae_lif_bridge = False, lif_add_at_last = False):
#         super(SAE_converted_conv1_2, self).__init__()
#         self.encoder_ch = encoder_ch
#         self.fc_dim = fc_dim
#         self.decoder_ch = self.encoder_ch[::-1]
#         self.padding = padding
#         self.stride = stride
#         self.kernel_size = kernel_size
#         self.input_channels = input_channels
#         self.input_length = input_length
#         self.output_padding = 0
#         self.need_bias = need_bias
#         self.encoder = []
#         self.decoder = []
#         self.current_length = input_length
#         self.init_type_conv = 'kaiming_uniform'
#         self.init_type_fc = "uniform"
#         self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)

#         self.synapse_fc_trace_const1 = synapse_fc_trace_const1
#         self.synapse_fc_trace_const2 = synapse_fc_trace_const2
#         self.TIME = TIME
#         self.v_init = v_init
#         self.v_decay = v_decay
#         self.v_threshold = v_threshold
#         self.v_reset = v_reset
#         self.sg_width = sg_width
#         self.surrogate = surrogate
#         self.BPTT_on = BPTT_on
#         self.lif_add_at_first = lif_add_at_first
#         self.lif_add_at_last = lif_add_at_last


#         self.encoder += [SSBH_DimChanger_one_two()]


#         # self.encoder.append(SSBH_size_detector())

#         # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 2))
#         if self.lif_add_at_first:
#             self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
#                                             v_decay=self.v_decay, 
#                                             v_threshold=self.v_threshold, 
#                                             v_reset=self.v_reset, 
#                                             sg_width=self.sg_width,
#                                             surrogate=self.surrogate,
#                                             BPTT_on=self.BPTT_on)]
#             self.encoder += [SSBH_mul_vth(self.v_threshold)]
#         # self.encoder.append(SSBH_size_detector())
#         past_channel = self.input_channels
#         for en_i in range(len(self.encoder_ch)):
#             # self.encoder.append(SSBH_size_detector())
#             self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
#             self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
#             self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
#             # self.encoder.append(SSBH_size_detector())
#             self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
#             past_channel = self.encoder_ch[en_i]
#             self.length_save.append(self.current_length)
#             self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
#                                             v_decay=self.v_decay, 
#                                             v_threshold=self.v_threshold, 
#                                             v_reset=self.v_reset, 
#                                             sg_width=self.sg_width,
#                                             surrogate=self.surrogate,
#                                             BPTT_on=self.BPTT_on)]
#             self.encoder += [SSBH_mul_vth(self.v_threshold)]
#             # self.encoder.append(SSBH_size_detector())
#             past_channel = self.encoder_ch[en_i]

#         # self.encoder.append(SSBH_size_detector())
#         # self.encoder += [SSBH_activation_watcher()]
#         self.encoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
#         self.encoder.append(SSBH_DimChanger_for_fc())
#         fc_length = self.current_length * self.encoder_ch[-1]
#         self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias))
#         self.encoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
#         if sae_lif_bridge:
#             self.encoder += [neuron.LIF_layer(v_init=self.v_init, 
#                                             v_decay=self.v_decay, 
#                                             v_threshold=self.v_threshold, 
#                                             v_reset=self.v_reset, 
#                                             sg_width=self.sg_width,
#                                             surrogate=self.surrogate,
#                                             BPTT_on=self.BPTT_on)]
#             self.encoder += [SSBH_mul_vth(self.v_threshold)]
#         self.encoder += [SSBH_mean(dim=0)] # time mean
#         if sae_l2_norm_bridge:
#             self.encoder += [SSBH_L2NormLayer()] 
#         # self.encoder += [SSBH_activation_watcher()]

#         # self.encoder.append(SSBH_size_detector())
#         self.encoder = nn.Sequential(*self.encoder)


#         print('conv length',self.length_save)

#         self.length_save = self.length_save[::-1]


#         # self.decoder.append(SSBH_size_detector())
#         self.decoder += [SSBH_repeat(self.TIME)]
#         self.decoder += [SSBH_DimChanger_one_two()]
        
#         # self.decoder.append(SSBH_size_detector())
#         self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
#         self.decoder.append(nn.Linear(self.fc_dim, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
#         self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
#         # self.decoder.append(SSBH_size_detector())
#         self.decoder += [neuron.LIF_layer(v_init=self.v_init, 
#                                         v_decay=self.v_decay, 
#                                         v_threshold=self.v_threshold, 
#                                         v_reset=self.v_reset, 
#                                         sg_width=self.sg_width,
#                                         surrogate=self.surrogate,
#                                         BPTT_on=self.BPTT_on)]
#         self.decoder += [SSBH_mul_vth(self.v_threshold)]
        
#         # self.decoder.append(SSBH_size_detector())
#         self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
#         self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))
#         self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]
#         # self.decoder.append(SSBH_size_detector())
        
#         for de_i in range(len(self.decoder_ch)):
#             if de_i != len(self.decoder_ch)-1:
#                 out_channel = self.decoder_ch[de_i + 1]
#             else: 
#                 out_channel = self.input_channels # 1

#             # self.decoder.append(SSBH_size_detector())
#             output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )

#             self.decoder += [SSBH_DimChanger_for_one_two_coupling(self.TIME)]
#             self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))
#             self.decoder += [SSBH_DimChanger_for_one_two_decoupling(self.TIME)]

#             if self.lif_add_at_last == True:
#                 self.decoder += [neuron.LIF_layer(v_init=self.v_init, 
#                                                 v_decay=self.v_decay, 
#                                                 v_threshold=self.v_threshold, 
#                                                 v_reset=self.v_reset, 
#                                                 sg_width=self.sg_width,
#                                                 surrogate=self.surrogate,
#                                                 BPTT_on=self.BPTT_on)]
#                 self.decoder += [SSBH_mul_vth(self.v_threshold)]
#             else: 
#                 if de_i != len(self.decoder_ch)-1:
#                     self.decoder += [neuron.LIF_layer(v_init=self.v_init, 
#                                                     v_decay=self.v_decay, 
#                                                     v_threshold=self.v_threshold, 
#                                                     v_reset=self.v_reset, 
#                                                     sg_width=self.sg_width,
#                                                     surrogate=self.surrogate,
#                                                     BPTT_on=self.BPTT_on)]
#                     self.decoder += [SSBH_mul_vth(self.v_threshold)]
#         # self.decoder.append(SSBH_size_detector())
#         # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=2)) 안 씀. 밖에서 그냥 채널로 받아버림.
        
#         self.decoder += [SSBH_DimChanger_one_two()]
#         # self.decoder.append(SSBH_size_detector())
#         self.decoder = nn.Sequential(*self.decoder)

        
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    



















# # Autoencoder 모델 정의 (1D Convolution 기반)
# class Autoencoder_conv1_old(nn.Module):
#     # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
#     # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
#     def __init__(self, input_channels=1, input_length=50, encoder_ch = [32, 64, 96], fc_dim = 4, padding = 0, stride = 2, kernel_size = 3, need_bias = False,
#                 l2norm_bridge=True, relu_bridge=False):
#         super(Autoencoder_conv1_old, self).__init__()
#         self.encoder_ch = encoder_ch
#         self.fc_dim = fc_dim
#         self.decoder_ch = self.encoder_ch[::-1]
#         self.padding = padding
#         self.stride = stride
#         self.kernel_size = kernel_size
#         self.input_channels = input_channels
#         self.input_length = input_length
#         self.output_padding = 0
#         self.need_bias = need_bias
#         self.l2norm_bridge = l2norm_bridge
#         self.relu_bridge = relu_bridge
#         self.encoder = []
#         self.decoder = []
#         self.current_length = input_length
#         self.init_type_conv = 'kaiming_uniform'
#         self.init_type_fc = "uniform"
#         self.length_save = [input_length] # [50, 24, 11, 5] (encoder_ch길이보다 1개 많다)
        

#         # self.encoder.append(SSBH_DimChanger_for_unsuqeeze(dim = 1))
#         past_channel = self.input_channels
#         for en_i in range(len(self.encoder_ch)):
#             # self.encoder.append(SSBH_size_detector())
#             self.encoder.append(nn.Conv1d(in_channels=past_channel, out_channels=self.encoder_ch[en_i], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.need_bias))
#             self.current_length = (self.current_length + 2*self.padding - (self.kernel_size-1) - 1)//self.stride + 1
#             past_channel = self.encoder_ch[en_i]
#             self.length_save.append(self.current_length)
#             self.encoder.append(nn.ReLU())

#         # self.encoder.append(SSBH_size_detector())
#         self.encoder.append(SSBH_DimChanger_for_fc())
#         fc_length = self.current_length * self.encoder_ch[-1]
#         self.encoder.append(nn.Linear(fc_length, self.fc_dim, bias=self.need_bias))
#         # self.encoder.append(SSBH_size_detector())

#         # 노말라이즈 안 할 거면 빼
#         if self.relu_bridge:
#             self.encoder.append(nn.ReLU())
#         if self.l2norm_bridge:
#             self.encoder.append(SSBH_L2NormLayer())

#         # self.encoder.append(SSBH_size_detector())

#         self.encoder = nn.Sequential(*self.encoder)

#         self.length_save = self.length_save[::-1]
#         # Decoder
#         # if self.l2norm_bridge==False:
#         #     self.decoder.append(SSBH_L2NormLayer())
#         self.decoder.append(nn.Linear(self.fc_dim, self.length_save[0]*self.decoder_ch[0], bias=self.need_bias))
#         self.decoder.append(nn.ReLU())
#         self.decoder.append(SSBH_DimChanger_for_conv1(self.decoder_ch[0]))

#         for de_i in range(len(self.decoder_ch)):
#             if de_i != len(self.decoder_ch)-1:
#                 out_channel = self.decoder_ch[de_i + 1]
#             else: 
#                 out_channel = self.input_channels

#             output_padding = self.length_save[de_i + 1] - ( (self.length_save[de_i] - 1) * self.stride - 2 * self.padding + self.kernel_size - 1 + 1  )

#             self.decoder.append(nn.ConvTranspose1d(in_channels=self.decoder_ch[de_i], out_channels=out_channel, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=output_padding, bias=self.need_bias))

#             if de_i != len(self.decoder_ch)-1:
#                 self.decoder.append(nn.ReLU())
            
        
#         # self.decoder.append(SSBH_DimChanger_for_suqeeze(dim=1)) 안 씀. 밖에서 그냥 채널로 받아버림.
#         self.decoder = nn.Sequential(*self.decoder)
        
#         # Xavier 초기화 적용
#         # self._initialize_weights()  # 일단 이거 주석처리.

#     def forward(self, x):
#         # Encoder
#         x = self.encoder(x)  # Conv1d를 통해 압축
#         # Decoder
#         x = self.decoder(x)  # Transposed Conv1d를 통해 복원
#         return x


#     def _initialize_weights(self):
#         """
#         다양한 초기화 방법을 선택할 수 있도록 옵션 제공
#         """
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
#                 if self.init_type_conv == 'xavier_uniform':
#                     torch.nn.init.xavier_uniform_(m.weight)
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 elif self.init_type_conv == 'xavier_normal':
#                     torch.nn.init.xavier_normal_(m.weight)
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 elif self.init_type_conv == 'kaiming_uniform':
#                     torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 elif self.init_type_conv == 'kaiming_normal':
#                     torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 elif self.init_type_conv == 'normal':
#                     torch.nn.init.normal_(m.weight, mean=0, std=0.02)
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 elif self.init_type_conv == 'uniform':
#                     torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 else:
#                     raise ValueError(f"Unknown initialization type: {self.init_type_conv}")
#             elif isinstance(m, nn.Linear):
#                 if self.init_type_fc == 'xavier_uniform':
#                     torch.nn.init.xavier_uniform_(m.weight)
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 elif self.init_type_fc == 'xavier_normal':
#                     torch.nn.init.xavier_normal_(m.weight)
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 elif self.init_type_fc == 'kaiming_uniform':
#                     torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 elif self.init_type_fc == 'kaiming_normal':
#                     torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 elif self.init_type_fc == 'normal':
#                     torch.nn.init.normal_(m.weight, mean=0, std=0.02)
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 elif self.init_type_fc == 'uniform':
#                     torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias)
#                 else:
#                     raise ValueError(f"Unknown initialization type: {self.init_type_fc}")





