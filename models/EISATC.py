"""
Copyright (C) 2023 Qufu Normal University, Guangjin Liang
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Author:  Guangjin Liang
"""
import os
import sys
import numpy as np
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)
if not hasattr(np, 'long'):
    np.long = np.int64

import torch
import torch.nn as nn
from torchinfo import summary
from torchstat import stat
from torch.autograd import Function
from einops import rearrange


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, bias=False, WeightNorm=False, group=True, max_norm=1.):
        super(TemporalBlock, self).__init__()
        if group:
            if n_inputs >= n_outputs:
                self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                                  dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_outputs)
            else:
                self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                                  dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_inputs)
            self.conv1_point = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size=1, stride=1, bias=bias)
        else:
            self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu1 = nn.ELU() # inplace=True
        self.dropout1 = nn.Dropout(dropout)

        if group:
            self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm, groups=n_outputs)
            self.conv2_point = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size=1, stride=1, bias=bias)
        else:
            self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        if group:
            self.net = nn.Sequential(self.conv1, self.conv1_point, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                     self.conv2, self.conv2_point, self.chomp2, self.bn2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = out+res
        out = self.relu(out)
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, bias=False, WeightNorm=False, group=True, max_norm=1.):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout, bias=bias, WeightNorm=WeightNorm, group=group, max_norm=max_norm)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def relative_pos_dis(height=1, weight=32):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    if height > 1:
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2 # Wh*Ww, Wh*Ww
    else:
        relative_coords = coords_flatten[1, :, None] - coords_flatten[1, None, :]  # Wh*Ww, Wh*Ww
        relative_coords = relative_coords.contiguous()  # Wh*Ww, Wh*Ww
        dis = relative_coords # Wh*Ww, Wh*Ww
    return  dis




class CNNAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, keral_size=3, dropout=0., patch_height=1, patch_width=1, max_norm1=1., max_norm2=1., device='cpu', groups=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.patch_width = patch_width

        if groups:
            self.to_qkv = Conv2dWithConstraint(dim, inner_dim*3, kernel_size=(1, keral_size), padding='same', bias=False, max_norm=max_norm1, groups=dim)
        else:
            self.to_qkv = Conv2dWithConstraint(dim, inner_dim*3, kernel_size=(1,keral_size), padding='same', bias=False, max_norm=max_norm1)

        self.dis = relative_pos_dis(patch_height, patch_width).to(device) # n n
        self.headsita = nn.Parameter(torch.randn(heads), requires_grad=True)
        self.sig = nn.Sigmoid()
        self.ones_matrix = torch.ones(patch_height*patch_width, patch_height*patch_width).to(device)

        self.to_out = nn.Sequential(
            Conv2dWithConstraint(inner_dim, dim, kernel_size=1, padding=0, bias=False, max_norm=max_norm2),
            nn.BatchNorm2d(dim), # inner_dim
            nn.ELU(), # inplace=True
            nn.Dropout(p=dropout)
        )

    def forward(self, x, mode="train", smooth=1e-4):
        qkv = self.to_qkv(x).chunk(3, dim=1) # b (g d) h w
        q, k, v = map(lambda t: rearrange(t, 'b (g d) h w -> b g (h w) d', g=self.heads), qkv) # b g n d
        attn = torch.matmul(q, k.transpose(-1, -2)) # b g n n
        qk_norm = torch.sqrt(torch.sum(q ** 2, dim=-1)+smooth)[:, :, :, None] * torch.sqrt(torch.sum(k ** 2, dim=-1)+smooth)[:, :, None, :] + smooth
        attn = attn/qk_norm # b g n n

        f = self.ones_matrix*(self.sig(self.headsita)*(self.patch_width-1)+1)[:, None, None]
        cycle_attn = 0.5*torch.cos(f*self.dis)+0.5 # g n n
        attention = attn * cycle_attn[None, :, :, :] # b g n n

        out = torch.matmul(attention, v) # b g n d

        out = rearrange(out, 'b g (h w) d -> b (g d) h w', h=x.shape[2])
        if mode=="train":
            return self.to_out(out)
        elif mode=="test":
            return self.to_out(out), attention, attn, cycle_attn

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None



class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based braincomputer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        # if self.bias:
        #     self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

    def __call__(self, *input, **kwargs):
        return super()._call_impl(*input, **kwargs)


class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based braincomputer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)
        if self.bias:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)



class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PEAttention(nn.Module):
    def __init__(self, kernel_size=(3, 3)):
        super(PEAttention, self).__init__()

        self.maxpol1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.maxpol2 = nn.MaxPool2d(kernel_size=(5, 5), stride=1, padding=(2, 2))
        self.maxpol3 = nn.MaxPool2d(kernel_size=(7, 7), stride=1, padding=(3, 3))
        self.conv1 = nn.Conv2d(3, 1, kernel_size=(1, 1), stride=1, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        max_out1 = self.maxpol1(avg_out)
        max_out2 = self.maxpol2(avg_out)
        max_out3 = self.maxpol3(avg_out)
        out = torch.cat([max_out1, max_out2, max_out3], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class SEAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, C, 1, W)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # (batch_size, channels, 1, width)
        y = self.avg_pool(x).view(b, c)  # (batch_size, channels)
        y = self.fc(y).view(b, c, 1, 1)  # (batch_size, channels, 1, 1)
        return x * y.expand_as(x)  #


class DSEAttention(nn.Module):
    def __init__(self, channels, reduction_list=[4, 8, 16]):
        super(DSEAttention, self).__init__()
        self.branches = nn.ModuleList([
            SEAttention(channels, r) for r in reduction_list
        ])
        self.num_branches = len(reduction_list)

        self.gate = nn.Sequential(
            # nn.Linear(channels, channels // 4),
            # nn.ReLU(inplace=True),
            nn.Linear(channels, self.num_branches),
            nn.Softmax(dim=1)  # (B, K)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: (B, C, 1, T)
        B, C, _, T = x.shape

        # Global AvgPool over temporal dim
        z = x.mean(dim=-1).squeeze(2)  # shape: (B, C)

        gate_weights = self.gate(z)  # shape: (B, K)

        branch_outputs = []
        for branch in self.branches:
            out = branch(z)  # shape: (B, C)
            branch_outputs.append(out)
        # Stack to (B, K, C)
        stacked = torch.stack(branch_outputs, dim=1)  # (B, K, C)
        gate_weights = gate_weights.unsqueeze(-1)  # (B, K, 1)
        fused = (stacked * gate_weights).sum(dim=1)  # (B, C)
        scale = self.sigmoid(fused).unsqueeze(2).unsqueeze(-1)  # (B, C, 1, 1)

        return x * scale.expand_as(x)


class DCTFCAttention(nn.Module):
    def __init__(self, channels, reduction=8, fs=250, band=(8, 30)):
        super(DCTFCAttention, self).__init__()
        self.fs = fs
        self.band = band
        self.reduction = reduction

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def dct_2d(self, x):
        # x: [B, C1, C2, T] -> apply 2D-DCT on last two dims (C2, T)
        # Step 1: DCT over time (T)
        N = x.size(-1)
        x_v = torch.cat([x, x.flip(dims=[-1])], dim=-1)  # [B, C1, C2, 2T]
        X = torch.fft.fft(x_v, dim=-1).real[..., :N] / 2  # [B, C1, C2, T]

        # Step 2: DCT over spatial (C2)
        N2 = x.size(-2)
        X_v = torch.cat([X, X.flip(dims=[-2])], dim=-2)  # [B, C1, 2C2, T]
        X2D = torch.fft.fft(X_v, dim=-2).real[..., :N2, :] / 2  # [B, C1, C2, T]

        return X2D

    def forward(self, x):  # x: [B, C1, C2, T]
        B, C1, C2, T = x.shape

        x_freq = self.dct_2d(x)  # [B, C1, C2, T]

        f_step = self.fs / (2 * T)
        low_idx = int(self.band[0] / f_step)
        high_idx = int(self.band[1] / f_step)

        band_power = (x_freq[..., low_idx:high_idx] ** 2).mean(dim=-1)  # [B, C1, C2]
        channel_power = band_power.mean(dim=-1)  # [B, C1]

        attn = self.fc(channel_power).unsqueeze(-1).unsqueeze(-1)  # [B, C1, 1, 1]

        return x * attn

class TemporalInception(nn.Module):
    def __init__(self, in_chan=1, kerSize_1=(1, 3), kerSize_2=(1, 5), kerSize_3=(1, 7),
                 kerStr=1, out_chan=4, pool_ker=(1, 3), pool_str=1, bias=False, max_norm=1., point_fusion=True):
        '''
        Inception模块的实现代码,

        '''
        super(TemporalInception, self).__init__()
        self.point_fusion = point_fusion

        self.conv1 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_1,
            stride=kerStr,
            padding='same',
            groups=out_chan,
            bias=bias,
            max_norm=max_norm
        )

        self.conv2 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_2,
            stride=kerStr,
            padding='same',
            groups=out_chan,
            bias=bias,
            max_norm=max_norm
        )

        self.conv3 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_3,
            stride=kerStr,
            padding='same',
            groups=out_chan,
            bias=bias,
            max_norm=max_norm
        )

        self.pool4 = nn.MaxPool2d(
            kernel_size=pool_ker,
            stride=pool_str,
            padding=(round(pool_ker[0] / 2 + 0.1) - 1, round(pool_ker[1] / 2 + 0.1) - 1)
        )
        self.conv4 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=1,
            stride=1,
            bias=bias,
            max_norm=max_norm
        )

        if point_fusion:
            self.point_conv = Conv2dWithConstraint(
                in_channels=in_chan,
                out_channels=in_chan,
                kernel_size=1,
                stride=1,
                bias=False
            )

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(x)
        p4 = self.conv4(self.pool4(x))
        out = torch.cat((p1, p2, p3, p4), dim=1)
        if self.point_fusion:
            out = self.point_conv(out)
        return out


# %%
class EISATC_feature(nn.Module):
    def __init__(self, eeg_chans=22, samples=1000,
                 kerSize=32, kerSize_Tem=4, F1=16, D=2, poolSize1=8, poolSize2=8,
                 heads_num=8, head_dim=8,
                 tcn_filters=32, tcn_kernelSize=4,
                 dropout_dep=0.1, dropout_temp=0.3, dropout_atten=0.3, dropout_tcn=0.3,
                 n_classes=4, device='cpu'):
        super(EISATC_feature, self).__init__()
        self.F2 = F1 * D

        # ============================= EEGINC model =============================
        self.temp_conv = Conv2dWithConstraint(  # Conv2dWithConstraint( # sincConv
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kerSize),
            stride=1,
            padding='same',
            bias=False,
            max_norm=.5
        )
        self.bn = nn.BatchNorm2d(num_features=F1)  # bn_sinc

        self.conv_depth = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(eeg_chans, 1),
            groups=F1,
            bias=False,
            max_norm=.5
        )
        self.bn_depth = nn.BatchNorm2d(num_features=self.F2)
        self.act_depth = nn.ELU()  # inplace=True
        self.avgpool_depth = nn.AvgPool2d(
            kernel_size=(1, poolSize1),
            stride=(1, poolSize1)
        )
        self.drop_depth = nn.Dropout(p=dropout_dep)

        self.incept_temp = TemporalInception(
            in_chan=self.F2,
            kerSize_1=(1, kerSize_Tem * 4),
            kerSize_2=(1, kerSize_Tem * 2),
            kerSize_3=(1, kerSize_Tem),
            kerStr=1,
            out_chan=self.F2 // 4,
            pool_ker=(1, 3),
            pool_str=1,
            bias=False,
            max_norm=.5
        )
        self.bn_temp = nn.BatchNorm2d(num_features=self.F2)
        self.act_temp = nn.ELU()
        self.avgpool_temp = nn.AvgPool2d(
            kernel_size=(1, poolSize2),
            stride=(1, poolSize2)
        )
        self.drop_temp = nn.Dropout(p=dropout_temp)

        # ============================= Decision Fusion model =============================
        self.flatten_eeg = nn.Flatten()
        self.liner_eeg = LinearWithConstraint(
            in_features=self.F2 * (samples // poolSize1 // poolSize2),
            out_features=n_classes,
            max_norm=.5,
            bias=True
        )

        # ============================= MSA model =============================
        self.layerNorm = nn.LayerNorm(
            normalized_shape=(samples // poolSize1 // poolSize2),
            eps=1e-6
        )
        self.cnnMSA = CNNAttention(
            dim=self.F2,
            heads=heads_num,
            dim_head=head_dim,
            keral_size=3,
            patch_height=1,
            patch_width=(samples // poolSize1 // poolSize2),
            dropout=dropout_atten,
            max_norm1=.5,
            max_norm2=.5,
            device=device,
            groups=True
        )

        # ============================= TCN model =============================
        self.tcn_block = TemporalConvNet(
            num_inputs=self.F2 * 2,
            num_channels=[tcn_filters * 2, tcn_filters * 2],
            kernel_size=tcn_kernelSize,
            dropout=dropout_tcn,
            bias=False,
            WeightNorm=True,
            group=True,
            max_norm=.5
        )

        # ============================= Decision Fusion model =============================
        self.flatten_tcn = nn.Flatten()
        self.liner_tcn = LinearWithConstraint(
            in_features=tcn_filters * 2,
            out_features=n_classes,
            max_norm=.5,
            bias=True
        )

        # ============================= Faeture Fusion model EEG TCN =============================
        # self.flatten_fusion = nn.Flatten()
        # self.liner_fusion = LinearWithConstraint(
        #     in_features  = self.F2*((samples//poolSize1//poolSize2)*1+1),
        #     out_features = n_classes,
        #     max_norm     = .5,
        #     bias         = True
        # )

        # ============================= Decision Fusion model =============================
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True))
        self.beta_sigmoid = nn.Sigmoid()

        # self.flatten = nn.Flatten()
        # self.liner_cla = LinearWithConstraint(
        #     in_features=self.F2*(samples//poolSize1//poolSize2), # tcn_filters, self.F2*(samples//poolSize1//poolSize2)
        #     out_features=n_classes,
        #     max_norm=.5,
        #     bias=True
        # )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, pooling=True):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
            x = x.transpose(2, 3)
        # ============================= EEGINC model =============================
        x = self.temp_conv(x)
        x = self.bn(x)
        x = self.conv_depth(x)
        x = self.drop_depth(self.avgpool_depth(self.act_depth(self.bn_depth(x))))
        x = self.incept_temp(x)
        x = self.drop_temp(self.avgpool_temp(self.act_temp(self.bn_temp(x))))  # (batch, F1*D, 1, 15)

        # eegFatures = torch.squeeze(x, dim=2) # (batch, F1*D, 15)
        eegFatures = x

        # ============================= Decision Fusion model =============================
        eeg_out = self.liner_eeg(self.flatten_eeg(x))

        # ============================= MSA model =============================
        x = self.layerNorm(x)
        x = self.cnnMSA(x)
        # x, attention_cycle, attention, cycle_attn = self.cnnMSA(x, mode="test") # (batch, F1*D, 1, 15)

        # msaFatures = torch.squeeze(x, dim=2) # (batch, F1*D, 15)
        msaFatures = x

        # ============================= Feature Fusion model =============================
        fusionFeature = torch.cat((eegFatures, msaFatures), dim=1)

        # ============================= TCN model =============================
        x = torch.squeeze(fusionFeature, dim=2)  # (batch, F1*D, 15)
        x = self.tcn_block(x)
        x = x[:, :, -1]
        tcnFeature = x  # (batch, F1*D)

        # tcnFeature = torch.unsqueeze(tcnFeature, 2)
        # fusionFeature = torch.cat((tcnFeature, eegFatures), dim=2)
        # fusionFeature_out = self.liner_fusion(self.flatten_fusion(fusionFeature))


        return tcnFeature


# %%
###============================ Initialization parameters ============================###
channels = 22
samples = 1000


###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = EISATC_feature(eeg_chans=22, n_classes=4)
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    # print('attention_scores', attention_scores.shape)
    print('model', model)
    # summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))


if __name__ == "__main__":
    main()
