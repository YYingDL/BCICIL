
from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from models.utils import *
from scipy import signal

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Net_ln2(nn.Module):
    def __init__(self, n_feature, n_hidden, bottleneck_dim):
        super(Net_ln2, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.fc2 = nn.Linear(n_hidden, bottleneck_dim)
        self.fc2.apply(init_weights)
        self.ln2 = nn.LayerNorm(bottleneck_dim)

    def forward(self, x, pooling=True):
        x = x.reshape(x.size(0), -1)
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        if pooling:
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), 1, -1)
        return x


class EEGNet_feature(nn.Module):
    """
    EEGNet encoder for Motor Imagery (MI) and other EEG-based tasks.
    Also aliased as MI_EEGNet for compatibility.
    """

    def __init__(self,
                 Chans: int,
                 Samples: int,
                 feature_dim:int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate:  float,
                 norm_rate: float):
        super(EEGNet_feature, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.feature_dim = feature_dim
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                      out_features=self.feature_dim,
                      bias=True))

    def forward(self, x, pooling=True) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        output = self.block1(x)
        output = self.block2(output)
        # output = self.fc(output.reshape(output.size(0), -1))
        if pooling:
            output = output.reshape(output.size(0), -1)
        else:
            output = output.reshape(output.size(0), 1, -1)

        return output


class SSVEPFormer(nn.Module):
    def __init__(self, n_channels=9, dropout_rate=0.5, sample_length=280 * 2, feature_dims=40):
        super(SSVEPFormer, self).__init__()
        # 定义卷积层和池化层
        self.channel_combination = nn.Sequential(
            nn.Conv1d(in_channels=n_channels, out_channels=2 * n_channels, kernel_size=1),
            nn.LayerNorm([2 * n_channels, sample_length]),
            nn.GELU(),
            nn.Dropout(dropout_rate))

        self.subencoder_cnn1 = nn.Sequential(
            nn.LayerNorm(sample_length),
            nn.Conv1d(in_channels=2 * n_channels, out_channels=2 * n_channels, kernel_size=31, padding=15),
            nn.LayerNorm([2 * n_channels, sample_length]),
            nn.GELU(),
            nn.Dropout(dropout_rate))

        self.subencoder_mlp1 = nn.Sequential(
            nn.LayerNorm([2 * n_channels, sample_length]),
            nn.Linear(in_features=sample_length, out_features=sample_length),
            nn.GELU(),
            nn.Dropout(dropout_rate))

        self.subencoder_cnn2 = nn.Sequential(
            nn.LayerNorm([2 * n_channels, sample_length]),
            nn.Conv1d(in_channels=2 * n_channels, out_channels=2 * n_channels, kernel_size=31, padding=15),
            nn.LayerNorm([2 * n_channels, sample_length]),
            nn.GELU(),
            nn.Dropout(dropout_rate))

        self.subencoder_mlp2 = nn.Sequential(
            nn.LayerNorm([2 * n_channels, sample_length]),
            nn.Linear(in_features=sample_length, out_features=sample_length),
            nn.GELU(),
            nn.Dropout(dropout_rate))

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=2 * n_channels * sample_length, out_features=feature_dims),
            nn.LayerNorm(feature_dims),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x, pooling=True):
        x = x.permute(0, 2, 1)
        x = torch.fft.rfft(x, n=1250, dim=-1)
        real = torch.real(x[:, :, 40:320])
        imag = torch.imag(x[:, :, 40:320])
        x = torch.cat((real, imag), dim=2)
        # 前向传播
        x = self.channel_combination(x)
        x = x + self.subencoder_cnn1(x)
        x = x + self.subencoder_mlp1(x)
        x = x + self.subencoder_cnn2(x)
        x = x + self.subencoder_mlp2(x)

        x = self.mlp(x)
        return x


# class FBSSVEPFormer(nn.Module):
#     def __init__(self, subbands=3, n_channels=9, dropout_rate=0.5, sample_length=560, feature_dims=20):
#         super(FBSSVEPFormer, self).__init__()
#         # 定义卷积层和池化层
#         self.layer = nn.ModuleList([SSVEPFormer(n_channels=n_channels, dropout_rate=dropout_rate,
#                                                 sample_length=sample_length, feature_dims = feature_dims) for _ in
#                                     range(subbands)])
#         self.convfuse = nn.Conv1d(in_channels=subbands, out_channels=1, kernel_size=1)
#
#     def forward(self, x, pooling=True):
#         x = x.permute(0, 1, 3, 2)
#         # 前向传播
#         output_layers = []
#         for idx, layer in enumerate(self.layer):
#             output_layers.append(layer(x[:, idx, :, :]).unsqueeze(1))
#         x = torch.cat(output_layers, dim=1)
#
#         x = self.convfuse(x)
#         return x.squeeze(1)


# def filterbank(X, num_subbands=3):
#     """
#     Suggested filterbank function for benchmark dataset
#     """
#
#
#     srate = 250
#     X = np.transpose(X,(0, 2, 1))
#     filterbank_X = np.zeros((X.shape[0], num_subbands, X.shape[-2], X.shape[-1]))
#     for i in range(X.shape[0]):
#         x_tmp = X[i]
#         for k in range(1, num_subbands + 1, 1):
#             Wp = [(8 * k) / (srate / 2), 90 / (srate / 2)]
#             Ws = [(8 * k - 2) / (srate / 2), 100 / (srate / 2)]
#             gstop = 40
#             while gstop >= 20:
#                 try:
#                     N, Wn = signal.cheb1ord(Wp, Ws, 3, gstop)
#                     bpB, bpA = signal.cheby1(N, 0.5, Wn, btype='bandpass')
#                     filterbank_X[i, k - 1, :, :] = signal.filtfilt(bpB, bpA, x_tmp, axis=1, padtype='odd',
#                                                                    padlen=3 * (max(len(bpB), len(bpA)) - 1))
#                     break
#                 except:
#                     gstop -= 1
#     return torch.tensor(filterbank_X).to(torch.float32)


class DER_Net(nn.Module):
    """
    DER (Dynamically Expandable Representation) network wrapper.
    Wraps an existing encoder (e.g., EEGNet, FBSSVEPFormer) to enable dynamic branch expansion.

    Key features:
    1. Dynamic network expansion - adds new branch for each task
    2. Feature concatenation - concatenates features from all branches
    3. Old branch freezing - freezes historical branches during training

    Args:
        encoder_class: The encoder class to wrap (e.g., EEGNet_feature, FBSSVEPFormer)
        encoder_args: Dictionary of arguments for the encoder
        feature_dim: The output feature dimension of a single branch
    """
    def __init__(self, encoder_class, encoder_args, feature_dim):
        super(DER_Net, self).__init__()
        self.encoder_class = encoder_class
        self.encoder_args = encoder_args
        self.feature_dim = feature_dim
        self.branches = nn.ModuleList()
        # Initialize with the first branch
        first_branch = encoder_class(**encoder_args)
        self.branches.append(first_branch)

        # Infer actual feature dimension by running a forward pass with a dummy input
        # This is needed because some encoders (like EEGNet_feature) have commented-out FC layers
        self._actual_feature_dim = self._infer_feature_dim()

    def _infer_feature_dim(self):
        """
        Infer the actual output feature dimension by running a forward pass.
        Returns the dimension of the output when pooling=True.
        """
        try:
            # Create a dummy input based on encoder type
            if 'Chans' in self.encoder_args and 'Samples' in self.encoder_args:
                # EEGNet_feature: (batch, Samples, Chans)
                dummy_input = torch.randn(1, self.encoder_args['Samples'], self.encoder_args['Chans'])
            elif 'n_channels' in self.encoder_args and 'sample_length' in self.encoder_args:
                # FBSSVEPFormer/SSVEPFormer: (batch, subbands, sample_length, n_channels) or similar
                dummy_input = torch.randn(1, 3, self.encoder_args['sample_length'], self.encoder_args['n_channels'])
            elif 'n_feature' in self.encoder_args:
                # Net_ln2: (batch, n_feature)
                dummy_input = torch.randn(1, self.encoder_args['n_feature'])
            else:
                # Fallback: use feature_dim
                return self.feature_dim

            # Run forward pass
            self.branches[0].eval()
            with torch.no_grad():
                output = self.branches[0](dummy_input, pooling=True)
            return output.shape[-1]
        except Exception as e:
            print(f'Warning: Could not infer feature dim ({e}), using provided feature_dim: {self.feature_dim}')
            return self.feature_dim

    def add_branch(self):
        """
        Add a new branch by copying the last branch's weights.
        This enables knowledge transfer from previous task.
        """
        new_branch = self.encoder_class(**self.encoder_args)
        # Load weights from the last branch for knowledge transfer
        if len(self.branches) > 0:
            new_branch.load_state_dict(self.branches[-1].state_dict())
        # Move new branch to the same device as existing branches
        new_branch.to(self.branches[0].parameters().__iter__().__next__().device)
        self.branches.append(new_branch)

    def freeze_branches(self, exclude_last=True):
        """
        Freeze old branches to prevent catastrophic forgetting.

        Args:
            exclude_last: If True, only freeze old branches and keep the last branch trainable
        """
        end_idx = len(self.branches) - 1 if exclude_last else len(self.branches)
        for i in range(end_idx):
            for p in self.branches[i].parameters():
                p.requires_grad = False

    def forward(self, x, pooling=True):
        """
        Forward pass through all branches and concatenate features.
        During training, only the last branch is in train mode, old branches are in eval mode.
        During evaluation, all branches are in eval mode.

        Args:
            x: Input tensor
            pooling: If True, return pooled features; if False, return feature maps

        Returns:
            Concatenated features from all branches
        """
        if self.training and len(self.branches) > 1:
            # During training with multiple branches: only last branch in train mode
            features = []
            for i, branch in enumerate(self.branches):
                if i == len(self.branches) - 1:
                    branch.train()
                else:
                    branch.eval()
                features.append(branch(x, pooling=pooling))
        else:
            # Single branch or evaluation mode: all branches in same mode
            features = [branch(x, pooling=pooling) for branch in self.branches]
        return torch.cat(features, dim=1)

    @property
    def out_features(self):
        """Return the total output feature dimension (branch_count * actual_feature_dim)."""
        return self._actual_feature_dim * len(self.branches)

    @property
    def n_branches(self):
        """Return the current number of branches."""
        return len(self.branches)

    def get_single_branch_feature_dim(self):
        """
        Get the feature dimension of a single branch.
        Useful for creating auxiliary classifiers.
        """
        return self._actual_feature_dim
