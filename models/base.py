# -*- coding: UTF-8 -*-
import math
import torch.nn as nn
from models.classifier import SingleHead, CosineLinear, SplitCosineLinear, IL2AHead
from utils.setup_elements import input_size_match, n_classes, get_num_classes,n_classes_init_task
from models.encoders import Net_ln2, EEGNet_feature, SSVEPFormer, DER_Net
from models.utils import TransposedInstanceNorm1d
from models.ADFCNN import ADFCNN
from models.IFNet import IFNet
import torch


class SingleHeadModel(nn.Module):
    def __init__(self, encoder, head, input_channels, feature_dims, n_layers, seq_len, n_base_nodes, norm, input_norm, dropout,
                 der_dynamic=False, der_encoder_class=None, der_encoder_args=None):
        super(SingleHeadModel, self).__init__()

        if input_norm == 'LN':
            self.input_norm = nn.LayerNorm(input_channels, elementwise_affine=False)  # Without learnable transform
        elif input_norm == 'IN':
            self.input_norm = TransposedInstanceNorm1d(input_channels, affine=False)  # Not perform well on GrabMyo
        else:
            self.input_norm = None

        # Store encoder class and args for DER dynamic expansion
        self.encoder_class = None
        self.encoder_args = None
        self.der_dynamic = der_dynamic

        if  encoder == 'Emotion_Net_ln2':
            base_encoder = Net_ln2(input_channels*seq_len, feature_dims, feature_dims)
            if der_dynamic:
                self.encoder_class = Net_ln2
                self.encoder_args = {'n_feature': input_channels*seq_len, 'n_hidden': feature_dims, 'bottleneck_dim': feature_dims}

        elif encoder == 'MI_EEGNet':
            base_encoder = EEGNet_feature(Chans=input_channels, Samples=seq_len, feature_dim= feature_dims,
                        kernLenght=int(250 // 2), F1=4, D=2, F2=8, dropoutRate=0.25, norm_rate=0.5)
            if der_dynamic:
                self.encoder_class = EEGNet_feature
                self.encoder_args = {'Chans': input_channels, 'Samples': seq_len, 'feature_dim': feature_dims,
                                     'kernLenght': int(250 // 2), 'F1': 4, 'D': 2, 'F2': 8, 'dropoutRate': 0.25, 'norm_rate': 0.5}

        elif encoder == 'MI_ADFCNN':
            base_encoder = ADFCNN(num_channels=input_channels, sampling_rate=seq_len)
            if der_dynamic:
                self.encoder_class = ADFCNN
                self.encoder_args = {'num_channels': input_channels, 'sampling_rate': seq_len}
        elif encoder == 'MI_EISATC':
            from models.EISATC import EISATC_feature
            base_encoder = EISATC_feature(eeg_chans=input_channels, device='cuda:0')
            if der_dynamic:
                self.encoder_class = EISATC_feature
                self.encoder_args = {'eeg_chans': input_channels, 'n_classes': 4, 'device': 'cuda:0'}
        elif encoder == 'MI_IFNet':
            base_encoder = IFNet(in_planes=input_channels, out_planes=64, kernel_size=63, radix=2, patch_size=125)
            if der_dynamic:
                self.encoder_class = IFNet
                self.encoder_args = {'in_planes': input_channels, 'out_planes': 64, 'kernel_size': 63, 'radix': 2, 'patch_size': 125}
        elif encoder == 'SSVEPFormer':
            base_encoder = SSVEPFormer(n_channels=input_channels, feature_dims=feature_dims)
            if der_dynamic:
                self.encoder_class = SSVEPFormer
                self.encoder_args = {'n_channels': input_channels, 'feature_dims': feature_dims, 'sample_length': seq_len}
        else:
            raise ValueError("Backbone must be CNN or TST")

        # Wrap with DER_Net if dynamic expansion is enabled
        if der_dynamic and self.encoder_class is not None:
            self.encoder = DER_Net(self.encoder_class, self.encoder_args, feature_dim=feature_dims)
            # For DER, use the actual inferred feature dimension for the head
            head_in_features = self.encoder._actual_feature_dim
        else:
            self.encoder = base_encoder
            head_in_features = feature_dims

        if head == 'Linear':
            self.head = SingleHead(in_features=head_in_features, out_features=n_base_nodes)
        elif head in ['CosineLinear', 'SplitCosineLinear']:
            self.head = CosineLinear(in_features=head_in_features, out_features=n_base_nodes)
        elif head == 'IL2AHead':
            # IL2A specialized head with virtual class support
            self.head = IL2AHead(in_features=head_in_features, n_real_classes=n_base_nodes, use_virtual=True)
        else:
            raise ValueError("Wrong head type")
        self.head_type = head

    @property
    def out_features(self):
        """Return the current output feature dimension of the encoder."""
        if hasattr(self.encoder, 'out_features'):
            return self.encoder.out_features
        elif hasattr(self.encoder, 'feature_dim'):
            return self.encoder.feature_dim
        else:
            # Fallback: try to infer from head
            return self.head.in_features

    def feature_map(self, x):
        """
        Return the feature map produced by encoder, (N, D, L)
        """
        if self.input_norm:
            x = self.input_norm(x)
        # For DER_Net, get features from last branch only
        if self.der_dynamic and hasattr(self.encoder, 'branches'):
            feature_map = self.encoder.branches[-1](x, pooling=False)
        else:
            feature_map = self.encoder(x, pooling=False)
        return feature_map

    def feature(self, x):
        """
        Return the feature vector after GAP, (N, D)
        For DER_Net, returns concatenated features from all branches.
        """
        if self.input_norm:
            x = self.input_norm(x)
        feature = self.encoder(x, pooling=True)
        return feature

    def forward(self, x):
        if self.input_norm:
            x = self.input_norm(x)
        x = self.encoder(x)
        x = self.head(x)
        return x

    def update_head_for_der_branch(self):
        """
        Update the classification head when a new DER branch is added.
        The head's input features need to expand to accommodate concatenated features.
        Preserves old class weights while expanding input dimension.
        """
        if not self.der_dynamic:
            return

        # Get the new total feature dimension
        new_in_features = self.encoder.out_features
        old_in_features = self.head.in_features

        # Create new head with updated input features and same output features
        if self.head_type == 'Linear':
            new_head = SingleHead(in_features=new_in_features, out_features=self.head.out_features)
            # Copy old head weights to new head (for existing features)
            with torch.no_grad():
                new_head.fc.weight[:, :old_in_features] = self.head.fc.weight
                new_head.fc.bias = self.head.fc.bias
        elif self.head_type == 'CosineLinear':
            new_head = CosineLinear(in_features=new_in_features, out_features=self.head.out_features)
            # Copy old head weights to new head
            with torch.no_grad():
                new_head.weight[:, :old_in_features] = self.head.weight
        else:
            raise ValueError("DER dynamic expansion only supports Linear or CosineLinear head")

        # Move to device
        new_head.to(next(self.parameters()).device)
        self.head = new_head

    def update_head_for_der_branch_and_classes(self, n_new_classes, task_now):
        """
        Update the classification head when a new DER branch is added AND new classes arrive.
        Handles both input feature expansion (DER branches) and output class expansion.

        Args:
            n_new_classes: Number of new classes in current task
            task_now: Current task index (1-based)
        """
        if not self.der_dynamic:
            return

        # Get the new total feature dimension from concatenated branches
        new_in_features = self.encoder.out_features
        old_in_features = self.head.in_features
        old_out_features = self.head.out_features
        new_out_features = old_out_features + n_new_classes

        if self.head_type == 'Linear':
            # Create new head with expanded input and output dimensions
            new_head = SingleHead(in_features=new_in_features, out_features=new_out_features)
            # Copy old weights for existing features and classes
            with torch.no_grad():
                new_head.fc.weight[:old_out_features, :old_in_features] = self.head.fc.weight
                new_head.fc.bias[:old_out_features] = self.head.fc.bias
                # New feature dimensions for old classes: keep random initialization (consistent with DER)
                # This allows old classes to gradually adapt to new features during training
                # Initialize new class weights with xavier
                nn.init.xavier_uniform_(new_head.fc.weight[old_out_features:])
                nn.init.zeros_(new_head.fc.bias[old_out_features:])
        elif self.head_type == 'CosineLinear':
            # Create new head with expanded input and output dimensions
            new_head = CosineLinear(in_features=new_in_features, out_features=new_out_features)
            # Copy old weights
            with torch.no_grad():
                new_head.weight[:old_out_features, :old_in_features] = self.head.weight
                # New feature dimensions for old classes: keep random initialization (consistent with DER)
                # Initialize new class weights
                stdv = 1. / math.sqrt(new_in_features)
                new_head.weight[old_out_features:, :old_in_features].uniform_(-stdv, stdv)
                # Initialize new class weights for new features
                new_head.weight[old_out_features:, old_in_features:].uniform_(-stdv, stdv)
        elif self.head_type == 'SplitCosineLinear':
            # For SplitCosineLinear, we need to handle both old class expansion and new classes
            if task_now == 1:
                # Task 1: First expansion - convert from single head to SplitCosineLinear
                # fc1 = old classes, fc2 = new classes
                new_head = SplitCosineLinear(in_features=new_in_features,
                                              out_features1=old_out_features,
                                              out_features2=n_new_classes)
                # Copy old weights to fc1 (need to expand column dimension too)
                with torch.no_grad():
                    # Expand old weights from [old_out, old_in] to [old_out, new_in]
                    new_head.fc1.weight[:, :old_in_features] = self.head.weight
                    # New feature dimensions for old classes: keep random initialization (consistent with DER)
                    # Copy sigma
                    new_head.sigma.data = self.head.sigma.data
                # Initialize new class weights (fc2)
                stdv = 1. / math.sqrt(new_in_features)
                with torch.no_grad():
                    new_head.fc2.weight.uniform_(-stdv, stdv)
            else:
                # Task 2+: fc1 has old classes, fc2 has classes from previous task
                # We need to expand fc2 with new classes AND expand input features
                # Strategy: Create new SplitCosineLinear with all old classes in fc1, new classes in fc2
                out_features1 = old_out_features  # All previously learned classes
                out_features2 = n_new_classes  # New classes only

                new_head = SplitCosineLinear(in_features=new_in_features,
                                              out_features1=out_features1,
                                              out_features2=out_features2)
                # Copy old fc1 weights with expanded input features
                with torch.no_grad():
                    new_head.fc1.weight[:, :old_in_features] = self.head.fc1.weight
                    # New feature dimensions for old classes: keep random initialization (consistent with DER)

                    # Copy old fc2 weights for new classes
                    new_head.fc2.weight[:self.head.fc2.out_features, :old_in_features] = self.head.fc2.weight
                    # New feature dimensions for fc2 classes: keep random initialization (consistent with DER)
                    # Initialize new class weights in fc2
                    stdv = 1. / math.sqrt(new_in_features)
                    new_head.fc2.weight[self.head.fc2.out_features:, :old_in_features].uniform_(-stdv, stdv)
                    new_head.fc2.weight[self.head.fc2.out_features:, old_in_features:].uniform_(-stdv, stdv)

                    new_head.sigma.data = self.head.sigma.data
        else:
            raise ValueError("DER dynamic expansion only supports Linear, CosineLinear, or SplitCosineLinear head")

        # Move to device
        new_head.to(next(self.parameters()).device)
        self.head = new_head

    def update_head(self, n_new, task_now=None):
        if self.head_type == 'SplitCosineLinear':
            assert task_now is not None
            assert task_now > 0
            if task_now == 1:
                in_features, out_features = self.head.in_features, self.head.out_features
                new_head = SplitCosineLinear(in_features, out_features, n_new)
                new_head.fc1.weight.data = self.head.weight.data
                new_head.sigma.data = self.head.sigma.data
                self.head = new_head
            else:
                in_features = self.head.in_features
                out_features1 = self.head.fc1.out_features
                out_features2 = self.head.fc2.out_features
                new_head = SplitCosineLinear(in_features, out_features1 + out_features2, n_new)
                new_head.fc1.weight.data[:out_features1] = self.head.fc1.weight.data
                new_head.fc1.weight.data[out_features1:] = self.head.fc2.weight.data
                new_head.sigma.data = self.head.sigma.data
                self.head = new_head
        elif self.head_type == 'IL2AHead':
            # IL2A head handles virtual classes automatically
            self.head.increase_neurons(n_new)
        else:
            self.head.increase_neurons(n_new)


def setup_model(args):
    Model = SingleHeadModel
    data = args.data
    n_offline_base_nodes = n_classes[data]

    # Enable DER dynamic expansion if using DER agent with der_dynamic flag
    der_dynamic = getattr(args, 'der_dynamic', False) and args.agent == 'DER'

    return Model(encoder=args.encoder,
                 head=args.head,
                 input_channels=input_size_match[data][-1],
                 feature_dims=args.feature_dim,
                 n_layers=args.n_layers,
                 seq_len=input_size_match[data][-2],
                 n_base_nodes=n_offline_base_nodes if args.agent == 'Offline' else n_classes_init_task[data],
                 norm=args.norm,
                 input_norm=args.input_norm,
                 dropout=args.dropout,
                 der_dynamic=der_dynamic,
                 ).to(args.device)
