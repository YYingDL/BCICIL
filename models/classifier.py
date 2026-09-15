# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import copy

import math
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class SingleHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(SingleHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features=in_features, out_features=self.out_features)
        # torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)

    def increase_neurons(self, n_new):
        n_old_classes = self.out_features
        self.out_features += n_new
        old_head = copy.deepcopy(self.fc)

        self.fc = nn.Linear(in_features=self.in_features, out_features=self.out_features)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        with torch.no_grad():
            self.fc.weight[:n_old_classes] = old_head.weight
            self.fc.bias[:n_old_classes] = old_head.bias

    def add_virtual_classes(self, n_virtual):
        """
        Add virtual classes to the head for IL2A.
        Virtual classes are used for mixup-based augmentation.
        """
        n_old_classes = self.out_features
        self.out_features += n_virtual
        old_head = copy.deepcopy(self.fc)

        self.fc = nn.Linear(in_features=self.in_features, out_features=self.out_features)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        with torch.no_grad():
            # Copy old class weights
            self.fc.weight[:n_old_classes] = old_head.weight
            self.fc.bias[:n_old_classes] = old_head.bias


class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def increase_neurons(self, n_new):
        n_old_classes = self.out_features
        self.out_features += n_new
        old_weight = copy.deepcopy(self.weight)

        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.reset_parameters()

        with torch.no_grad():
            self.weight.data[:n_old_classes] = old_weight.data

    def add_virtual_classes(self, n_virtual):
        """
        Add virtual classes to the head for IL2A.
        Virtual classes are used for mixup-based augmentation.
        """
        n_old_classes = self.out_features
        self.out_features += n_virtual
        old_weight = copy.deepcopy(self.weight)

        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.reset_parameters()

        with torch.no_grad():
            self.weight.data[:n_old_classes] = old_weight.data

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class SplitCosineLinear(Module):
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1)
        if self.sigma is not None:
            out = self.sigma * out
        return out

    def increase_neurons(self, n_new):
        """
        Add n_new classes to fc2 (the expanding part).
        fc1 remains unchanged (old classes).
        """
        old_out_features2 = self.fc2.out_features
        new_out_features2 = old_out_features2 + n_new

        # Create new fc2 with more classes
        new_fc2 = CosineLinear(self.in_features, new_out_features2, False)

        # Copy old fc2 weights to new fc2
        with torch.no_grad():
            new_fc2.weight.data[:old_out_features2] = self.fc2.weight.data
            if self.fc2.sigma is not None:
                new_fc2.sigma.data = self.fc2.sigma.data

        self.fc2 = new_fc2
        self.out_features = self.fc1.out_features + self.fc2.out_features

    def add_virtual_classes(self, n_virtual):
        """
        Add virtual classes to the head for IL2A.
        Virtual classes are appended to fc2.
        """
        old_out_features2 = self.fc2.out_features
        new_out_features2 = old_out_features2 + n_virtual

        # Create new fc2 with more classes
        new_fc2 = CosineLinear(self.in_features, new_out_features2, False)

        # Copy old fc2 weights to new fc2
        with torch.no_grad():
            new_fc2.weight.data[:old_out_features2] = self.fc2.weight.data
            if self.fc2.sigma is not None:
                new_fc2.sigma.data = self.fc2.sigma.data

        self.fc2 = new_fc2
        self.out_features = self.fc1.out_features + self.fc2.out_features


class IL2AHead(nn.Module):
    """
    Specialized head for IL2A with virtual class support.

    Structure:
    - Real classes: [0, n_real_classes)
    - Virtual classes: [n_real_classes, n_real_classes + n_virtual_classes)

    Virtual classes are generated for pairs of real classes (i, j) where i < j.
    Number of virtual classes = C(n_real, 2) = n_real * (n_real - 1) / 2

    The head supports:
    1. Dynamic expansion of real classes
    2. Dynamic expansion of virtual classes when real classes are added
    3. Evaluation mode: only use real class outputs
    """
    def __init__(self, in_features, n_real_classes, use_virtual=True):
        super(IL2AHead, self).__init__()
        self.in_features = in_features
        self.n_real_classes = n_real_classes
        self.use_virtual = use_virtual

        # Calculate initial virtual class count
        self.n_virtual_classes = n_real_classes * (n_real_classes - 1) // 2 if use_virtual else 0
        self.total_out_features = self.n_real_classes + self.n_virtual_classes

        # Main classifier
        self.fc = nn.Linear(in_features, self.total_out_features, bias=True)

        # Virtual class index mapping: (real_i, real_j) -> virtual_class_index
        self._update_virtual_index_map()

    def _update_virtual_index_map(self):
        """Build mapping from (real_i, real_j) pair to virtual class index."""
        self.virtual_index_map = {}  # (i, j) -> virtual_idx where i < j
        virtual_idx = 0
        for i in range(self.n_real_classes):
            for j in range(i + 1, self.n_real_classes):
                self.virtual_index_map[(i, j)] = virtual_idx
                virtual_idx += 1

    def get_virtual_class_index(self, class_i, class_j):
        """
        Get virtual class index for a pair of real classes.
        Assumes class_i < class_j. If not, swaps them.
        """
        if class_i > class_j:
            class_i, class_j = class_j, class_i
        return self.virtual_index_map.get((class_i, class_j), None)

    def forward(self, x, use_virtual=False):
        """
        Forward pass.

        Args:
            x: Input features
            use_virtual: If True, return all outputs (real + virtual).
                        If False, return only real class outputs (for evaluation).
        """
        outputs = self.fc(x)
        if use_virtual:
            return outputs
        else:
            return outputs[:, :self.n_real_classes]

    def increase_neurons(self, n_new_real):
        """
        Add n_new_real real classes and update virtual classes.

        When adding new real classes:
        1. Old virtual classes become invalid (need reindexing)
        2. New virtual classes are created for all pairs involving new real classes
        """
        old_n_real = self.n_real_classes
        new_n_real = old_n_real + n_new_real
        new_n_virtual = new_n_real * (new_n_real - 1) // 2

        # Save old weights
        old_weight = self.fc.weight.data.clone()
        old_bias = self.fc.bias.data.clone()

        # Create new classifier
        new_total = new_n_real + new_n_virtual
        self.fc = nn.Linear(self.in_features, new_total, bias=True)

        # Initialize with xavier
        nn.init.xavier_uniform_(self.fc.weight.data)
        nn.init.zeros_(self.fc.bias.data)

        with torch.no_grad():
            # Copy old real class weights
            self.fc.weight.data[:old_n_real, :] = old_weight.data[:old_n_real, :]
            self.fc.bias.data[:old_n_real] = old_bias.data[:old_n_real]

            # Copy old virtual class weights (need reindexing)
            # Old virtual classes map to new virtual class indices
            for i in range(old_n_real):
                for j in range(i + 1, old_n_real):
                    old_virtual_idx = self._get_old_virtual_index(i, j, old_n_real)
                    new_virtual_idx = self.n_real_classes + self.virtual_index_map.get((i, j), 0)
                    if old_virtual_idx is not None and (i, j) in self.virtual_index_map:
                        new_virtual_idx = self.n_real_classes + self.virtual_index_map[(i, j)]
                        self.fc.weight.data[new_virtual_idx, :] = old_weight.data[old_n_real + old_virtual_idx, :]
                        self.fc.bias.data[new_virtual_idx] = old_bias.data[old_n_real + old_virtual_idx]

        # Update state
        self.n_real_classes = new_n_real
        self.n_virtual_classes = new_n_virtual
        self.total_out_features = new_n_real + new_n_virtual
        self._update_virtual_index_map()

    def _get_old_virtual_index(self, i, j, old_n_real):
        """Get virtual index in old mapping (before expansion)."""
        if i > j:
            i, j = j, i
        if i >= old_n_real or j >= old_n_real:
            return None
        # Calculate index based on old mapping
        idx = 0
        for a in range(i):
            idx += (old_n_real - 1 - a)
        idx += (j - i - 1)
        return idx

    def add_virtual_classes(self, n_virtual):
        """Not used - virtual classes are automatically managed."""
        pass
