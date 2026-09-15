# -*- coding: utf-8 -*-
"""
IL2A (Class-Incremental Learning via Dual Augmentation) for TSCIL framework.
OPTIMIZED VERSION - Vectorized operations for speed.

NeurIPS 2021: "Class-Incremental Learning via Dual Augmentation"
Fei Zhu, Zhen Cheng, Xu-Yao Zhang, Cheng-Lin Liu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy


class IL2A(BaseLearner):
    """
    OPTIMIZED IL2A Agent with vectorized operations.
    Key optimizations:
    1. Virtual class matrix for O(1) label lookup (no CPU-GPU transfer)
    2. Covariance tensor for batched lookup
    3. Label-to-prototype index cache
    """

    def __init__(self, model, args):
        super(IL2A, self).__init__(model, args)

        # IL2A hyperparameters
        self.seman_weight = args.seman_weight
        self.kd_weight = args.il2a_kd_weight
        self.temperature = args.il2a_temperature
        self.mix_alpha = args.mix_alpha
        self.mix_times = args.mix_times
        self.aug_ratio = args.aug_ratio
        self.kd_temperature = args.il2a_kd_temperature

        # Class tracking
        self.numclass = 0
        self.task_size = 0
        self.n_virtual = 0

        # Prototype and covariance storage (as tensors for speed)
        self.prototype = []
        self.cov = []
        self.class_label = []

        # Optimized: Tensor versions for GPU operations
        self.cov_tensor = None  # (n_classes, feature_dim, feature_dim)
        self.label_to_proto_idx = {}  # label -> index in prototype list

        # Old model for KD
        self.old_model = None

        # Virtual class matrix (n_classes x n_classes) for O(1) lookup
        self.virtual_class_matrix = None

        self.feature_dim = args.feature_dim

        print(f'[IL2A-Optimized] Initialized with seman_weight={self.seman_weight}, kd_weight={self.kd_weight}')

    def _build_virtual_class_matrix(self, n_real):
        """Build virtual class lookup matrix (GPU tensor)."""
        device = self.device
        matrix = -torch.ones(n_real, n_real, dtype=torch.long, device=device)

        idx = 0
        for i in range(n_real):
            for j in range(i + 1, n_real):
                # Virtual class index is offset by n_real
                virt_label = n_real + idx
                matrix[i, j] = virt_label
                matrix[j, i] = virt_label  # Symmetric
                idx += 1

        return matrix

    def _get_virtual_labels_batch(self, y_a, y_b):
        """
        Vectorized virtual label lookup.
        y_a, y_b: tensors of same shape with class indices
        Returns: tensor of virtual labels
        """
        if self.virtual_class_matrix is None:
            return y_a  # Fallback

        return self.virtual_class_matrix[y_a, y_b]

    def before_task(self, y_train):
        """Prepare for learning a new task"""
        # Store old model for KD
        if self.model is not None and self.numclass > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            print(f'[IL2A] Stored old model for KD at Task {self.task_now + 1}')

        super().before_task(y_train)

        # Update class counters
        n_new_classes = len(self.classes_in_task)
        self.task_size = n_new_classes
        self.numclass += n_new_classes
        self.n_virtual = self.numclass * (self.numclass - 1) // 2

        # Rebuild virtual class matrix (OPTIMIZED)
        self.virtual_class_matrix = self._build_virtual_class_matrix(self.numclass)

        # Rebuild label-to-prototype index cache
        self.label_to_proto_idx = {label: idx for idx, label in enumerate(self.class_label)}

        print(f'[IL2A] Task {self.task_now}: numclass={self.numclass}, n_virtual={self.n_virtual}')

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with optimized operations."""
        total = 0
        correct = 0
        epoch_loss = 0
        epoch_loss_cls = 0
        epoch_loss_seman = 0
        epoch_loss_kd_feat = 0
        epoch_loss_kd_logits = 0

        self.model.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)

            self.optimizer.zero_grad()

            # ClassAug: OPTIMIZED - fully vectorized
            x_aug, y_aug = self._class_aug_fast(x, y)

            # Forward pass
            features = self.model.feature(x_aug)
            outputs = self.model.head(features, use_virtual=True) if hasattr(self.model.head, 'use_virtual') else self.model.head(features)

            # Classification loss
            loss_cls = self.criterion(outputs, y_aug)

            # Initialize auxiliary losses
            loss_seman = torch.tensor(0.0, device=self.device)
            loss_kd_feat = torch.tensor(0.0, device=self.device)
            loss_kd_logits = torch.tensor(0.0, device=self.device)

            # Compute KD and SemanAug only if we have old model and prototypes
            if self.old_model is not None and len(self.prototype) > 0:
                # Feature-level KD (OPTIMIZED: single forward pass)
                with torch.no_grad():
                    old_outputs, old_features = self._old_model_forward(x_aug)
                loss_kd_feat = F.mse_loss(features, old_features)

                # Logits-level KD
                loss_kd_logits = self._compute_logits_kd(outputs, old_outputs)

                # Prototype-based Semantic Augmentation (OPTIMIZED)
                loss_seman = self._compute_seman_aug_loss_fast()

            # Total loss
            total_loss = loss_cls + self.seman_weight * loss_seman + self.kd_weight * (loss_kd_feat + loss_kd_logits)
            total_loss.backward()
            self.optimizer_step(epoch=epoch)

            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_loss_cls += loss_cls.item()
            epoch_loss_seman += loss_seman.item()
            epoch_loss_kd_feat += loss_kd_feat.item()
            epoch_loss_kd_logits += loss_kd_logits.item()

            # Accuracy (only on real classes)
            outputs_real = outputs[:, :self.numclass]
            prediction = torch.argmax(outputs_real, dim=1)
            correct += prediction.eq(y_aug).sum().item()
            total += y_aug.size(0)

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)
        epoch_loss_cls /= (batch_id + 1)
        epoch_loss_seman /= (batch_id + 1)
        epoch_loss_kd_feat /= (batch_id + 1)
        epoch_loss_kd_logits /= (batch_id + 1)

        if self.verbose and epoch == 0:
            print(f'  Task {self.task_now}: loss_cls={epoch_loss_cls:.4f}, '
                  f'loss_seman={epoch_loss_seman:.4f}, '
                  f'loss_kd_feat={epoch_loss_kd_feat:.4f}, loss_kd_logits={epoch_loss_kd_logits:.4f}')

        self._epoch_losses = (epoch_loss, epoch_loss_cls, epoch_loss_seman,
                              epoch_loss_kd_feat, epoch_loss_kd_logits)
        return epoch_loss, epoch_acc

    def _old_model_forward(self, x):
        """Get both outputs and features from old model in single pass."""
        with torch.no_grad():
            features = self.old_model.feature(x)
            outputs = self.old_model.head(features)
        return outputs, features

    def _class_aug_fast(self, x, y):
        """
        OPTIMIZED ClassAug: Fully vectorized, no CPU-GPU transfer.
        Uses precomputed virtual class matrix for O(1) lookup.
        """
        if self.mix_times == 0:
            return x, y

        batch_size = x.size(0)
        device = x.device

        all_mixed = []
        all_targets = []

        for _ in range(self.mix_times):
            # Shuffle indices
            index = torch.randperm(batch_size, device=device)
            y_shuffled = y[index]

            # Find pairs with different labels (vectorized)
            diff_mask = (y != y_shuffled)
            if not diff_mask.any():
                continue

            valid_indices = torch.nonzero(diff_mask, as_tuple=True)[0]
            n_valid = len(valid_indices)

            # Sample lambdas (vectorized)
            lambdas = torch.from_numpy(np.random.beta(self.mix_alpha, self.mix_alpha, size=n_valid)).float().to(device)

            # Clip lambdas
            clip_mask = (lambdas < 0.4) | (lambdas > 0.6)
            lambdas[clip_mask] = 0.5

            # Vectorized mixup
            lambdas_view = lambdas.view(-1, *([1] * (x.dim() - 1)))
            mixed = lambdas_view * x[valid_indices] + (1 - lambdas_view) * x[index[valid_indices]]

            # OPTIMIZED: Virtual label lookup via matrix indexing (no CPU transfer!)
            y_a = y[valid_indices]
            y_b = y_shuffled[valid_indices]
            virtual_labels = self._get_virtual_labels_batch(y_a, y_b)

            all_mixed.append(mixed)
            all_targets.append(virtual_labels)

        if len(all_mixed) > 0:
            x_mixed = torch.cat(all_mixed, dim=0)
            y_mixed = torch.cat(all_targets, dim=0)
            x_aug = torch.cat([x, x_mixed], dim=0)
            y_aug = torch.cat([y, y_mixed], dim=0)
            return x_aug, y_aug
        else:
            return x, y

    def _compute_logits_kd(self, outputs, old_outputs):
        """Logits-level KD using KL divergence on real classes only."""
        if len(self.class_label) == 0:
            return torch.tensor(0.0, device=self.device)

        # Use min of both output dimensions to handle class expansion
        n_old_classes = old_outputs.size(1)
        n_classes_kd = min(self.numclass, n_old_classes)

        outputs_real = outputs[:, :n_classes_kd]
        old_outputs_real = old_outputs[:, :n_classes_kd]

        loss = F.kl_div(
            F.log_softmax(outputs_real / self.kd_temperature, dim=1),
            F.softmax(old_outputs_real / self.kd_temperature, dim=1),
            reduction='batchmean'
        ) * (self.kd_temperature ** 2)
        return loss

    def _compute_seman_aug_loss_fast(self):
        """
        OPTIMIZED Semantic augmentation using tensor operations.
        - Covariance stored as tensor (no numpy conversion)
        - Batched lookup via index_select
        """
        if len(self.prototype) == 0 or self.cov_tensor is None:
            return torch.tensor(0.0, device=self.device)

        # Sample prototypes
        proto_aug, proto_aug_label = self._sample_from_prototypes_fast()
        if len(proto_aug) == 0:
            return torch.tensor(0.0, device=self.device)

        # Get classifier weights
        weight_m = self._get_classifier_weights()
        if weight_m is None:
            return torch.tensor(0.0, device=self.device)

        weight_m = weight_m[:self.numclass, :]
        C = self.numclass
        N = proto_aug.size(0)
        A = proto_aug.size(1)

        # Compute sigma2 (OPTIMIZED: batched tensor operations)
        sigma2 = self._compute_sigma2_fast(proto_aug_label, weight_m, C, A)

        # Get logits
        proto_outputs = self.model.head(proto_aug)
        logits = proto_outputs[:, :C]

        # Augment and compute loss
        aug_logits = logits + 0.5 * sigma2
        loss = F.cross_entropy(aug_logits / self.temperature, proto_aug_label)
        return loss

    def _sample_from_prototypes_fast(self, batch_size=None):
        """Fast prototype sampling using tensor operations."""
        if batch_size is None:
            batch_size = self.batch_size

        n_protos = len(self.prototype)
        if n_protos == 0:
            return torch.tensor([]), torch.tensor([])

        if batch_size >= n_protos:
            indices = torch.randint(0, n_protos, (batch_size,), device=self.device)
        else:
            indices = torch.randperm(n_protos, device=self.device)[:batch_size]

        # Stack prototypes (already tensors)
        proto_aug = torch.stack(self.prototype, dim=0)[indices]

        # Optimized: Use tensor indexing instead of list comprehension
        class_label_tensor = torch.tensor(self.class_label, dtype=torch.long, device=self.device)
        proto_aug_label = class_label_tensor[indices]

        return proto_aug, proto_aug_label

    def _get_classifier_weights(self):
        """Get classifier weight matrix from the model head."""
        if hasattr(self.model.head, 'fc'):
            return self.model.head.fc.weight
        elif hasattr(self.model.head, 'weight'):
            return self.model.head.weight
        else:
            return None

    def _compute_sigma2_fast(self, labels, weight_m, C, A):
        """
        OPTIMIZED sigma2 computation using batched tensor operations.
        """
        device = self.device
        N = labels.size(0)

        # Map labels to prototype indices (using cache)
        proto_indices = torch.tensor([self.label_to_proto_idx.get(l.item(), 0) for l in labels],
                                      dtype=torch.long, device=device)

        # Batched covariance lookup
        CV_batch = self.cov_tensor[proto_indices]  # (N, A, A)

        # Expand weights
        NxW_ij = weight_m.unsqueeze(0).expand(N, C, A)  # (N, C, A)
        labels_expanded = labels.view(N, 1, 1).expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels_expanded)  # (N, C, A)

        # Compute sigma2
        diff = NxW_ij - NxW_kj  # (N, C, A)
        sigma2 = self.aug_ratio * torch.bmm(torch.bmm(diff, CV_batch), diff.permute(0, 2, 1))  # (N, C, C)

        # Take diagonal
        eye_C = torch.eye(C, device=device)
        sigma2 = sigma2.mul(eye_C.expand(N, C, C)).sum(2).view(N, C)
        return sigma2

    def after_task(self, x_train, y_train):
        """After learning a task, store prototypes and covariance as tensors."""
        self.learned_classes += self.classes_in_task

        self.model.load_state_dict(torch.load(self.ckpt_path))
        self.model.eval()

        # Compute prototypes for NEW classes
        self._compute_prototypes(x_train, y_train)

        # Handle buffer
        if self.buffer and self.er_mode == 'task':
            dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
            for batch_id, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.buffer.update(x, y)

        # NCM classifier
        if self.ncm_classifier:
            from agents.utils.functions import compute_cls_feature_mean_buffer
            self.means_of_exemplars = compute_cls_feature_mean_buffer(self.buffer, self.model)

        # OPTIMIZED: Convert covariance list to tensor for fast lookup
        if len(self.cov) > 0:
            self.cov_tensor = torch.stack([
                torch.tensor(c, dtype=torch.float32, device=self.device)
                for c in self.cov
            ])

        # Update label-to-prototype index cache
        self.label_to_proto_idx = {label: idx for idx, label in enumerate(self.class_label)}

        if self.verbose:
            print(f'[IL2A] Stored prototypes for {len(self.prototype)} classes')

    def _compute_prototypes(self, x_train, y_train):
        """Compute prototypes and covariance for NEW classes."""
        self.model.eval()
        dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=False)

        class_features = {}

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                features = self.model.feature(x)

                for i, label in enumerate(y):
                    label_id = int(label)
                    if label_id not in self.class_label:
                        if label_id not in class_features:
                            class_features[label_id] = []
                        class_features[label_id].append(features[i])

        new_prototypes = []
        new_covs = []
        new_labels = []

        for class_id, features in sorted(class_features.items()):
            if len(features) > 0:
                features_np = torch.stack(features, dim=0).cpu().numpy()

                # Mean (prototype) - store as tensor
                mean = features_np.mean(axis=0)
                new_prototypes.append(torch.tensor(mean, dtype=torch.float32, device=self.device))

                # Covariance
                n_samples = features_np.shape[0]
                if n_samples > 1:
                    cov = np.cov(features_np.T)
                    if cov.ndim == 2:
                        cov += np.eye(self.feature_dim) * 1e-5
                    else:
                        cov = np.eye(self.feature_dim) * 1e-5
                else:
                    cov = np.eye(self.feature_dim) * 1e-5

                new_covs.append(cov)
                new_labels.append(class_id)

        self.prototype.extend(new_prototypes)
        self.cov.extend(new_covs)
        self.class_label.extend(new_labels)

        if self.verbose:
            print(f'Computed prototypes and covariances for {len(new_prototypes)} new classes')

    def epoch_loss_printer(self, epoch, acc, loss):
        """Print epoch losses with breakdown"""
        if isinstance(loss, tuple) and len(loss) >= 5:
            print('Epoch {}/{}: Accuracy = {}, Total Loss = {}, CE Loss = {}, SemanAug Loss = {}, KD Feature Loss = {}, KD Logits Loss = {}'.format(
                epoch + 1, self.epochs, acc, loss[0], loss[1], loss[2], loss[3], loss[4]))
        else:
            super().epoch_loss_printer(epoch, acc, loss)

    @torch.no_grad()
    def evaluate(self, task_stream, path=None):
        """Evaluate using only REAL class outputs."""
        if self.task_now == 0:
            self.num_tasks = task_stream.n_tasks
            self.Acc_tasks = {
                'valid': np.zeros((self.num_tasks, self.num_tasks + 1)),
                'test': np.zeros((self.num_tasks, self.num_tasks + 1))
            }

        self.model.load_state_dict(torch.load(self.ckpt_path))
        self.model.eval()

        eval_modes = ['valid', 'test']
        eval_acc_i = 0.0

        for mode in eval_modes:
            if self.verbose:
                print(f'\n ======== Evaluate on {mode} set ========')

            x_eval_all_data, y_eval_all_data = [], []
            for i in range(self.task_now + 1):
                (x_eval, y_eval) = task_stream.tasks[i][1] if mode == 'valid' else task_stream.tasks[i][2]
                eval_dataloader_i = Dataloader_from_numpy(x_eval, y_eval, self.batch_size, shuffle=False)
                x_eval_all_data.append(x_eval)
                y_eval_all_data.append(y_eval)

                if self.cf_matrix and self.task_now + 1 == self.num_tasks and mode == 'test':
                    eval_loss_i, eval_acc_i = self.test_for_cf_matrix(eval_dataloader_i)
                else:
                    eval_loss_i, eval_acc_i = self.cross_entropy_epoch_run(eval_dataloader_i, mode='test')

                if self.verbose:
                    print(f'Task {i}: Accuracy == {eval_acc_i:.2f}, Test CE Loss == {eval_loss_i:.4f}')

                self.Acc_tasks[mode][self.task_now][i] = np.around(eval_acc_i, decimals=2)

            if len(x_eval_all_data) > 0:
                x_eval_all = torch.FloatTensor(np.concatenate(x_eval_all_data, axis=0)).to(self.device)
                y_eval_all = torch.LongTensor(np.concatenate(y_eval_all_data, axis=0)).to(self.device)
                eval_dataloader_all = Dataloader_from_numpy(x_eval_all, y_eval_all, self.batch_size, shuffle=False)
                eval_loss_i, eval_acc_i = self.test_for_cf_matrix(eval_dataloader_all)
                self.Acc_tasks[mode][self.task_now][-1] = eval_acc_i

                if self.verbose and mode == 'test':
                    print(f'Mean Accuracy == {eval_acc_i:.2f}, Test CE Loss == {eval_loss_i:.4f}')

            if self.task_now + 1 == self.num_tasks and self.verbose:
                with np.printoptions(suppress=True):
                    print('Accuracy matrix of all tasks:')
                    print(self.Acc_tasks[mode])

        if self.tsne and not self.args.tune:
            tsne_path = path + 't{}'.format(self.task_now)
            self.feature_space_tsne_visualization(task_stream, path=tsne_path)

        return eval_acc_i
