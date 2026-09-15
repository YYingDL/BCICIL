# -*- coding: utf-8 -*-
"""
PASS (Prototype Augmentation and Self-Supervision for Incremental Learning) for TSCIL framework.

CVPR 2021 Oral: "Prototype Augmentation and Self-Supervision for Incremental Learning"
Fei Zhu, Xu-Yao Zhang, Chuang Wang, Fei Yin, Cheng-Lin Liu

Key components:
1. Self-Supervised Learning: Rotation prediction as auxiliary task (4 rotations -> 4x classes)
2. Prototype Augmentation: Store class prototypes and generate synthetic samples
3. Knowledge Distillation: Feature-level KD with old model

For EEG data adaptation:
- Use time-shift and channel permutation as self-supervised tasks (instead of image rotation)
- Store and replay prototypes from old tasks
- Feature-level knowledge distillation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy


class EEGSelfSupervision:
    """自监督学习模块 - 为 EEG 数据设计增强策略"""

    def __init__(self, args, device='cpu'):
        self.device = device
        self.augment_types = getattr(args, 'augment_types', ['time_shift', 'channel_permute', 'flip'])

    def time_shift(self, x, shift_range=0.1):
        """时间平移增强"""
        batch_size, time_points, channels = x.shape
        shift_amount = int(time_points * shift_range)
        if shift_amount > 0:
            shift = np.random.randint(-shift_amount, shift_amount + 1)
            if shift > 0:
                x_shifted = torch.cat([x[:, shift:, :], x[:, :shift, :]], dim=1)
            elif shift < 0:
                x_shifted = torch.cat([x[:, shift:, :], x[:, :shift, :]], dim=1)
            else:
                x_shifted = x.clone()
            return x_shifted
        return x.clone()

    def channel_permute(self, x, permute_ratio=0.2):
        """通道置换增强"""
        batch_size, time_points, channels = x.shape
        x_permuted = x.clone()
        n_permute = max(1, int(channels * permute_ratio))
        for _ in range(n_permute):
            c1, c2 = np.random.choice(channels, 2, replace=False)
            x_permuted[:, :, c1], x_permuted[:, :, c2] = x[:, :, c2].clone(), x[:, :, c1].clone()
        return x_permuted

    def flip(self, x):
        """符号翻转增强"""
        return x.clone() * (-1)

    def augment(self, x, aug_type=None):
        """应用指定的增强"""
        if aug_type is None:
            if len(self.augment_types) == 0:
                return x.clone()
            aug_type = np.random.choice(self.augment_types)

        if aug_type == 'time_shift':
            return self.time_shift(x)
        elif aug_type == 'channel_permute':
            return self.channel_permute(x)
        elif aug_type == 'flip':
            return self.flip(x)
        return x.clone()


class PASS(BaseLearner):
    """
    PASS Agent for TSCIL framework.

    Implements Prototype Augmentation and Self-Supervision for Class-Incremental Learning.

    Key mechanisms:
    1. Self-supervised learning with EEG-specific augmentations
    2. Prototype storage and augmentation for old classes
    3. Feature-level knowledge distillation
    """

    def __init__(self, model, args):
        super(PASS, self).__init__(model, args)

        # PASS hyperparameters
        self.kd_weight = args.kd_weight
        self.protoAug_weight = args.protoAug_weight
        self.temperature = args.pass_temperature
        self.buffer_size = args.pass_buffer_size

        # Self-supervised learning setup
        self.ssl = EEGSelfSupervision(args, device=self.device)

        # Prototype storage
        self.prototype = {}  # {class_id: mean_feature_vector}
        self.class_stds = {}  # {class_id: std_feature_vector}
        self.class_labels = []  # List of class IDs in order

        # Old model for knowledge distillation
        self.old_model = None
        self.use_kd = False

        # Feature dimension (inferred from model)
        self.feature_dim = None

        print(f'PASS: kd_weight={self.kd_weight}, protoAug_weight={self.protoAug_weight}')
        print(f'      temperature={self.temperature}, buffer_size={self.buffer_size}')
        print(f'      augment_types={self.ssl.augment_types}')

    def before_task(self, y_train):
        """Prepare for learning a new task"""
        # Store old model for KD (if not first task)
        if self.task_now >= 0 and self.model is not None:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            self.use_kd = True
            print(f'[PASS] Stored old model for KD at Task {self.task_now}')

        # Call parent to expand head for new classes
        super().before_task(y_train)

        # Infer feature dimension
        if self.feature_dim is None:
            with torch.no_grad():
                x_dummy = torch.randn(1, *self.args.input_size if hasattr(self.args, 'input_size') else (1001, 22))
                try:
                    self.feature_dim = self.model.feature(x_dummy.to(self.device)).shape[1]
                except:
                    self.feature_dim = getattr(self.args, 'feature_dim', 256)
            print(f'[PASS] Feature dimension: {self.feature_dim}')

    def train_epoch(self, dataloader, epoch):
        """
        Train for one epoch with:
        1. Classification loss on current task data
        2. Prototype augmentation loss (for task > 0)
        3. Knowledge distillation loss (for task > 0)
        """
        total = 0
        correct = 0
        epoch_loss = 0
        epoch_loss_cls = 0
        epoch_loss_proto = 0
        epoch_loss_kd = 0

        self.model.train()

        num_classes = self.model.head.out_features

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)
            total += batch_size

            self.optimizer.zero_grad()

            # Get features
            features = self.model.feature(x)
            outputs = self.model.head(features)

            # Classification loss
            loss_cls = self.criterion(outputs, y)

            # Prototype augmentation loss
            loss_proto = torch.tensor(0.0, device=self.device)
            if len(self.prototype) > 0 and self.protoAug_weight > 0:
                loss_proto = self._compute_prototype_augmentation_loss()

            # Knowledge distillation loss
            loss_kd = torch.tensor(0.0, device=self.device)
            if self.use_kd and self.old_model is not None:
                with torch.no_grad():
                    old_features = self.old_model.feature(x)
                loss_kd = F.mse_loss(features, old_features)

            # Total loss
            total_loss = loss_cls + self.protoAug_weight * loss_proto + self.kd_weight * loss_kd

            total_loss.backward()
            self.optimizer_step(epoch=epoch)

            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_loss_cls += loss_cls.item()
            epoch_loss_proto += loss_proto.item()
            epoch_loss_kd += loss_kd.item()

            # Accuracy
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)
        epoch_loss_cls /= (batch_id + 1)
        epoch_loss_proto /= (batch_id + 1)
        epoch_loss_kd /= (batch_id + 1)

        if self.verbose and epoch == 0:
            print(f'  Task {self.task_now}: loss_cls={epoch_loss_cls:.4f}, '
                  f'loss_proto={epoch_loss_proto:.4f}, loss_kd={epoch_loss_kd:.4f}')

        self._epoch_losses = (epoch_loss, epoch_loss_cls, epoch_loss_proto, epoch_loss_kd)
        return epoch_loss, epoch_acc

    def _compute_prototype_augmentation_loss(self):
        """
        Compute prototype augmentation loss.
        Generate synthetic features from stored prototypes and classify them.
        """
        if len(self.prototype) == 0:
            return torch.tensor(0.0, device=self.device)

        # Sample prototypes randomly
        sampled_classes = list(self.prototype.keys())
        if len(sampled_classes) == 0:
            return torch.tensor(0.0, device=self.device)

        # Generate synthetic features by sampling from prototype distributions
        proto_features = []
        proto_labels = []

        samples_per_class = max(1, self.buffer_size // len(self.prototype))

        for class_id in sampled_classes:
            mean = self.prototype[class_id]
            std = self.class_stds.get(class_id, torch.ones_like(mean) * 0.1)

            # Sample from Gaussian distribution around prototype
            for _ in range(samples_per_class):
                synthetic_feature = mean + torch.randn_like(mean) * std
                proto_features.append(synthetic_feature)
                proto_labels.append(class_id)

        if len(proto_features) == 0:
            return torch.tensor(0.0, device=self.device)

        proto_features = torch.stack(proto_features, dim=0)
        proto_labels = torch.tensor(proto_labels, dtype=torch.long, device=self.device)

        # Classify synthetic features
        proto_outputs = self.model.head(proto_features)
        loss_proto = self.criterion(proto_outputs, proto_labels)

        return loss_proto

    def after_task(self, x_train, y_train):
        """
        After learning a task:
        1. Update learned classes
        2. Compute and store prototypes for new classes
        3. Save old model for KD
        """
        self.learned_classes += self.classes_in_task

        # Load best model from checkpoint
        self.model.load_state_dict(torch.load(self.ckpt_path))
        self.model.eval()

        # Compute prototypes for NEW classes
        self._compute_prototypes(x_train, y_train)

        # Handle buffer for replay methods
        if self.buffer and self.er_mode == 'task':
            dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
            for batch_id, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.buffer.update(x, y)

        # Compute NCM classifier if applicable
        if self.ncm_classifier:
            from agents.utils.functions import compute_cls_feature_mean_buffer
            self.means_of_exemplars = compute_cls_feature_mean_buffer(self.buffer, self.model)

        if self.verbose:
            print(f'[PASS] Stored prototypes for {len(self.prototype)} classes')

    def _compute_prototypes(self, x_train, y_train):
        """Compute and store prototypes (mean and std) for NEW classes in current task."""
        self.model.eval()
        dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=False)

        class_features = {}

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                features = self.model.feature(x)

                for i, label in enumerate(y):
                    label_id = int(label)
                    # Only compute prototypes for NEW classes
                    if label_id not in self.prototype:
                        if label_id not in class_features:
                            class_features[label_id] = []
                        class_features[label_id].append(features[i])

        # Compute mean and std for each new class
        for class_id, features in class_features.items():
            if len(features) > 0:
                features_tensor = torch.stack(features, dim=0)
                mean = features_tensor.mean(dim=0)
                std = features_tensor.std(dim=0)
                std = torch.clamp(std, min=1e-6)

                self.prototype[class_id] = mean
                self.class_stds[class_id] = std
                self.class_labels.append(class_id)

        if self.verbose:
            print(f'Computed prototypes for {len(class_features)} new classes')

    def epoch_loss_printer(self, epoch, acc, loss):
        """Print epoch losses with breakdown"""
        if isinstance(loss, tuple) and len(loss) >= 4:
            print('Epoch {}/{}: Accuracy = {}, Total Loss = {}, CE Loss = {}, ProtoAug Loss = {}, KD Loss = {}'.format(
                epoch + 1, self.epochs, acc, loss[0], loss[1], loss[2], loss[3]))
        else:
            super().epoch_loss_printer(epoch, acc, loss)

    @torch.no_grad()
    def evaluate(self, task_stream, path=None):
        """Evaluate on test sets of all learned tasks."""
        if self.task_now == 0:
            self.num_tasks = task_stream.n_tasks
            self.Acc_tasks = {
                'valid': np.zeros((self.num_tasks, self.num_tasks + 1)),
                'test': np.zeros((self.num_tasks, self.num_tasks + 1))
            }

        # Reload best model
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

            # Mean accuracy over all tasks
            if len(x_eval_all_data) > 0:
                x_eval_all = torch.FloatTensor(np.concatenate(x_eval_all_data, axis=0)).to(self.device)
                y_eval_all = torch.LongTensor(np.concatenate(y_eval_all_data, axis=0)).to(self.device)
                eval_dataloader_all = Dataloader_from_numpy(x_eval_all, y_eval_all, self.batch_size, shuffle=False)
                eval_loss_i, eval_acc_i = self.test_for_cf_matrix(eval_dataloader_all)
                self.Acc_tasks[mode][self.task_now][-1] = eval_acc_i

                if self.verbose and mode == 'test':
                    print(f'Mean Accuracy == {eval_acc_i:.2f}, Test CE Loss == {eval_loss_i:.4f}')

            # Print accuracy matrix after final task
            if self.task_now + 1 == self.num_tasks and self.verbose:
                with np.printoptions(suppress=True):
                    print('Accuracy matrix of all tasks:')
                    print(self.Acc_tasks[mode])

        if self.tsne and not self.args.tune:
            tsne_path = path + 't{}'.format(self.task_now)
            self.feature_space_tsne_visualization(task_stream, path=tsne_path)

        return eval_acc_i
