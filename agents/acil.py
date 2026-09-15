# -*- coding: utf-8 -*-
"""
ACIL (Analytic Class-Incremental Learning) for TSCIL framework.

Based on:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    NeurIPS 2022.
[2] Zhuang, Huiping, et al.
    "G-ACIL: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning"
    arXiv:2403.15706 (2024).

Key differences from existing agents:
- Uses analytic closed-form solution (RLS) instead of gradient descent
- Backbone is frozen after base training
- Employs RandomBuffer for feature expansion
- Supports both base training (gradient-based) and incremental learning (analytic)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base import BaseLearner
from agents.utils.analytic_linear import RecursiveLinear
from utils.data import Dataloader_from_numpy
from utils.utils import EarlyStopping
from utils.optimizer import adjust_learning_rate
from torch.optim import lr_scheduler
import random


# ===================== Gradient Reversal Layer (GRL) =====================
class GradReverse(torch.autograd.Function):
    """梯度反转层，用于领域自适应"""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# ===================== NT-Xent Contrastive Loss =====================
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for contrastive learning.
    Supports both self-supervised and supervised contrastive learning.
    """
    def __init__(self, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j, labels=None):
        """
        计算对比损失
        Args:
            z_i, z_j: 增强样本对的特征，形状为 [batch_size, feature_dim]
            labels: 如果提供，则使用有监督对比学习（相同标签为正样本）
        """
        batch_size = z_i.shape[0]

        # 拼接所有样本 [2*batch_size, feature_dim]
        z = torch.cat([z_i, z_j], dim=0)

        # 计算相似度矩阵
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # 生成正样本对标签
        if labels is not None:
            # 有监督对比学习：相同标签的样本为正样本对
            labels = torch.cat([labels, labels], dim=0)
            sim_pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
            sim_pos_mask.fill_diagonal_(0)  # 排除自身
        else:
            # 自监督对比学习：同一图像的增强为正样本对
            sim_pos_mask = torch.zeros((2 * batch_size, 2 * batch_size),
                                       dtype=torch.float32, device=self.device)
            sim_pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=self.device)
            sim_pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=self.device)

        # 生成负样本对掩码
        sim_neg_mask = 1 - torch.eye(2 * batch_size, device=self.device, dtype=torch.float32)
        if labels is None:
            # 自监督时排除同一图像的另一个增强
            for i in range(batch_size):
                sim_neg_mask[i, batch_size + i] = 0
                sim_neg_mask[batch_size + i, i] = 0

        # 计算分子（正样本相似度）
        pos = torch.exp(sim) * sim_pos_mask
        numerator = pos.sum(dim=1)

        # 计算分母（所有负样本相似度）
        neg = torch.exp(sim) * sim_neg_mask
        denominator = neg.sum(dim=1)

        # 避免除以 0
        denominator = torch.max(denominator, torch.ones_like(denominator) * 1e-8)

        # 计算损失
        loss = -torch.log(numerator / denominator)
        loss = loss.sum() / (2 * batch_size)

        return loss


# ===================== Projection Head =====================
class ProjectionHead(nn.Module):
    """
    对比学习的投影头，将特征映射到对比学习空间
    使用 LayerNorm 替代 BatchNorm 以避免 batch size 变化时的问题
    """
    def __init__(self, input_dim=100, output_dim=64, hidden_dim=256):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=1)  # L2 归一化


# ===================== Subject Discriminator =====================
class Discriminator(nn.Module):
    """
    Subject 判别器，用于领域自适应
    学习 subject 不变特征，减少 subject 间差异
    """
    def __init__(self, hidden_1, num_subjects=8):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_1)
        self.fc2 = nn.Linear(hidden_1, num_subjects)
        self.dropout1 = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


# ===================== EEG Augmentor =====================
class EEGAugmentor:
    """EEG数据增强器"""

    def __init__(self, augment_types=['noise', 'time_warp', 'channel_drop'], device='cpu'):
        self.augment_types = augment_types
        self.device = device

    def to_device(self, tensor, device=None):
        """将张量移动到指定设备"""
        if device is None:
            device = self.device
        return tensor.to(device)

    def add_gaussian_noise(self, x, noise_level=0.05):
        """添加高斯噪声"""
        noise = torch.randn_like(x, device=x.device) * noise_level
        return x + noise

    def time_warp(self, x, warp_factor=0.1):
        """时间扭曲"""
        batch_size, time_points, channels = x.shape  # 正确格式：(batch, time, channels)
        warp_points = int(time_points * warp_factor)

        # 随机选择扭曲点
        if warp_points > 0:
            warp_idx = random.sample(range(time_points), warp_points)
            for idx in warp_idx:
                # 轻微的时间偏移
                shift = random.randint(-2, 2)
                if 0 <= idx + shift < time_points:
                    x[:, idx, :] = x[:, idx + shift, :]  # 修改索引方式
        return x

    def channel_drop(self, x, drop_prob=0.1):
        """通道丢弃"""
        batch_size, time_points, channels = x.shape  # 正确格式：(batch, time, channels)
        mask = torch.ones_like(x, device=x.device)

        # 确保drop_mask在正确的设备上
        drop_mask = (torch.rand(channels, device=x.device) > drop_prob).float().unsqueeze(0).unsqueeze(1)

        # 扩展drop_mask以匹配x的形状
        drop_mask = drop_mask.expand(batch_size, time_points, channels)

        return x * drop_mask

    def random_augment(self, x, y=None):
        """随机应用一种增强"""
        if len(self.augment_types) == 0:
            return x, x  # 如果没有增强，返回两个相同的副本

        # 第一个增强
        x1 = x.clone()
        aug_type1 = random.choice(self.augment_types)
        if aug_type1 == 'noise':
            x1 = self.add_gaussian_noise(x1)
        elif aug_type1 == 'time_warp':
            x1 = self.time_warp(x1)
        elif aug_type1 == 'channel_drop':
            x1 = self.channel_drop(x1)

        # 第二个增强（可能不同）
        x2 = x.clone()
        aug_type2 = random.choice(self.augment_types)
        if aug_type2 == 'noise':
            x2 = self.add_gaussian_noise(x2)
        elif aug_type2 == 'time_warp':
            x2 = self.time_warp(x2)
        elif aug_type2 == 'channel_drop':
            x2 = self.channel_drop(x2)

        return x1, x2


class RandomBuffer(nn.Module):
    """
    Random projection buffer for feature expansion.

    Projects input features to a higher-dimensional space using
    fixed random weights, enabling linear separability.
    Uses ReLU activation (consistent with independent ACIL implementation).
    """

    def __init__(self, input_dim: int, buffer_size: int,
                 device=None, dtype=torch.double):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_dim = input_dim
        self.out_features = buffer_size
        self.buffer_size = buffer_size

        # Fixed random projection matrix
        # Shape: (buffer_size, input_dim) - same as nn.Linear
        weight = torch.empty((buffer_size, input_dim), **factory_kwargs)
        self.register_buffer("weight", weight)

        # Kaiming uniform initialization (same as independent version)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming uniform initialization, same as nn.Linear."""
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Random feature expansion with ReLU activation.
        Uses nn.Linear-style forward: X @ weight.T

        Args:
            X: Input features (N, input_dim)
        Returns:
            Expanded features (N, buffer_size)
        """
        X = X.to(self.weight)
        return torch.relu(X @ self.weight.T)


class ACILWrapper(nn.Module):
    """
    ACIL model wrapper combining backbone, random buffer, and analytic classifier.
    """

    def __init__(self, backbone: nn.Module, data: str, backbone_output: int,
                 buffer_size: int, gamma: float, n_classes: int,
                 overconf_threshold: float = 3.0, relax_margin: float = 0.3,
                 device=None, dtype=torch.double):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.backbone = backbone
        self.backbone_output = backbone_output

        # Random buffer for feature expansion
        self.buffer = RandomBuffer(backbone_output, buffer_size,  **factory_kwargs)

        # Analytic classifier with RLS
        self.classifier = RecursiveLinear(
            buffer_size,
            gamma,
            data,
            overconf_threshold=overconf_threshold,
            relax_margin=relax_margin,
            **factory_kwargs
        )

        self.n_classes = n_classes

    @torch.no_grad()
    def extract_features(self, X: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone and expand via buffer."""
        features = self.backbone.feature(X)  # (N, backbone_output)
        return self.buffer(features)  # (N, buffer_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass: backbone → buffer → analytic classifier."""
        phi = self.extract_features(X)
        return self.classifier(phi)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Analytic learning step.

        Args:
            X: Input samples (N, L, C)
            y: Labels (N,) as class indices
        """
        # Extract and expand features
        phi = self.extract_features(X)

        # Convert labels to one-hot
        y = y.view(-1).long()
        num_classes = max(y.max().item() + 1, self.n_classes)
        Y = F.one_hot(y, num_classes=num_classes).to(phi.dtype)

        # Update RLS classifier
        self.classifier.fit(phi, Y)

    @torch.no_grad()
    def update(self) -> None:
        """Finalize RLS update (check numerical stability)."""
        self.classifier.update()


class ACIL(BaseLearner):
    """
    ACIL Agent for TSCIL framework.

    Two-phase learning:
    1. Base Training (Task 0): Train backbone with gradient descent
    2. Incremental Learning (Task 1+): Freeze backbone, use analytic RLS for classifier

    Key advantages:
    - No catastrophic forgetting (analytic solution preserves old knowledge)
    - Fast incremental updates (no backpropagation needed)
    - Memory efficient (fixed-size random buffer)
    """

    def __init__(self, model: nn.Module, args):
        super(ACIL, self).__init__(model, args)

        # ACIL hyperparameters
        self.buffer_size = args.acil_buffer_size
        self.gamma = args.acil_gamma
        self.overconf_threshold = args.acil_overconf_threshold
        self.relax_margin = args.acil_relax_margin
        self.base_epochs = args.epochs
        self.dtype = torch.double  # Use double for numerical stability

        # Create ACIL wrapper with frozen backbone
        self.acil_model = ACILWrapper(
            data=args.data,
            backbone=model,
            backbone_output=args.feature_dim,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            n_classes=model.head.out_features,
            overconf_threshold=self.overconf_threshold,
            relax_margin=self.relax_margin,
            device=self.device,
            dtype=self.dtype
        ).to(self.device)

        # Base training optimizer (only for first task)
        self.base_optimizer = None

        # Track if base training is done
        self.base_trained = False

        # ========== 领域自适应和对比学习 ==========
        self.use_contrastive = args.use_contrastive


        if self.use_contrastive:
            # Projection Head for contrastive learning
            self.projection_head = ProjectionHead(
                input_dim=args.feature_dim,
                output_dim=args.projection_dim,
                hidden_dim=args.projection_hidden).to(self.device)
            self.optimizer_p = torch.optim.AdamW(
                self.projection_head.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
            # NT-Xent Loss
            self.contrastive_criterion = NTXentLoss(
                temperature=args.temperature,
                device=self.device
            )
            # EEG Augmentor
            self.augmentor = EEGAugmentor(
                augment_types=args.augment_types,
                device=self.device
            )
            self.lambda_contrast = args.lambda_contrast


        print(
            f'ACIL: buffer_size={self.buffer_size}, gamma={self.gamma}, '
            f'overconf_threshold={self.overconf_threshold}, '
            f'relax_margin={self.relax_margin}, base_epochs={self.base_epochs}'
        )

        if self.use_contrastive:
            print(f'  Contrastive Learning: lambda_contrast={self.lambda_contrast}, temperature={self.contrastive_criterion.temperature}')

    def before_task(self, y_train):
        """
        Override to handle ACIL-specific task preparation.
        """
        # Call parent method first (this updates task_now and classes_in_task)
        super().before_task(y_train)

        # Update ACIL model's n_classes to match current head
        self.acil_model.n_classes = self.model.head.out_features

    def learn_task(self, task):
        """
        Override learn_task for ACIL's two-phase learning.

        Task 0 (Base Training): Multiple epochs with gradient descent
        Task 1+ (Incremental): Single pass with analytic RLS updates
        """
        (x_train, y_train), (x_val, y_val), _ = task

        self.before_task(y_train)
        train_dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=True)
        val_dataloader = Dataloader_from_numpy(x_val, y_val, self.batch_size, shuffle=False)

        if not self.base_trained:
            # Phase 1: Base Training with gradient descent
            self._base_training_loop(train_dataloader, val_dataloader)
            # Note: base_trained is set to True AFTER after_task in Task 0
            # This allows after_task to initialize RLS properly
        else:
            # Phase 2: Incremental Learning with analytic RLS
            self._incremental_learning_loop(train_dataloader)

        self.after_task(x_train, y_train)

        # Mark base training as completed after Task 0's after_task
        if not self.base_trained:
            self.base_trained = True

    def _base_training_loop(self, train_dataloader, val_dataloader):
        """
        Base training phase: train backbone with gradient descent.
        Uses early stopping on training set (no validation set).
        Integrates contrastive learning and domain adaptation when enabled.
        """
        self.model.train()
        if self.use_contrastive:
            self.projection_head.train()

        early_stopping = EarlyStopping(path=self.ckpt_path, patience=self.args.patience, mode='max', verbose=False)

        if self.base_optimizer is None:
            self.base_optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.base_optimizer,
            steps_per_epoch=len(train_dataloader),
            epochs=self.base_epochs,
            max_lr=self.args.lr
        )

        for epoch in range(self.base_epochs):
            # Train for one epoch
            epoch_loss_train, epoch_acc_train = self._base_train_epoch(
                train_dataloader, epoch,
                optimizer_p=self.optimizer_p if self.use_contrastive else None
            )

            # Validate on training set for early stopping
            epoch_loss_val, epoch_acc_val = self.cross_entropy_epoch_run(val_dataloader, mode='val')

            if self.args.lradj != 'TST':
                adjust_learning_rate(self.base_optimizer, scheduler, epoch + 1, self.args)

            if self.verbose:
                self.epoch_loss_printer(epoch, epoch_acc_train, epoch_loss_train)

            early_stopping(epoch_acc_val, self.model)
            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

    def _base_train_epoch(self, dataloader, epoch, optimizer_p=None):
        """
        Base training epoch: train backbone with gradient descent.

        Integrates:
        1. Contrastive Learning (NT-Xent loss + ProjectionHead)
        2. Domain Adaptation (GRL + Subject Discriminator)
        """
        self.model.train()
        if self.use_contrastive:
            self.projection_head.train()

        total = 0
        correct = 0
        epoch_loss = 0

        # Clear features from previous epochs to save memory
        # (Removed - features are now re-extracted from best model in _initialize_rls_from_base)
        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            # ===== 前向传播 =====
            features = self.model.encoder(x)
            # outputs = features
            outputs = self.model.head(features)

            # ===== 1. 分类损失 =====
            classifier_loss = self.criterion(outputs, y)
            loss = classifier_loss

            # ===== 2. 对比学习 (NT-Xent + ProjectionHead) =====
            contrastive_loss_value = torch.tensor(0.0, device=self.device)
            if self.use_contrastive:
                # 数据增强
                with torch.no_grad():
                    x_aug1, x_aug2 = self.augmentor.random_augment(x)

                # 通过 backbone 获取增强特征（使用 no_grad 避免对 backbone 求梯度）
                # with torch.no_grad():
                features_aug1 = self.model.encoder(x_aug1)
                features_aug2 = self.model.encoder(x_aug2)

                # 通过投影头并计算对比损失
                proj_aug1 = self.projection_head(features_aug1)
                proj_aug2 = self.projection_head(features_aug2)
                contrastive_loss_value = self.contrastive_criterion(proj_aug1, proj_aug2, y)

                # 将对比损失加到总损失中
                loss = loss + self.lambda_contrast * contrastive_loss_value

            # ===== 反向传播 =====
            self.base_optimizer.zero_grad()
            loss.backward()
            self.optimizer_step(epoch=epoch)

            # 更新 projection head（在 main loss backward 之后）
            if optimizer_p is not None and self.use_contrastive:
                optimizer_p.zero_grad()
                # 重新前向传播以计算 projection head 的梯度
                with torch.no_grad():
                    x_aug1, x_aug2 = self.augmentor.random_augment(x)
                    features_aug1 = self.model.encoder(x_aug1)
                    features_aug2 = self.model.encoder(x_aug2)
                proj_aug1 = self.projection_head(features_aug1)
                proj_aug2 = self.projection_head(features_aug2)
                contrastive_loss = self.contrastive_criterion(proj_aug1, proj_aug2, y)
                contrastive_loss.backward()
                optimizer_p.step()

            epoch_loss += loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()


        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)

        return epoch_loss, epoch_acc

    def _incremental_learning_loop(self, train_dataloader):
        """
        Incremental learning phase: analytic RLS updates.
        Single pass through data (no multiple epochs needed).
        """
        self.model.eval()  # Freeze backbone
        self.acil_model.eval()

        # Single pass through training data
        total = 0
        correct = 0
        epoch_loss = 0

        for batch_id, (x, y) in enumerate(train_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            # Analytic update (no gradient)
            with torch.no_grad():
                self.acil_model.fit(x, y)

            # Compute loss for logging
            with torch.no_grad():
                outputs = self.acil_model(x)
                loss = self.criterion(outputs, y)
                prediction = torch.argmax(outputs, dim=1)
                correct += prediction.eq(y).sum().item()
                epoch_loss += loss
        # self.acil_model.update()
        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)

        if self.verbose:
            print(f'Incremental update: Accuracy = {epoch_acc:.2f}, Loss = {epoch_loss:.4f}')


    def train_epoch(self, dataloader, epoch):
        """
        Required abstract method, but ACIL uses custom learn_task loop.
        This is kept for compatibility but may not be called directly.
        """
        # This method is called by the parent learn_task, which we override
        # Returning dummy values to satisfy the interface
        return 0.0, 0.0

    def after_task(self, x_train, y_train):
        """
        Finalize task learning.
        """
        self.learned_classes += self.classes_in_task

        if not self.base_trained:
            # End of base training phase - load best model and initialize RLS
            # 1. Load best backbone weights from checkpoint
            self.model.load_state_dict(torch.load(self.ckpt_path))

            # 2. Sync to ACIL model
            self.acil_model.backbone.load_state_dict(self.model.state_dict())

            # 3. Initialize RLS using features from best model
            self._initialize_rls_from_base(x_train, y_train)
        else:
            # Incremental phase: finalize RLS update
            self.acil_model.update()

        # Save ACIL model checkpoint
        torch.save(self.acil_model.state_dict(), self.ckpt_path)

    def _initialize_rls_from_base(self, x_train, y_train, batch=20):
        """
        Initialize RLS classifier using features extracted by the best backbone.

        Args:
            x_train: Training samples from Task 0 (numpy array)
            y_train: Training labels from Task 0 (numpy array)
            batch: Batch size for RLS updates
        """
        # Ensure backbone is loaded with best weights and in eval mode
        self.model.eval()

        # Re-extract features using the best model
        all_features = []
        all_labels = []

        n_samples = x_train.shape[0]
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            x_batch = torch.FloatTensor(x_train[start:end]).to(self.device)
            y_batch = torch.LongTensor(y_train[start:end]).to(self.device)

            with torch.no_grad():
                features = self.model.feature(x_batch)
                all_features.append(features.cpu())
                all_labels.append(y_batch.cpu())

        all_features = torch.cat(all_features, dim=0).to(self.device)
        all_labels = torch.cat(all_labels, dim=0).to(self.device)

        # Expand features through random buffer
        phi = self.acil_model.buffer(all_features)

        # Convert to one-hot
        num_classes = len(self.learned_classes)
        Y = F.one_hot(all_labels, num_classes=int(num_classes)).to(phi.dtype)

        # Initialize RLS
        n_sample = phi.shape[0]
        for start_idx in range(0, n_sample, batch):
            end_idx = min(start_idx + batch, n_sample)
            batch_features = phi[start_idx:end_idx]
            batch_labels = Y[start_idx:end_idx]
            self.acil_model.classifier.fit(batch_features, batch_labels)
            self.acil_model.classifier.update()

        print(f'RLS initialized with {phi.shape[0]} base samples using best backbone')

    @torch.no_grad()
    def evaluate(self, task_stream, path=None):
        """
        Evaluate using ACIL model (analytic classifier) in incremental phase,
        or original model in base training phase (Task 0).
        Only evaluates on test set (no validation set).
        """
        # Get num_tasks and create Accuracy Matrix
        if self.task_now == 0:
            self.num_tasks = task_stream.n_tasks
            self.Acc_tasks = {
                'valid': np.zeros((self.num_tasks, self.num_tasks + 1 )),
                'test': np.zeros((self.num_tasks, self.num_tasks + 1 ))
            }

        self.acil_model.load_state_dict(torch.load(self.ckpt_path, weights_only=True))
        self.acil_model.eval()
        eval_model = self.acil_model

        eval_modes = ['valid', 'test']  # 'valid' is for checking generalization.
        eval_acc_i = 0.0  # Return value

        for mode in eval_modes:
            if self.verbose:
                print(f'\n ======== Evaluate on {mode} set ========')

            x_eval_all_data, y_eval_all_data = [], []
            for i in range(self.task_now + 1):
                (x_eval, y_eval) = task_stream.tasks[i][1] if mode == 'valid' else task_stream.tasks[i][2]

                x_eval_all_data.append(x_eval)
                y_eval_all_data.append(y_eval)

                # Evaluation without dataloader overhead
                x_tensor = torch.FloatTensor(x_eval).to(self.device)
                y_tensor = torch.LongTensor(y_eval).to(self.device)

                outputs = eval_model(x_tensor)
                prediction = torch.argmax(outputs, dim=1)

                # Collect predictions for confusion matrix
                if self.cf_matrix and self.task_now + 1 == self.num_tasks and mode == 'test':
                    self.y_pred_cf.extend(prediction.cpu().numpy())
                    self.y_true_cf.extend(y_eval)

                acc = 100. * (prediction == y_tensor).float().mean().item()

                if self.verbose:
                    print(f'Task {i}: Accuracy == {acc:.2f}')

                self.Acc_tasks[mode][self.task_now][i] = acc
                eval_acc_i = acc

            # Mean accuracy over all tasks
            if len(x_eval_all_data) > 0:
                x_eval_all = torch.FloatTensor(np.concatenate(x_eval_all_data, axis=0)).to(self.device)
                y_eval_all = torch.LongTensor(np.concatenate(y_eval_all_data, axis=0)).to(self.device)
                outputs_all = eval_model(x_eval_all)
                prediction_all = torch.argmax(outputs_all, dim=1)
                self.Acc_tasks[mode][self.task_now][-1] = 100. * (prediction_all == y_eval_all).float().mean().item()

                if self.verbose and mode == 'test':
                    print(f'Mean Accuracy == {self.Acc_tasks[mode][self.task_now][-1]:.2f}')

            # Print accuracy matrix after final task
            if self.task_now + 1 == self.num_tasks and self.verbose:
                with np.printoptions(precision=2, suppress=True):
                    print('Accuracy matrix of all tasks:')
                    print(self.Acc_tasks[mode])

        return eval_acc_i

    def cross_entropy_epoch_run(self, dataloader, epoch=None, mode='train'):
        """
        Override to use ACIL model in incremental phase.
        """
        if self.base_trained:
            # Use analytic model
            self.acil_model.eval()
            total = 0
            correct = 0
            epoch_loss = 0

            ce_loss = self.criterion
            for batch_id, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                total += y.size(0)

                with torch.no_grad():
                    outputs = self.acil_model(x)
                    step_loss = ce_loss(outputs, y)

                epoch_loss += step_loss
                prediction = torch.argmax(outputs, dim=1)
                correct += prediction.eq(y).sum().item()

            epoch_acc = 100. * (correct / total)
            epoch_loss /= (batch_id + 1)
            return float(epoch_loss), float(epoch_acc)
        else:
            # Use standard training (base phase)
            return super().cross_entropy_epoch_run(dataloader, epoch, mode)
