# -*- coding: utf-8 -*-
"""
Analytic Linear Layers for Class-Incremental Learning.

Based on ACIL (Analytic Class-Incremental Learning) framework:
[1] Zhuang et al., "ACIL: Analytic class-incremental learning with absolute
    memorization and privacy protection." NeurIPS 2022.
[2] Zhuang et al., "G-ACIL: Analytic Learning for Exemplar-Free Generalized
    Class Incremental Learning." arXiv:2403.15706 (2024).

These layers use Recursive Least Squares (RLS) for closed-form weight updates,
enabling fast incremental learning without gradient descent.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from abc import abstractmethod, ABCMeta


class AnalyticLinear(nn.Module, metaclass=ABCMeta):
    """
    Base class for analytic linear layers.

    Unlike standard nn.Linear, analytic layers compute weights using
    closed-form solutions (e.g., Ridge Regression, RLS) instead of
    gradient descent.

    Args:
        in_features: Number of input features
        gamma: Regularization parameter (larger = more regularization)
        bias: Whether to use bias (appends 1 to input)
        device: Device for computation
        dtype: Data type (torch.double recommended for stability)
    """

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gamma: float = gamma
        self.bias: bool = bias
        self.dtype = dtype

        # Weight matrix: (in_features, out_features)
        # out_features grows dynamically during incremental learning
        actual_in_features = in_features + 1 if bias else in_features
        weight = torch.zeros((actual_in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: X @ weight (+ bias handling).

        Args:
            X: Input tensor (N, in_features)
        Returns:
            Output tensor (N, out_features)
        """
        X = X.to(self.weight)
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)
        return X @ self.weight

    @property
    def in_features(self) -> int:
        if self.bias:
            return self.weight.shape[0] - 1
        return self.weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def reset_parameters(self) -> None:
        """Reset weights to zero (analytic layers don't need initialization)."""
        self.weight = torch.zeros((self.weight.shape[0], 0)).to(self.weight)

    @torch.no_grad()
    def expand_classes(self, new_num_classes: int) -> None:
        """
        Expand classifier to accommodate new classes.

        Args:
            new_num_classes: Total number of classes after expansion
        """
        current_classes = self.out_features
        if new_num_classes > current_classes:
            increment_size = new_num_classes - current_classes
            tail = torch.zeros((self.weight.shape[0], increment_size),
                              dtype=self.weight.dtype,
                              device=self.weight.device)
            self.weight = torch.cat((self.weight, tail), dim=1)

    @abstractmethod
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Fit the layer using analytic solution.

        Args:
            X: Input features (N, in_features)
            Y: Target labels (N, out_features) - typically one-hot encoded
        """
        raise NotImplementedError()

    def update(self) -> None:
        """
        Finalize update step (check numerical stability).

        Should be called after all fit() calls for a task.
        """
        assert torch.isfinite(self.weight).all(), (
            "Numerical instability detected! "
            "Try: (1) increase gamma, (2) use torch.double, "
            "(3) normalize input features."
        )


class RecursiveLinear(AnalyticLinear):
    """
    Recursive Least Squares (RLS) linear layer.

    Implements the core ACIL algorithm using Sherman-Morrison-Woodbury formula
    for efficient recursive updates without explicit matrix inversion.

    The update rule is:
        R_{t+1} = R_t - R_t X^T (I + X R_t X^T)^{-1} X R_t
        W_{t+1} = W_t + R_{t+1} X^T (Y - X W_t)

    where R is the inverse feature autocorrelation matrix and W is the weight.

    Args:
        in_features: Number of input features
        gamma: Regularization parameter (initial R = I/gamma)
        bias: Whether to use bias
        device: Device for computation
        dtype: Data type (torch.double strongly recommended)

    Example:
        >>> layer = RecursiveLinear(128, gamma=1e-3, dtype=torch.double)
        >>> layer.fit(X_batch1, Y_batch1)  # First batch
        >>> layer.fit(X_batch2, Y_batch2)  # Second batch (incremental)
        >>> layer.update()  # Finalize
        >>> output = layer(X_test)  # Inference
    """

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        data: str = None,
        overconf_threshold: float = 3.0,
        relax_margin: float = 0.3,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Regularized Feature Autocorrelation Matrix (inverse)
        # R = (X^T X + gamma * I)^{-1}
        # Initialized as I/gamma (prior before seeing data)
        self.data = data
        self.overconf_threshold = overconf_threshold
        self.relax_margin = relax_margin
        self.R: torch.Tensor
        R = torch.eye(self.in_features, **factory_kwargs) / gamma
        self.register_buffer("R", R)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Recursive Least Squares update for a mini-batch.

        This implementation follows G-ACIL [2], which supports:
        - Mini-batch learning (not just single samples)
        - General CIL setting (class expansion)
        - Target relaxation for improved stability

        Args:
            X: Input features (N, in_features)
            Y: Target one-hot labels (N, num_classes)
        """
        X, Y = X.to(self.weight), Y.to(self.weight)
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)

        num_targets = Y.shape[1]

        # Handle class expansion
        if num_targets > self.out_features:
            # Add new class columns
            increment_size = num_targets - self.out_features
            tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
            self.weight = torch.cat((self.weight, tail), dim=1)
        elif num_targets < self.out_features:
            # Pad Y with zeros for existing classes
            tail = torch.zeros((Y.shape[0], self.out_features - num_targets)).to(Y)
            Y = torch.cat((Y, tail), dim=1)

        # ========== Target Relaxation (G-ACIL technique) ==========
        # Pre-compute current predictions
        current_pred = X @ self.weight  # (N, num_classes)

        # Create dynamic targets with relaxation
        # neg_value = -0.15 if num_targets > 1 else 0.0
        neg_value = 0
        Y_dynamic = Y.clone()
        neg_value = -0.02 if num_targets > 1 else 0.0
        # seed 0.08
        Y_dynamic = torch.full_like(Y, neg_value)  # Negative classes: -0.15
        Y_dynamic[Y == 1] = 1.0  # Positive class: 1.0

        true_labels = torch.argmax(Y, dim=1)
        row_indices = torch.arange(X.shape[0], device=X.device)
        correct_class_scores = current_pred[row_indices, true_labels]
        #
        # Handle over-confident samples.
        # Keep original prediction to avoid distorting the hyperplane
        over_confident_mask = (correct_class_scores > self.overconf_threshold)
        if over_confident_mask.any():
            Y_dynamic[row_indices[over_confident_mask], true_labels[over_confident_mask]] = \
                correct_class_scores[over_confident_mask]

        # Handle negative classes with safe margin relaxation
        # If negative score > 0 but still below correct class by margin, keep it
        for c in range(Y_dynamic.shape[1]):
            non_target_mask = (true_labels != c)
            if non_target_mask.any():
                negative_scores = current_pred[non_target_mask, c]
                target_scores = correct_class_scores[non_target_mask]
                relax_mask = (negative_scores > 0) & (negative_scores < (target_scores - self.relax_margin))
                if relax_mask.any():
                    actual_indices = row_indices[non_target_mask][relax_mask]
                    Y_dynamic[actual_indices, c] = negative_scores[relax_mask]
        # ========== End Target Relaxation ==========

        # Compute Kalman gain matrix
        # K = (I + X R X^T)^{-1}
        # Note: For large batches, this inversion can be expensive
        # Consider using conjugate gradient for very large batches
        K_inv = torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T

        try:
            K = torch.inverse(K_inv)
        except RuntimeError as e:
            # Fallback: move to CPU for inversion (more stable)
            K = torch.inverse(K_inv.cpu()).to(self.weight.device)

        # Update R (Sherman-Morrison-Woodbury)
        # Equation (10) in ACIL paper
        self.R -= self.R @ X.T @ K @ X @ self.R

        # Update weights (RLS update rule with relaxed targets)
        # Equation (9) in ACIL paper
        self.weight += self.R @ X.T @ (Y_dynamic - X @ self.weight)

    def update(self) -> None:
        """Check numerical stability after updates."""
        super().update()


class RidgeLinear(AnalyticLinear):
    """
    Ridge Regression linear layer (batch version).

    Computes the closed-form solution:
        W = (X^T X + gamma * I)^{-1} X^T Y

    This is simpler than RecursiveLinear but requires storing all data
    or sufficient statistics.

    Args:
        in_features: Number of input features
        gamma: Regularization parameter
        bias: Whether to use bias
        device: Device for computation
        dtype: Data type
    """

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Sufficient statistics
        self.register_buffer("XtX", torch.zeros((self.in_features, self.in_features), **factory_kwargs))
        self.register_buffer("XtY", torch.zeros((self.in_features, 0), **factory_kwargs))

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        Accumulate sufficient statistics.

        Args:
            X: Input features (N, in_features)
            Y: Target one-hot labels (N, num_classes)
        """
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)

        num_targets = Y.shape[1]

        # Expand XtY if new classes are added
        if num_targets > self.XtY.shape[1]:
            tail = torch.zeros((self.XtY.shape[0], num_targets - self.XtY.shape[1])).to(self.XtY)
            self.XtY = torch.cat((self.XtY, tail), dim=1)

        # Update sufficient statistics
        self.XtX += X.T @ X
        self.XtY += X.T @ Y

    @torch.no_grad()
    def update(self) -> None:
        """
        Compute final weights from sufficient statistics.

        W = (XtX + gamma * I)^{-1} @ XtY
        """
        reg_matrix = torch.eye(self.XtX.shape[0]).to(self.XtX) / self.gamma
        self.weight = torch.inverse(self.XtX + reg_matrix) @ self.XtY
        super().update()
