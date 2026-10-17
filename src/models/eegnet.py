"""Compact EEGNet architecture for multi-channel pre-movement EEG decoding.

Based on Lawhern et al. (2018), "EEGNet: a compact convolutional neural network for
EEG-based brain-computer interfaces", J. Neural Eng.
Compact architecture (~1.5k parameters) designed to minimize overfitting on small EEG datasets.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin


class CompactEEGNetModule(nn.Module):
    """PyTorch implementation of Compact EEGNet."""

    def __init__(
        self,
        n_channels: int = 26,
        n_samples: int = 40,
        n_classes: int = 2,
        F1: int = 4,
        D: int = 2,
        F2: int = 8,
        temporal_kernel: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        # Block 1: Temporal Conv + Depthwise Spatial Conv
        self.temporal_conv = nn.Conv2d(
            1, F1, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(
            F1, F2, kernel_size=(n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable Conv (Depthwise + Pointwise)
        self.sep_depthwise = nn.Conv2d(
            F2, F2, kernel_size=(1, 8), padding=(0, 4), groups=F2, bias=False
        )
        self.sep_pointwise = nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop2 = nn.Dropout(dropout)

        # Calculate flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self._forward_features(dummy)
            self.flat_dim = out.view(1, -1).size(1)

        self.classifier = nn.Linear(self.flat_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = x.flatten(1)
        return self.classifier(x)


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper around CompactEEGNetModule.
    
    Expects 3D input: (n_trials, n_samples, n_channels).
    """

    def __init__(
        self,
        epochs: int = 35,
        batch_size: int = 32,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.model_: Optional[CompactEEGNetModule] = None
        self.classes_: np.ndarray = np.array([0, 1])

    def _prepare_tensor(self, X: np.ndarray) -> torch.Tensor:
        # X is (n_trials, n_samples, n_channels) -> convert to (n_trials, 1, n_channels, n_samples)
        X_trans = np.transpose(X, (0, 2, 1))  # (n_trials, n_channels, n_samples)
        X_trans = np.expand_dims(X_trans, axis=1)  # (n_trials, 1, n_channels, n_samples)
        return torch.tensor(X_trans, dtype=torch.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_trials, n_samples, n_channels = X.shape

        self.model_ = CompactEEGNetModule(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=len(self.classes_),
        )
        self.model_.train()

        tensor_X = self._prepare_tensor(X)
        tensor_y = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model_(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("EEGNet must be fitted before predict_proba.")
        self.model_.eval()
        with torch.no_grad():
            tensor_X = self._prepare_tensor(X)
            logits = self.model_(tensor_X)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
