"""Machine learning models for continuous 3D kinematic trajectory and endpoint forecasting."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.trajectory.schema import TrajectorySample
from src.trajectory.features import extract_kinematic_features


class RidgeTrajectoryForecaster:
    """Multi-output Ridge Regression forecaster predicting relative displacements from observed prefix."""

    def __init__(self, alpha: float = 10.0):
        self.name = "ridge_regression"
        self.alpha = alpha
        self.scaler: Optional[StandardScaler] = None
        self.endpoint_model: Optional[Ridge] = None
        self.horizon_models: Dict[int, Ridge] = {}
        self.horizon_point_models: Dict[int, Ridge] = {}

    def fit(self, train_samples: List[TrajectorySample]):
        """Fit feature scalers and regression weights strictly on training fold."""
        X_train = np.array([extract_kinematic_features(s) for s in train_samples], dtype=np.float32)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # 1. Fit endpoint displacement model: Delta p_end = p_end - p(t_obs)
        y_end = np.array([
            s.actual_endpoint - s.prefix_positions[-1] for s in train_samples
        ], dtype=np.float32)
        self.endpoint_model = Ridge(alpha=self.alpha)
        self.endpoint_model.fit(X_scaled, y_end)

        # 2. Fit horizon models for each horizon
        for h in [100, 200, 300, 500]:
            # Trajectory over horizon: shape (N, n_steps * 3)
            n_steps = int(h / 10)
            valid_train = [s for s in train_samples if h in s.horizon_trajectories]
            if not valid_train:
                continue

            X_h = np.array([extract_kinematic_features(s) for s in valid_train], dtype=np.float32)
            X_h_scaled = self.scaler.transform(X_h)

            y_traj_disp = np.array([
                (s.horizon_trajectories[h] - s.prefix_positions[-1][None, :]).flatten()
                for s in valid_train
            ], dtype=np.float32)

            m_traj = Ridge(alpha=self.alpha)
            m_traj.fit(X_h_scaled, y_traj_disp)
            self.horizon_models[h] = m_traj

            # Horizon final point displacement
            y_pt_disp = np.array([
                s.horizon_points[h] - s.prefix_positions[-1] for s in valid_train
            ], dtype=np.float32)
            m_pt = Ridge(alpha=self.alpha)
            m_pt.fit(X_h_scaled, y_pt_disp)
            self.horizon_point_models[h] = m_pt

    def predict_endpoint(self, sample: TrajectorySample) -> np.ndarray:
        """Predict absolute 3D endpoint: p(t_obs) + Delta p_end."""
        feats = extract_kinematic_features(sample)[None, :]
        feats_scaled = self.scaler.transform(feats)
        delta_end = self.endpoint_model.predict(feats_scaled)[0]
        return (sample.prefix_positions[-1] + delta_end).astype(np.float32)

    def predict_horizon(self, sample: TrajectorySample, horizon_ms: int) -> np.ndarray:
        """Predict multi-step future trajectory over horizon: p(t_obs) + Delta p(tau)."""
        if horizon_ms not in self.horizon_models:
            n_steps = int(horizon_ms / 10)
            return np.tile(sample.prefix_positions[-1], (n_steps, 1))

        feats = extract_kinematic_features(sample)[None, :]
        feats_scaled = self.scaler.transform(feats)
        delta_flat = self.horizon_models[horizon_ms].predict(feats_scaled)[0]
        n_steps = int(horizon_ms / 10)
        delta_traj = delta_flat.reshape(n_steps, 3)
        pred_traj = sample.prefix_positions[-1][None, :] + delta_traj
        return pred_traj.astype(np.float32)

    def predict_horizon_point(self, sample: TrajectorySample, horizon_ms: int) -> np.ndarray:
        """Predict 3D position at horizon_ms."""
        if horizon_ms not in self.horizon_point_models:
            return sample.prefix_positions[-1].copy()

        feats = extract_kinematic_features(sample)[None, :]
        feats_scaled = self.scaler.transform(feats)
        delta_pt = self.horizon_point_models[horizon_ms].predict(feats_scaled)[0]
        return (sample.prefix_positions[-1] + delta_pt).astype(np.float32)


class MLPTrajectoryModule(nn.Module):
    """Compact multi-layer perceptron for kinematic displacement forecasting."""

    def __init__(self, in_dim: int = 95, out_dim: int = 3, hidden_dims: Tuple[int, int] = (64, 32)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPTrajectoryForecaster:
    """Compact PyTorch MLP forecaster for endpoint and multi-horizon trajectory forecasting."""

    def __init__(self, epochs: int = 50, lr: float = 0.005, weight_decay: float = 1e-4, random_state: int = 42):
        self.name = "compact_mlp"
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.endpoint_net: Optional[MLPTrajectoryModule] = None
        self.horizon_nets: Dict[int, MLPTrajectoryModule] = {}
        self.horizon_point_nets: Dict[int, MLPTrajectoryModule] = {}

    def _train_module(self, X: np.ndarray, y: np.ndarray, out_dim: int) -> MLPTrajectoryModule:
        torch.manual_seed(self.random_state)
        module = MLPTrajectoryModule(in_dim=X.shape[1], out_dim=out_dim)
        optimizer = optim.Adam(module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        module.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = module(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()

        module.eval()
        return module

    def fit(self, train_samples: List[TrajectorySample]):
        """Train MLP networks on training fold."""
        X_train = np.array([extract_kinematic_features(s) for s in train_samples], dtype=np.float32)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # 1. Endpoint net
        y_end = np.array([
            s.actual_endpoint - s.prefix_positions[-1] for s in train_samples
        ], dtype=np.float32)
        self.endpoint_net = self._train_module(X_scaled, y_end, out_dim=3)

        # 2. Horizon nets
        for h in [100, 200, 300, 500]:
            valid_train = [s for s in train_samples if h in s.horizon_trajectories]
            if not valid_train:
                continue

            X_h = np.array([extract_kinematic_features(s) for s in valid_train], dtype=np.float32)
            X_h_scaled = self.scaler.transform(X_h)

            n_steps = int(h / 10)
            y_traj_disp = np.array([
                (s.horizon_trajectories[h] - s.prefix_positions[-1][None, :]).flatten()
                for s in valid_train
            ], dtype=np.float32)
            self.horizon_nets[h] = self._train_module(X_h_scaled, y_traj_disp, out_dim=n_steps * 3)

            y_pt_disp = np.array([
                s.horizon_points[h] - s.prefix_positions[-1] for s in valid_train
            ], dtype=np.float32)
            self.horizon_point_nets[h] = self._train_module(X_h_scaled, y_pt_disp, out_dim=3)

    def predict_endpoint(self, sample: TrajectorySample) -> np.ndarray:
        feats = extract_kinematic_features(sample)[None, :]
        feats_scaled = self.scaler.transform(feats)
        with torch.no_grad():
            delta = self.endpoint_net(torch.tensor(feats_scaled, dtype=torch.float32)).numpy()[0]
        return (sample.prefix_positions[-1] + delta).astype(np.float32)

    def predict_horizon(self, sample: TrajectorySample, horizon_ms: int) -> np.ndarray:
        if horizon_ms not in self.horizon_nets:
            n_steps = int(horizon_ms / 10)
            return np.tile(sample.prefix_positions[-1], (n_steps, 1))

        feats = extract_kinematic_features(sample)[None, :]
        feats_scaled = self.scaler.transform(feats)
        with torch.no_grad():
            delta_flat = self.horizon_nets[horizon_ms](torch.tensor(feats_scaled, dtype=torch.float32)).numpy()[0]
        n_steps = int(horizon_ms / 10)
        delta_traj = delta_flat.reshape(n_steps, 3)
        return (sample.prefix_positions[-1][None, :] + delta_traj).astype(np.float32)

    def predict_horizon_point(self, sample: TrajectorySample, horizon_ms: int) -> np.ndarray:
        if horizon_ms not in self.horizon_point_nets:
            return sample.prefix_positions[-1].copy()

        feats = extract_kinematic_features(sample)[None, :]
        feats_scaled = self.scaler.transform(feats)
        with torch.no_grad():
            delta = self.horizon_point_nets[horizon_ms](torch.tensor(feats_scaled, dtype=torch.float32)).numpy()[0]
        return (sample.prefix_positions[-1] + delta).astype(np.float32)


class GRUTrajectoryModule(nn.Module):
    """Recurrent neural network consuming temporal kinematic prefix sequences."""

    def __init__(self, in_features: int = 6, hidden_dim: int = 32, out_dim: int = 3):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


class GRUTrajectoryForecaster:
    """Compact Recurrent GRU forecaster consuming sequence (20, 6) of positions and velocities."""

    def __init__(self, epochs: int = 50, lr: float = 0.005, random_state: int = 42):
        self.name = "compact_gru"
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.pos_mean: Optional[np.ndarray] = None
        self.pos_std: Optional[np.ndarray] = None
        self.vel_mean: Optional[np.ndarray] = None
        self.vel_std: Optional[np.ndarray] = None
        self.endpoint_net: Optional[GRUTrajectoryModule] = None
        self.horizon_point_nets: Dict[int, GRUTrajectoryModule] = {}
        self.horizon_nets: Dict[int, GRUTrajectoryModule] = {}

    def _prepare_sequences(self, samples: List[TrajectorySample]) -> np.ndarray:
        seqs = []
        for s in samples:
            p_rel = s.prefix_positions - s.prefix_positions[0][None, :]
            v = s.prefix_velocities
            seq = np.concatenate([p_rel, v], axis=1) # (20, 6)
            seqs.append(seq)
        return np.array(seqs, dtype=np.float32)

    def _normalize(self, seqs: np.ndarray) -> np.ndarray:
        norm_seqs = seqs.copy()
        norm_seqs[:, :, :3] = (norm_seqs[:, :, :3] - self.pos_mean) / (self.pos_std + 1e-8)
        norm_seqs[:, :, 3:] = (norm_seqs[:, :, 3:] - self.vel_mean) / (self.vel_std + 1e-8)
        return norm_seqs

    def fit(self, train_samples: List[TrajectorySample]):
        raw_seqs = self._prepare_sequences(train_samples)
        self.pos_mean = np.mean(raw_seqs[:, :, :3], axis=(0, 1))
        self.pos_std = np.std(raw_seqs[:, :, :3], axis=(0, 1))
        self.vel_mean = np.mean(raw_seqs[:, :, 3:], axis=(0, 1))
        self.vel_std = np.std(raw_seqs[:, :, 3:], axis=(0, 1))

        norm_seqs = self._normalize(raw_seqs)

        # 1. Endpoint net
        y_end = np.array([s.actual_endpoint - s.prefix_positions[-1] for s in train_samples], dtype=np.float32)
        torch.manual_seed(self.random_state)
        self.endpoint_net = GRUTrajectoryModule(in_features=6, hidden_dim=32, out_dim=3)
        opt = optim.Adam(self.endpoint_net.parameters(), lr=self.lr)
        crit = nn.MSELoss()

        X_tensor = torch.tensor(norm_seqs, dtype=torch.float32)
        y_tensor = torch.tensor(y_end, dtype=torch.float32)
        self.endpoint_net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            l = crit(self.endpoint_net(X_tensor), y_tensor)
            l.backward()
            opt.step()
        self.endpoint_net.eval()

        # 2. Horizons
        for h in [100, 200, 300, 500]:
            valid_train = [s for s in train_samples if h in s.horizon_trajectories]
            if not valid_train:
                continue

            raw_h_seqs = self._prepare_sequences(valid_train)
            norm_h = self._normalize(raw_h_seqs)

            n_steps = int(h / 10)
            y_traj = np.array([
                (s.horizon_trajectories[h] - s.prefix_positions[-1][None, :]).flatten()
                for s in valid_train
            ], dtype=np.float32)
            net_traj = GRUTrajectoryModule(in_features=6, hidden_dim=32, out_dim=n_steps * 3)
            opt_t = optim.Adam(net_traj.parameters(), lr=self.lr)
            net_traj.train()
            X_t = torch.tensor(norm_h, dtype=torch.float32)
            y_t = torch.tensor(y_traj, dtype=torch.float32)
            for _ in range(self.epochs):
                opt_t.zero_grad()
                loss = crit(net_traj(X_t), y_t)
                loss.backward()
                opt_t.step()
            net_traj.eval()
            self.horizon_nets[h] = net_traj

            y_pt = np.array([s.horizon_points[h] - s.prefix_positions[-1] for s in valid_train], dtype=np.float32)
            net_pt = GRUTrajectoryModule(in_features=6, hidden_dim=32, out_dim=3)
            opt_p = optim.Adam(net_pt.parameters(), lr=self.lr)
            net_pt.train()
            y_p = torch.tensor(y_pt, dtype=torch.float32)
            for _ in range(self.epochs):
                opt_p.zero_grad()
                loss = crit(net_pt(X_t), y_p)
                loss.backward()
                opt_p.step()
            net_pt.eval()
            self.horizon_point_nets[h] = net_pt

    def predict_endpoint(self, sample: TrajectorySample) -> np.ndarray:
        raw_seq = self._prepare_sequences([sample])
        norm_seq = self._normalize(raw_seq)
        with torch.no_grad():
            delta = self.endpoint_net(torch.tensor(norm_seq, dtype=torch.float32)).numpy()[0]
        return (sample.prefix_positions[-1] + delta).astype(np.float32)

    def predict_horizon(self, sample: TrajectorySample, horizon_ms: int) -> np.ndarray:
        if horizon_ms not in self.horizon_nets:
            n_steps = int(horizon_ms / 10)
            return np.tile(sample.prefix_positions[-1], (n_steps, 1))

        raw_seq = self._prepare_sequences([sample])
        norm_seq = self._normalize(raw_seq)
        with torch.no_grad():
            delta_flat = self.horizon_nets[horizon_ms](torch.tensor(norm_seq, dtype=torch.float32)).numpy()[0]
        n_steps = int(horizon_ms / 10)
        delta_traj = delta_flat.reshape(n_steps, 3)
        return (sample.prefix_positions[-1][None, :] + delta_traj).astype(np.float32)

    def predict_horizon_point(self, sample: TrajectorySample, horizon_ms: int) -> np.ndarray:
        if horizon_ms not in self.horizon_point_nets:
            return sample.prefix_positions[-1].copy()

        raw_seq = self._prepare_sequences([sample])
        norm_seq = self._normalize(raw_seq)
        with torch.no_grad():
            delta = self.horizon_point_nets[horizon_ms](torch.tensor(norm_seq, dtype=torch.float32)).numpy()[0]
        return (sample.prefix_positions[-1] + delta).astype(np.float32)
