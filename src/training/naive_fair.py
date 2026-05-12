"""
Naive-FAIR baseline: trains fairness-constrained model on corrupted data
without robust reweighting (special case of DRO-FAIR with ρ=0).

Matches paper structure: full-batch training with fairness constraints.
Uses standard BCE (not tilted) — the tilted risk is specific to DRO-FAIR.

CRITICAL FIXES:
1. Full-batch training (no minibatch) so fairness gradients flow correctly.
2. Removed torch.no_grad() around fairness computation.
3. Dual ascent ONCE per epoch (not per minibatch).
4. Fixed τ: use σ(τ·logits) [multiply] not σ(logits/τ) [divide].
5. Use τ=1 for training to maintain gradient flow (τ=100 for evaluation only).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


class NaiveFairTrainer:
    """Naive-FAIR trainer enforcing DP + IF on corrupted data."""

    def __init__(self, model, device='cpu', lr_theta=1e-3, lr_lambda=5e-3,
                 lambda_max=10.0, tau=100.0, k=5, gamma=0.0,
                 epochs=50, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.lr_theta = lr_theta
        self.lr_lambda = lr_lambda
        self.lambda_max = lambda_max
        self.tau = tau
        self.k = k
        self.gamma = gamma
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.n_samples = None

    def _build_knn_graph(self, X):
        """Precompute k-NN graph and neighbor pair tensors for fast IF."""
        n = len(X)
        k_eff = min(self.k, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(X)
        distances, indices = nbrs.kneighbors(X)

        edges_i = []
        edges_j = []
        edge_dists = []
        for i in range(n):
            for idx in range(1, k_eff + 1):
                j = indices[i, idx]
                if j < n:
                    edges_i.append(i)
                    edges_j.append(j)
                    edge_dists.append(distances[i, idx])

        return (
            torch.tensor(edges_i, dtype=torch.long, device=self.device),
            torch.tensor(edges_j, dtype=torch.long, device=self.device),
            torch.tensor(edge_dists, dtype=torch.float32, device=self.device)
        )

    def _compute_dp_loss(self, h_tilde, a):
        """Compute DP violation: |E[h|A=1] - E[h|A=0]|."""
        group_rates = []
        for j in [0, 1]:
            mask = a == j
            if mask.sum() > 0:
                rate = h_tilde[mask].mean()
            else:
                rate = torch.tensor(0.0, device=self.device)
            group_rates.append(rate)
        return torch.abs(group_rates[1] - group_rates[0])

    def _compute_if_loss(self, h_tilde, edge_i, edge_j, edge_dists):
        """Compute IF violation using precomputed k-NN edges.

        Paper Algorithm 1, Line 12 (without reweighting, i.e., uniform p):
        g_IF = 1/(n-1) Σ_{i=1}^n Σ_{j∈N(i)} (|h̃_i - h̃_j| - d_{ij} - γ)_+
        """
        n = self.n_samples
        if len(edge_i) == 0 or n <= 1:
            return torch.tensor(0.0, device=self.device)

        h_i = h_tilde[edge_i]
        h_j = h_tilde[edge_j]
        violations = F.relu(torch.abs(h_i - h_j) - edge_dists - self.gamma)
        # Divide by (n-1) to match paper scaling
        return violations.sum() / (n - 1)

    def fit(self, X, y, a, X_val=None, y_val=None, a_val=None, verbose=False):
        """Train Naive-FAIR model with full-batch SGD."""
        n = len(X)
        self.n_samples = n

        # Convert to tensors
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device)

        # Optimizer for θ
        opt_theta = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr_theta, weight_decay=self.weight_decay
        )

        # Lagrange multipliers
        lambda_dp = torch.tensor(1.0, device=self.device)
        lambda_if = torch.tensor(1.0, device=self.device)

        # Precompute k-NN edges
        edge_i, edge_j, edge_dists = self._build_knn_graph(X)

        history = {'train_loss': [], 'val_acc': [], 'val_dp': [], 'val_if': []}

        for epoch in range(self.epochs):
            self.model.train()

            # Forward pass (full batch) — gradients flow for fairness constraints
            logits = self.model(X_t)
            # CRITICAL FIX: Paper uses σ(τ·f_θ(x)) [MULTIPLY]
            h_tilde = torch.sigmoid(logits * self.tau)

            # Standard BCE on full data
            cls_loss = F.binary_cross_entropy_with_logits(logits, y_t)

            # Fairness constraints — gradients flow through h_tilde
            g_dp = self._compute_dp_loss(h_tilde, a_t)
            g_if = self._compute_if_loss(h_tilde, edge_i, edge_j, edge_dists)

            # Total Lagrangian
            total_loss = cls_loss + lambda_dp * g_dp + lambda_if * g_if

            # Update θ
            opt_theta.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt_theta.step()

            # Dual ascent ONCE per epoch
            with torch.no_grad():
                lambda_dp = torch.clamp(
                    lambda_dp + self.lr_lambda * g_dp, 0, self.lambda_max
                )
                lambda_if = torch.clamp(
                    lambda_if + self.lr_lambda * g_if, 0, self.lambda_max
                )

            history['train_loss'].append(total_loss.item())

            # Validation every 5 epochs
            if X_val is not None and (epoch + 1) % 5 == 0:
                from src.evaluation.metrics import compute_metrics_torch
                metrics = compute_metrics_torch(
                    self.model, X_val, y_val, a_val,
                    device=self.device, temperature=self.tau, k=self.k, gamma=self.gamma
                )
                history['val_acc'].append(metrics['accuracy'])
                history['val_dp'].append(metrics['dp_violation'])
                history['val_if'].append(metrics['if_violation'])

                if verbose:
                    print(
                        f"Epoch {epoch+1}/{self.epochs}: loss={total_loss.item():.4f}, "
                        f"val_acc={metrics['accuracy']:.4f}, "
                        f"val_dp={metrics['dp_violation']:.4f}, "
                        f"val_if={metrics['if_violation']:.4f}"
                    )

        return history

    def predict(self, X):
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            probs = torch.sigmoid(self.model(X_t))
            return (probs >= 0.5).cpu().numpy()
