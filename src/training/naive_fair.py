"""
Naive-FAIR baseline: trains fairness-constrained model on corrupted data
without robust reweighting (special case of DRO-FAIR with ρ=0).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


class NaiveFairTrainer:
    """Naive-FAIR trainer enforcing DP + IF on corrupted data."""
    
    def __init__(self, model, device='cpu', lr_theta=1e-3, lr_lambda=1e-2,
                 lambda_max=10.0, tau=100.0, beta=5.0, k=5, gamma=0.0,
                 batch_size=256, epochs=50, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.lr_theta = lr_theta
        self.lr_lambda = lr_lambda
        self.lambda_max = lambda_max
        self.tau = tau
        self.beta = beta
        self.k = k
        self.gamma = gamma
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        
    def _build_knn_graph(self, X):
        """Precompute k-NN graph and neighbor pair tensors for fast IF."""
        n = len(X)
        k_eff = min(self.k, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Build edge list (i, j) for all neighbor pairs
        edges_i = []
        edges_j = []
        edge_dists = []
        for i in range(n):
            for idx in range(1, k_eff + 1):  # skip self
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
        """Compute IF violation using precomputed k-NN edges."""
        if len(edge_i) == 0:
            return torch.tensor(0.0, device=self.device)
        
        h_i = h_tilde[edge_i]
        h_j = h_tilde[edge_j]
        violations = F.relu(torch.abs(h_i - h_j) - edge_dists - self.gamma)
        return violations.mean()
    
    def fit(self, X, y, a, X_val=None, y_val=None, a_val=None, verbose=False):
        """Train Naive-FAIR model."""
        n = len(X)
        
        # Convert to tensors
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device)
        
        # Optimizers
        opt_theta = torch.optim.AdamW(self.model.parameters(), lr=self.lr_theta, weight_decay=self.weight_decay)
        
        # Lagrange multipliers
        lambda_dp = torch.tensor(1.0, device=self.device, requires_grad=False)
        lambda_if = torch.tensor(1.0, device=self.device, requires_grad=False)
        
        # Precompute k-NN edges
        edge_i, edge_j, edge_dists = self._build_knn_graph(X)
        
        history = {'train_loss': [], 'val_acc': [], 'val_dp': [], 'val_if': []}
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # Full batch forward pass
            logits = self.model(X_t)
            
            # Classification loss (BCE)
            cls_loss = F.binary_cross_entropy_with_logits(logits, y_t)
            
            # Soft predictions for fairness
            h_tilde = torch.sigmoid(logits / self.tau)
            
            # DP violation
            dp_loss = self._compute_dp_loss(h_tilde, a_t)
            
            # IF violation
            if_loss = self._compute_if_loss(h_tilde, edge_i, edge_j, edge_dists)
            
            # Total Lagrangian loss
            total_loss = cls_loss + lambda_dp * dp_loss + lambda_if * if_loss
            
            # Update theta
            opt_theta.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt_theta.step()
            
            # Dual ascent on lambda
            with torch.no_grad():
                lambda_dp = torch.clamp(lambda_dp + self.lr_lambda * dp_loss, 0, self.lambda_max)
                lambda_if = torch.clamp(lambda_if + self.lr_lambda * if_loss, 0, self.lambda_max)
            
            history['train_loss'].append(total_loss.item())
            
            # Validation
            if X_val is not None and (epoch + 1) % 5 == 0:
                from src.evaluation.metrics import compute_metrics_torch
                metrics = compute_metrics_torch(self.model, X_val, y_val, a_val,
                                               device=self.device, temperature=self.tau, k=self.k, gamma=self.gamma)
                history['val_acc'].append(metrics['accuracy'])
                history['val_dp'].append(metrics['dp_violation'])
                history['val_if'].append(metrics['if_violation'])
                
                if verbose:
                    print(f"Epoch {epoch+1}/{self.epochs}: loss={total_loss.item():.4f}, "
                          f"val_acc={metrics['accuracy']:.4f}, val_dp={metrics['dp_violation']:.4f}, "
                          f"val_if={metrics['if_violation']:.4f}")
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            probs = torch.sigmoid(self.model(X_t))
            return (probs >= 0.5).cpu().numpy()
