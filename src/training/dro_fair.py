"""
DRO-FAIR trainer (Algorithm 1 from the paper).
Implements min-max Lagrangian with corruption-calibrated TV uncertainty sets.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from src.utils.projections import project_simplex_l1_ball


class DroFairTrainer:
    """DRO-FAIR trainer with robust fairness guarantees."""
    
    def __init__(self, model, alpha, device='cpu', lr_theta=1e-3, lr_lambda=1e-2,
                 lr_p=5e-3, lambda_max=10.0, tau=100.0, beta=5.0, k=5, gamma=0.0,
                 K_inner=5, batch_size=256, epochs=50, weight_decay=1e-4,
                 use_dp=True, use_if=True):
        self.model = model.to(device)
        self.device = device
        self.alpha = alpha
        self.lr_theta = lr_theta
        self.lr_lambda = lr_lambda
        self.lr_p = lr_p
        self.lambda_max = lambda_max
        self.tau = tau
        self.beta = beta
        self.k = k
        self.gamma = gamma
        self.K_inner = K_inner
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.use_dp = use_dp
        self.use_if = use_if
        
        # Corruption-calibrated radii (computed in fit)
        self.rho_dp = None
        self.rho_if = None
    
    def _compute_radii(self, a):
        """Compute corruption-calibrated TV radii."""
        n = len(a)
        pi = np.array([np.mean(a == j) for j in [0, 1]])
        
        # ρ_DP,j = α / ((1-α)π_j + α)
        rho_dp = []
        for j in [0, 1]:
            denom = (1 - self.alpha) * pi[j] + self.alpha
            rho_dp.append(self.alpha / denom if denom > 0 else 1.0)
        
        # ρ_IF = 2α - α²
        rho_if = 2 * self.alpha - self.alpha ** 2
        
        return rho_dp, rho_if
    
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
    
    def _init_weights(self, n, group_mask_dict):
        """Initialize importance weights uniformly per group and globally."""
        p_dp = {}
        for j in [0, 1]:
            mask = group_mask_dict[j]
            nj = mask.sum().item()
            p_dp[j] = torch.ones(nj, device=self.device) / nj
        
        p_if = torch.ones(n, device=self.device) / n
        return p_dp, p_if
    
    def _project_dp_weights(self, p, center, radius):
        """Project group weights onto simplex ∩ l1-ball."""
        p_np = p.detach().cpu().numpy()
        center_np = center.detach().cpu().numpy()
        proj = project_simplex_l1_ball(p_np, center_np, 2 * radius, max_iter=50, tol=1e-5)
        return torch.tensor(proj, dtype=p.dtype, device=p.device)
    
    def _project_if_weights(self, p, center, radius):
        """Project global IF weights onto simplex ∩ l1-ball."""
        p_np = p.detach().cpu().numpy()
        center_np = center.detach().cpu().numpy()
        proj = project_simplex_l1_ball(p_np, center_np, 2 * radius, max_iter=50, tol=1e-5)
        return torch.tensor(proj, dtype=p.dtype, device=p.device)
    
    def _compute_dp_loss_weighted(self, h_tilde, a, p_dp_dict, group_mask_dict):
        """Compute weighted DP violation with group-specific weights."""
        group_rates = []
        for j in [0, 1]:
            mask = group_mask_dict[j]
            if mask.sum() > 0:
                h_group = h_tilde[mask]
                weights = p_dp_dict[j]
                rate = (weights * h_group).sum()
            else:
                rate = torch.tensor(0.0, device=self.device)
            group_rates.append(rate)
        return torch.abs(group_rates[1] - group_rates[0])
    
    def _compute_if_loss_weighted(self, h_tilde, p_if, edge_i, edge_j, edge_dists):
        """Compute weighted IF violation using precomputed k-NN edges."""
        if len(edge_i) == 0:
            return torch.tensor(0.0, device=self.device)
        
        h_i = h_tilde[edge_i]
        h_j = h_tilde[edge_j]
        p_i = p_if[edge_i]
        p_j = p_if[edge_j]
        weights = (p_i + p_j) / 2.0
        violations = F.relu(torch.abs(h_i - h_j) - edge_dists - self.gamma)
        return (weights * violations).mean()
    
    def fit(self, X, y, a, X_val=None, y_val=None, a_val=None, verbose=False):
        """Train DRO-FAIR model (Algorithm 1)."""
        n = len(X)
        
        # Convert to tensors
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device)
        
        # Compute radii
        self.rho_dp, self.rho_if = self._compute_radii(a)
        
        # Group masks
        group_mask_dict = {j: (a_t == j) for j in [0, 1]}
        group_sizes = {j: group_mask_dict[j].sum().item() for j in [0, 1]}
        
        # Initialize weights
        p_dp_dict, p_if = self._init_weights(n, group_mask_dict)
        
        # Uniform centers for projection
        p_dp_center = {}
        for j in [0, 1]:
            nj = group_sizes[j]
            p_dp_center[j] = torch.ones(nj, device=self.device) / nj
        p_if_center = torch.ones(n, device=self.device) / n
        
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
            
            # === Outer minimization: update θ ===
            logits = self.model(X_t)
            
            # Classification loss (tilted BCE)
            per_sample_loss = F.binary_cross_entropy_with_logits(logits, y_t, reduction='none')
            max_loss = per_sample_loss.max()
            weights = F.softmax((per_sample_loss - max_loss) / self.beta, dim=0)
            cls_loss = (weights * per_sample_loss).sum()
            
            # Soft predictions for fairness
            h_tilde = torch.sigmoid(logits / self.tau)
            
            # DP violation (worst-case within uncertainty set)
            dp_loss = self._compute_dp_loss_weighted(h_tilde, a_t, p_dp_dict, group_mask_dict) if self.use_dp else torch.tensor(0.0, device=self.device)
            
            # IF violation (worst-case within uncertainty set)
            if_loss = self._compute_if_loss_weighted(h_tilde, p_if, edge_i, edge_j, edge_dists) if self.use_if else torch.tensor(0.0, device=self.device)
            
            # Total Lagrangian
            total_loss = cls_loss + (lambda_dp * dp_loss if self.use_dp else 0.0) + (lambda_if * if_loss if self.use_if else 0.0)
            
            # Update θ
            opt_theta.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt_theta.step()
            
            # === Dual ascent: update λ ===
            with torch.no_grad():
                if self.use_dp:
                    lambda_dp = torch.clamp(lambda_dp + self.lr_lambda * dp_loss, 0, self.lambda_max)
                if self.use_if:
                    lambda_if = torch.clamp(lambda_if + self.lr_lambda * if_loss, 0, self.lambda_max)
            
            # === Inner maximization: update p̃ via projected gradient ascent ===
            with torch.enable_grad():
                for s in range(self.K_inner):
                    total_inner = torch.tensor(0.0, device=self.device, requires_grad=True)
                    
                    if self.use_dp:
                        p_dp_grad = {j: p_dp_dict[j].clone().detach().requires_grad_(True)
                                     for j in [0, 1]}
                        dp_loss_inner = self._compute_dp_loss_weighted(
                            h_tilde.detach(), a_t, p_dp_grad, group_mask_dict
                        )
                        total_inner = total_inner + lambda_dp * dp_loss_inner
                        
                        # Backprop and update DP weights
                        if total_inner.requires_grad:
                            total_inner.backward(retain_graph=True)
                        with torch.no_grad():
                            for j in [0, 1]:
                                if p_dp_grad[j].grad is not None:
                                    p_dp_dict[j] = p_dp_dict[j] + self.lr_p * p_dp_grad[j].grad
                                    p_dp_dict[j] = self._project_dp_weights(
                                        p_dp_dict[j], p_dp_center[j], self.rho_dp[j]
                                    )
                    
                    if self.use_if:
                        p_if_grad = p_if.clone().detach().requires_grad_(True)
                        if_loss_inner = self._compute_if_loss_weighted(
                            h_tilde.detach(), p_if_grad, edge_i, edge_j, edge_dists
                        )
                        total_inner = total_inner + lambda_if * if_loss_inner
                        
                        if total_inner.requires_grad:
                            total_inner.backward()
                        with torch.no_grad():
                            if p_if_grad.grad is not None:
                                p_if = p_if + self.lr_p * p_if_grad.grad
                                p_if = self._project_if_weights(p_if, p_if_center, self.rho_if)
            
            history['train_loss'].append(total_loss.item())
            
            # Validation
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
                    print(f"Epoch {epoch+1}/{self.epochs}: loss={total_loss.item():.4f}, "
                          f"val_acc={metrics['accuracy']:.4f}, val_dp={metrics['dp_violation']:.4f}, "
                          f"val_if={metrics['if_violation']:.4f}, "
                          f"lambda_dp={lambda_dp.item():.2f}, lambda_if={lambda_if.item():.2f}")
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            probs = torch.sigmoid(self.model(X_t))
            return (probs >= 0.5).cpu().numpy()
