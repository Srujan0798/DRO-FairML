"""
DRO-FAIR trainer (Algorithm 1 from the paper).
Implements min-max Lagrangian with corruption-calibrated TV uncertainty sets.

Key implementation details matching the paper exactly:
- Minibatch SGD for θ (L_tilt computed per-minibatch)
- Full-batch g_DP and g_IF (fairness constraints over all samples)
- K=10 inner projected gradient steps on reweighting vectors
- Adam-style updates for p weights (per paper line 1793)
- Exact tilted risk: L_tilt = β * log(mean(exp(ℓ/β)))
- IF scaling: divide by (n-1) as per Algorithm 1 line 12
- Temperature τ tuned by corruption level (τ=1 at α≥0.4 per paper)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from src.utils.projections import project_simplex_l1_ball


class DroFairTrainer:
    """DRO-FAIR trainer with robust fairness guarantees."""

    def __init__(self, model, alpha, device='cpu', lr_theta=1e-3, lr_lambda=5e-3,
                 lr_p=5e-3, lambda_max=10.0, tau=100.0, beta=5.0, k=5, gamma=0.0,
                 K_inner=10, batch_size=256, epochs=50, weight_decay=1e-4,
                 use_dp=True, use_if=True, adam_p=True):
        """
        Args:
            model: PyTorch classifier (outputs raw logits)
            alpha: corruption rate (used for radius calibration)
            device: 'cpu' or 'cuda'
            lr_theta: learning rate for model parameters θ
            lr_lambda: learning rate for Lagrange multipliers λ
            lr_p: learning rate for importance weights p
            lambda_max: clamp λ to [0, lambda_max]
            tau: temperature for soft predictions σ(τ·f(x))
            beta: tilting parameter for exponential tilting / CVaR
            k: number of nearest neighbors for IF approximation
            gamma: slack parameter for metric fairness
            K_inner: number of inner PGD steps on p (paper uses K=10)
            batch_size: minibatch size for θ updates
            epochs: number of training epochs
            weight_decay: L2 regularization for AdamW
            use_dp: whether to enforce DP constraint
            use_if: whether to enforce IF constraint
            adam_p: use Adam for p-updates (paper line 1793)
        """
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
        self.adam_p = adam_p

        # Corruption-calibrated radii (computed in fit)
        self.rho_dp = None
        self.rho_if = None
        self.n_samples = None

        # Centers for projection (set in fit)
        self.p_dp_center = {}
        self.p_if_center = None

    def _compute_radii(self, a):
        """Compute corruption-calibrated TV radii (Equation 16 in paper)."""
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
        """Project group weights onto simplex ∩ l1-ball.
        TV distance: ||P - Q||_1 ≤ 2ρ, so L1-ball radius = 2ρ.
        """
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
        """Compute weighted DP violation with group-specific weights.
        g_DP = |h̄_1 - h̄_0| where h̄_j = Σ_i p̃_{j,i} h̃_i for i in group j.
        """
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
        """Compute weighted IF violation using precomputed k-NN edges.

        Paper Algorithm 1, Line 12:
        g_IF = 1/(n-1) Σ_{i=1}^n Σ_{j∈N(i)} (p̃_IF,i + p̃_IF,j)/2 * (|h̃_i - h̃_j| - d_{ij} - γ)_+

        CRITICAL FIX: divide by (n-1), NOT by number of edges.
        """
        n = self.n_samples
        if len(edge_i) == 0 or n <= 1:
            return torch.tensor(0.0, device=self.device)

        h_i = h_tilde[edge_i]
        h_j = h_tilde[edge_j]
        p_i = p_if[edge_i]
        p_j = p_if[edge_j]
        weights = (p_i + p_j) / 2.0
        violations = F.relu(torch.abs(h_i - h_j) - edge_dists - self.gamma)
        # Paper divides by (n-1), not by number of edges
        return (weights * violations).sum() / (n - 1)

    def _compute_tilted_loss(self, per_sample_loss):
        """Compute exact tilted risk L_tilt = β * log(mean(exp(ℓ/β))).

        Paper Algorithm 1, Line 9:
        L_tilt ← β log( 1/|M| Σ_{i∈M} exp(ℓ_i/β) )

        Numerically stable implementation using logsumexp.
        """
        m = len(per_sample_loss)
        if m == 0:
            return torch.tensor(0.0, device=self.device)
        # β * (logsumexp(ℓ/β) - log(m))
        return self.beta * (
            torch.logsumexp(per_sample_loss / self.beta, dim=0)
            - torch.log(torch.tensor(m, dtype=torch.float32, device=self.device))
        )

    def _update_p_weights(self, h_tilde, a_t, group_mask_dict, p_dp_dict, p_if,
                          edge_i, edge_j, edge_dists, lambda_dp, lambda_if, adam_states):
        """Inner maximization: projected gradient ascent on importance weights.

        Paper Algorithm 1, Lines 20-23:
        for s = 1 to K do
          p̃_j ← Proj_{U_j}(p̃_j + η_p ∇_{p̃_j} g_DP)
          p̃_IF ← Proj_{U_IF}(p̃_IF + η_p ∇_{p̃_IF} g_IF)
        end for

        Paper line 1793: "Adam for p̃-updates".
        """
        for _ in range(self.K_inner):
            # --- DP weights update ---
            if self.use_dp:
                for j in [0, 1]:
                    p_j_tensor = p_dp_dict[j].clone().detach().requires_grad_(True)
                    # Create temporary dict with only j's tensor as grad-enabled
                    p_temp = {0: p_dp_dict[0].detach(), 1: p_dp_dict[1].detach()}
                    p_temp[j] = p_j_tensor

                    dp_loss = self._compute_dp_loss_weighted(
                        h_tilde, a_t, p_temp, group_mask_dict
                    )
                    dp_loss.backward()
                    grad = p_j_tensor.grad

                    if grad is None:
                        continue

                    with torch.no_grad():
                        if self.adam_p:
                            state = adam_states['dp'][j]
                            state['t'] += 1
                            state['m'] = 0.9 * state['m'] + 0.1 * grad
                            state['v'] = 0.999 * state['v'] + 0.001 * (grad ** 2)
                            m_hat = state['m'] / (1 - 0.9 ** state['t'])
                            v_hat = state['v'] / (1 - 0.999 ** state['t'])
                            step = self.lr_p * m_hat / (torch.sqrt(v_hat) + 1e-8)
                        else:
                            step = self.lr_p * grad

                        p_dp_dict[j] = p_j_tensor + step
                        p_dp_dict[j] = self._project_dp_weights(
                            p_dp_dict[j], self.p_dp_center[j], self.rho_dp[j]
                        )

            # --- IF weights update ---
            if self.use_if:
                p_if_tensor = p_if.clone().detach().requires_grad_(True)
                if_loss = self._compute_if_loss_weighted(
                    h_tilde, p_if_tensor, edge_i, edge_j, edge_dists
                )
                if_loss.backward()
                grad = p_if_tensor.grad

                if grad is not None:
                    with torch.no_grad():
                        if self.adam_p:
                            state = adam_states['if']
                            state['t'] += 1
                            state['m'] = 0.9 * state['m'] + 0.1 * grad
                            state['v'] = 0.999 * state['v'] + 0.001 * (grad ** 2)
                            m_hat = state['m'] / (1 - 0.9 ** state['t'])
                            v_hat = state['v'] / (1 - 0.999 ** state['t'])
                            step = self.lr_p * m_hat / (torch.sqrt(v_hat) + 1e-8)
                        else:
                            step = self.lr_p * grad

                        p_if = p_if_tensor + step
                        p_if = self._project_if_weights(p_if, self.p_if_center, self.rho_if)

        return p_dp_dict, p_if

    def fit(self, X, y, a, X_val=None, y_val=None, a_val=None, verbose=False):
        """Train DRO-FAIR model (Algorithm 1).

        Structure:
        1. Full forward pass → compute h̃ for all samples
        2. Compute g_DP and g_IF on full data
        3. Update p weights (K inner steps)
        4. Loop over minibatches:
           a. Compute L_tilt on minibatch (exact β·log·mean·exp)
           b. Total loss = L_tilt + λ_DP·g_DP + λ_IF·g_IF
           c. Update θ via AdamW with gradient clipping
           d. Update λ via dual ascent
        """
        n = len(X)
        self.n_samples = n

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
        for j in [0, 1]:
            nj = group_sizes[j]
            self.p_dp_center[j] = torch.ones(nj, device=self.device) / nj
        self.p_if_center = torch.ones(n, device=self.device) / n

        # Optimizer for θ (AdamW with gradient clipping per paper line 1793)
        opt_theta = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr_theta, weight_decay=self.weight_decay
        )

        # Lagrange multipliers
        lambda_dp = torch.tensor(1.0, device=self.device)
        lambda_if = torch.tensor(1.0, device=self.device)

        # Precompute k-NN edges
        edge_i, edge_j, edge_dists = self._build_knn_graph(X)

        # Adam states for p weights (paper line 1793)
        adam_states = {
            'dp': {
                j: {'m': torch.zeros(group_sizes[j], device=self.device),
                    'v': torch.zeros(group_sizes[j], device=self.device),
                    't': 0} for j in [0, 1]
            },
            'if': {'m': torch.zeros(n, device=self.device),
                   'v': torch.zeros(n, device=self.device),
                   't': 0}
        }

        history = {'train_loss': [], 'val_acc': [], 'val_dp': [], 'val_if': []}

        for epoch in range(self.epochs):
            self.model.train()

            # === Full forward pass for fairness constraints ===
            # h̃_i = σ(τ · f_θ(x_i)) for all i (Algorithm 1, Line 7 extended to all samples)
            with torch.no_grad():
                logits_full = self.model(X_t)
                h_tilde_full = torch.sigmoid(logits_full / self.tau)

            # Compute g_DP and g_IF on full data (Algorithm 1, Lines 10-12)
            g_dp = self._compute_dp_loss_weighted(
                h_tilde_full, a_t, p_dp_dict, group_mask_dict
            ) if self.use_dp else torch.tensor(0.0, device=self.device)

            g_if = self._compute_if_loss_weighted(
                h_tilde_full, p_if, edge_i, edge_j, edge_dists
            ) if self.use_if else torch.tensor(0.0, device=self.device)

            # === Inner maximization: update p̃ via projected gradient ascent ===
            # (Algorithm 1, Lines 19-23)
            p_dp_dict, p_if = self._update_p_weights(
                h_tilde_full, a_t, group_mask_dict, p_dp_dict, p_if,
                edge_i, edge_j, edge_dists, lambda_dp, lambda_if, adam_states
            )

            # === Outer minimization: minibatch SGD for θ ===
            # (Algorithm 1, Lines 5-15)
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = perm[start:end]

                # Forward pass on minibatch
                logits_batch = self.model(X_t[batch_idx])

                # Classification loss: exact tilted risk (Algorithm 1, Line 9)
                per_sample_loss = F.binary_cross_entropy_with_logits(
                    logits_batch, y_t[batch_idx], reduction='none'
                )
                L_tilt = self._compute_tilted_loss(per_sample_loss)

                # Total Lagrangian (Algorithm 1, Line 14)
                total_loss = L_tilt
                if self.use_dp:
                    total_loss = total_loss + lambda_dp * g_dp
                if self.use_if:
                    total_loss = total_loss + lambda_if * g_if

                # Update θ (Algorithm 1, Line 15)
                opt_theta.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt_theta.step()

                # Dual ascent: update λ (Algorithm 1, Lines 17-18)
                with torch.no_grad():
                    if self.use_dp:
                        lambda_dp = torch.clamp(
                            lambda_dp + self.lr_lambda * g_dp, 0, self.lambda_max
                        )
                    if self.use_if:
                        lambda_if = torch.clamp(
                            lambda_if + self.lr_lambda * g_if, 0, self.lambda_max
                        )

                epoch_loss += total_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history['train_loss'].append(avg_loss)

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
                        f"Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.4f}, "
                        f"val_acc={metrics['accuracy']:.4f}, "
                        f"val_dp={metrics['dp_violation']:.4f}, "
                        f"val_if={metrics['if_violation']:.4f}, "
                        f"lambda_dp={lambda_dp.item():.2f}, "
                        f"lambda_if={lambda_if.item():.2f}"
                    )

        return history

    def predict(self, X):
        """Make binary predictions."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            probs = torch.sigmoid(self.model(X_t))
            return (probs >= 0.5).cpu().numpy()
