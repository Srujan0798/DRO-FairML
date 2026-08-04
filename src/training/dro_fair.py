"""
DRO-FAIR trainer (Algorithm 1 from the paper).
Implements min-max Lagrangian with corruption-calibrated TV uncertainty sets.

CRITICAL FIX: Paper uses h̃ = σ(τ·f_θ(x)) [MULTIPLY], not σ(f_θ(x)/τ) [DIVIDE].
Multiply makes predictions sharp so fairness constraints are meaningful.

CANONICAL / MASTER_PLAN: fixed tau=1 (verified headline: DRO beats Naive on DP
at every α on Adult; old fragility was tau=100 artifact). Use epochs=60,
K_inner=10 (default here), lambda_init=0.0 (exposed ONLY for Q1 ablation),
lambda_max=1.5 (all datasets), step order θ→λ→p, inner-max ascends ∇g (not λ∇g).
Corruption always via FairnessTargetedPGD (adversarial); RandomCorruptor only baseline.

ALGORITHM FIX: Per paper Algorithm 1 (page 33), order is:
θ update (outer min) → λ dual ascent → inner maximization on p (K steps).
Inner max runs AFTER θ update so p adapts to the updated model.
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
                 lr_p=5e-3, lambda_max=1.5, tau=1.0, beta=5.0, k=5, gamma=0.0,
                 K_inner=10, epochs=60, weight_decay=1e-4,
                 use_dp=True, use_if=True, tau_warmup_epochs=15,
                 lambda_init=0.0, radii_mode='uniform', history_path=None,
                 radii_scale=1.0, radii_clamp=None, pi_shrinkage_k=0.0):
        self.model = model.to(device)
        self.device = device
        self.alpha = alpha
        self.lr_theta = lr_theta
        self.lr_lambda = lr_lambda
        self.lr_p = lr_p
        self.lambda_max = lambda_max
        # lambda_init: paper spec is 0.0. Exposed only for Kuldeep's Q1
        # hyperparameter ablation (try different initial dual values).
        self.lambda_init = lambda_init
        self.tau = tau
        self.beta = beta
        self.k = k
        self.gamma = gamma
        self.K_inner = K_inner
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.use_dp = use_dp
        self.use_if = use_if
        self.tau_warmup_epochs = tau_warmup_epochs
        # radii_mode (Kuldeep Q5): 'uniform' (default) uses the closed-form
        # bias correction pi_clean = (pi_obs - alpha) / (1 - 2*alpha), which
        # assumes the corruption budget is distributed uniformly across groups.
        # 'empirical' exploits the known coordinated 70%-minority attack
        # structure to invert the attribute-flip equations to recover the clean proportions
        # exactly. No true corruption mask is used (no oracle leak).
        if radii_mode not in ('uniform', 'empirical'):
            raise ValueError(f"radii_mode must be 'uniform' or 'empirical', got {radii_mode!r}")
        self.radii_mode = radii_mode
        # radii_scale (Agent N1): global multiplier on rho_dp/rho_if. Default 1.0
        # leaves the canonical closed form unchanged. Used to test whether DRO's
        # advantage is a function of the match between attack strength and radius
        # calibration (Kuldeep's first question, May 29). Provenance-recorded.
        self.radii_scale = float(radii_scale)
        # radii_clamp (Agent L2): per-group cap on rho_dp[j]. Default None leaves
        # canonical behavior. When set, no single group's radius exceeds the cap —
        # prevents the tiny minority group's radius from blowing up on imbalanced
        # data like LSAC (~90/10), where uniform mode produces rho_dp[minority]=1.0
        # (degenerate). This is HYPOTHESIS TESTING, not tuning-until-it-wins.
        self.radii_clamp = float(radii_clamp) if radii_clamp is not None else None
        # pi_shrinkage_k (LSAC-degeneracy hypothesis, alternative to radii_clamp):
        # additive (Laplace-style) shrinkage of pi_clean toward 0.5 before computing
        # rho_dp. pi_shrunk[j] = (pi_clean[j]*n_eff + k*0.5) / (n_eff + k), where n_eff
        # is the sample count that informed pi_clean. k=0 (default) is a no-op —
        # canonical behavior unchanged. Unlike radii_clamp (a hard post-hoc cap on
        # rho_dp itself), this smooths the INPUT proportion, which changes how the
        # cap is shaped as alpha varies rather than pinning a single fixed ceiling.
        # HYPOTHESIS TESTING, not tuning-until-it-wins: if the tiny minority group's
        # true proportion is genuinely small (not just noisily estimated), shrinkage
        # may fail to fix LSAC just as radii_clamp did — report honestly either way.
        self.pi_shrinkage_k = float(pi_shrinkage_k)
        self.history_path = history_path
        self.rho_dp = None
        self.rho_if = None
        self.n_samples = None

    def _compute_radii(self, a, a_val=None):
        """Compute corruption-calibrated TV radii with bias-corrected group proportions.

        FAIR VERSION: Uses clean validation data (a_val) to estimate true clean
        group proportions. Both Naive and DRO have access to clean validation data,
        so this is NOT an oracle leak.

        Kuldeep Q5: when a_val is unavailable, the trainer can still exploit
        the known attack structure. 'uniform' (default) uses the generic
        closed form pi_clean = (pi_obs - alpha) / (1 - 2*alpha). 'empirical'
        assumes a coordinated attack with 70% of the corruption budget hitting
        the minority group and the other 30% hitting the majority, and inverts
        the resulting attribute-flip equations to recover the clean proportions
        exactly. No true corruption mask is used (no oracle leak).

        NOTE (audit): group proportions computed for binary a in [0,1] (see pi_obs).
        Most datasets binary protected attr; metrics.compute_dp_violation supports >2.
        """
        n = len(a)
        pi_obs = np.array([np.mean(a == j) for j in [0, 1]])

        if self.radii_mode == 'empirical':
            # Q5: empirical mode DELIBERATELY ignores clean a_val and infers the
            # clean group proportions from the KNOWN coordinated attack structure.
            # This demonstrates the "attack known, no clean validation" scenario
            # Kuldeep asked about. Requires coordinated=True so the 70/30 inversion
            # in _empirical_pi_clean matches the actual attack (see run_canonical_empirical.py).
            pi_clean = self._empirical_pi_clean(pi_obs)
            n_eff = n
        elif a_val is not None and len(a_val) > 0:
            pi_clean = np.array([np.mean(a_val == j) for j in [0, 1]])
            n_eff = len(a_val)
        else:
            pi_clean = np.zeros(2)
            for j in [0, 1]:
                if self.alpha != 0.5:
                    pi_clean[j] = (pi_obs[j] - self.alpha) / (1 - 2 * self.alpha)
                else:
                    pi_clean[j] = pi_obs[j]
                pi_clean[j] = np.clip(pi_clean[j], 0.0, 1.0)
            # Renormalize to ensure valid probability distribution (sum to 1)
            pi_sum = np.sum(pi_clean)
            if pi_sum > 0:
                pi_clean = pi_clean / pi_sum
            n_eff = n

        if self.pi_shrinkage_k > 0.0:
            k = self.pi_shrinkage_k
            pi_clean = (pi_clean * n_eff + k * 0.5) / (n_eff + k)

        rho_dp = []
        for j in [0, 1]:
            denom = (1 - self.alpha) * pi_clean[j] + self.alpha
            rho_dp.append(self.alpha / denom if denom > 0 else 1.0)
        rho_if = 2 * self.alpha - self.alpha ** 2

        # Agent N1: global radius scale (provenance-recorded; default 1.0 = canonical).
        if self.radii_scale != 1.0:
            rho_dp = [r * self.radii_scale for r in rho_dp]
            rho_if = rho_if * self.radii_scale
        # Agent L2: per-group radius clamp (LSAC degeneracy fix; default None = canonical).
        # Cap rho_dp[j] so the tiny minority group's radius cannot blow up on
        # imbalanced data. Applied after scaling so the cap is in scaled units.
        if self.radii_clamp is not None:
            rho_dp = [min(r, self.radii_clamp) for r in rho_dp]
        return rho_dp, rho_if

    def _empirical_pi_clean(self, pi_obs):
        """Recover clean group proportions from observed post-attack proportions.

        Q5: exploits *known attack structure* (coordinated 70%-minority attribute
        flips) + only alpha (no per-sample mask, no oracle leak) to invert exactly.
        Used when a_val unavailable for direct clean pi; see _compute_radii.

        Clean derivation (see also src/Q5_derivation.md):
            Let n_c = alpha * n = corruption budget (exact).
            Coordinated targeting: 0.7*n_c clean-minority samples flipped min->maj;
            0.3*n_c clean-majority samples flipped maj->min.
            Net effect on observed counts:
                observed_min_count = N_min_clean - 0.7 n_c + 0.3 n_c = N_min_clean - 0.4 n_c
                observed_maj_count = N_maj_clean - 0.3 n_c + 0.7 n_c = N_maj_clean + 0.4 n_c
            Thus (divide by n):
                pi_obs[min] = pi_clean[min] - 0.4 * alpha
                pi_obs[maj] = pi_clean[maj] + 0.4 * alpha
            Solve for clean (add 0.4 alpha to the observed min side):
                pi_clean[min] = pi_obs[min] + 0.4 * alpha
                pi_clean[maj] = pi_obs[maj] - 0.4 * alpha
            Then clip to [0,1] and renormalize (sum to 1).

            In code: minority_idx = argmin(pi_obs)  [post-attack observed smaller group
            remains the original min for alpha<=0.4 on our data; formula uses whichever
            is smaller in obs].

        Validity: exact inversion under the 70/30 coordinated attr-flip model when
        0.7*alpha*n <= clean_min_count (no capping of flips). Holds for alpha<=0.4
        on Adult/Credit/LSAC. Uniform mode uses the generic (pi_obs - alpha)/(1-2alpha)
        instead (assumes uniform corruption across groups).
        """
        if self.alpha == 0.0:
            return pi_obs.copy()
        minority_idx = int(np.argmin(pi_obs))
        majority_idx = 1 - minority_idx
        pi_clean = pi_obs.copy()
        pi_clean[minority_idx] = pi_obs[minority_idx] + 0.4 * self.alpha
        pi_clean[majority_idx] = pi_obs[majority_idx] - 0.4 * self.alpha
        pi_clean = np.clip(pi_clean, 0.0, 1.0)
        s = pi_clean.sum()
        if s > 0:
            pi_clean = pi_clean / s
        return pi_clean

    def _build_knn_graph(self, X):
        """Precompute k-NN graph for IF (cosine; matches attack+eval after Agent A fix)."""
        n = len(X)
        k_eff = min(self.k, n - 1)
        # Cosine: same metric as FairnessTargetedPGD IF attack and metrics.compute_if_violation
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric='cosine', n_jobs=1).fit(X)
        distances, indices = nbrs.kneighbors(X)
        edges_i, edges_j, edge_dists = [], [], []
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
        """Initialize importance weights uniformly."""
        p_dp = {}
        for j in [0, 1]:
            mask = group_mask_dict[j]
            nj = mask.sum().item()
            p_dp[j] = torch.ones(nj, device=self.device) / nj
        p_if = torch.ones(n, device=self.device) / n
        return p_dp, p_if

    def _project_dp_weights(self, p, center, radius):
        """Project onto simplex ∩ L1-ball (TV distance → L1 radius = 2ρ)."""
        p_np = p.detach().cpu().numpy()
        center_np = center.detach().cpu().numpy()
        proj = project_simplex_l1_ball(p_np, center_np, 2 * radius, max_iter=500, tol=1e-5)
        return torch.tensor(proj, dtype=p.dtype, device=p.device)

    def _compute_dp_loss_weighted(self, h_tilde, a, p_dp_dict, group_mask_dict):
        """Weighted DP violation: |h̄_1 - h̄_0|.

        NOTE (audit): assumes binary protected attr (groups 0/1) — see group_mask_dict
        built in fit(). Most datasets (Adult/Credit/LSAC) are binary; compute_dp_violation
        in src/evaluation/metrics.py already generalizes to >2 groups via max_g - min_g
        (tested in test_metrics.py for 3/4 groups). Trainers remain binary-only per current spec.
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
        """Weighted IF violation: 1/(n-1) Σ_i Σ_j (p_i+p_j)/2 * (|h_i-h_j|-d_ij-γ)_+"""
        n = self.n_samples
        if len(edge_i) == 0 or n <= 1:
            return torch.tensor(0.0, device=self.device)
        h_i = h_tilde[edge_i]
        h_j = h_tilde[edge_j]
        p_i = p_if[edge_i]
        p_j = p_if[edge_j]
        weights = (p_i + p_j) / 2.0
        violations = F.relu(torch.abs(h_i - h_j) - edge_dists - self.gamma)
        return (weights * violations).sum() / (n - 1)

    def _compute_tilted_loss(self, per_sample_loss):
        """Exact tilted risk: β * log(mean(exp(ℓ/β)))."""
        m = len(per_sample_loss)
        if m == 0:
            return torch.tensor(0.0, device=self.device)
        return self.beta * (
            torch.logsumexp(per_sample_loss / self.beta, dim=0)
            - torch.log(torch.tensor(m, dtype=torch.float32, device=self.device))
        )

    def fit(self, X, y, a, X_val=None, y_val=None, a_val=None, verbose=False):
        """Train DRO-FAIR (Algorithm 1) — full-batch implementation.

        Paper Algorithm 1 ordering (Page 33):
        1. Forward pass: logits = model(X); h_tilde = sigmoid(logits * tau)
        2. Compute losses: L_tilt, g_dp, g_if with current p
        3. Theta update: total_loss.backward(); opt_theta.step()
        4. Dual ascent: lambda += lr_lambda * g (clamped to lambda_max)
        5. Inner maximization: K steps of projected gradient ascent on p (after θ update)
        """
        n = len(X)
        self.n_samples = n

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device)

        self.rho_dp, self.rho_if = self._compute_radii(a, a_val=a_val)
        group_mask_dict = {j: (a_t == j) for j in [0, 1]}
        group_sizes = {j: group_mask_dict[j].sum().item() for j in [0, 1]}

        p_dp_dict, p_if = self._init_weights(n, group_mask_dict)
        self.p_dp_center = {j: torch.ones(group_sizes[j], device=self.device) / group_sizes[j] for j in [0, 1]}
        self.p_if_center = torch.ones(n, device=self.device) / n

        opt_theta = torch.optim.AdamW(self.model.parameters(), lr=self.lr_theta, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(opt_theta, step_size=30, gamma=0.5)
        lambda_dp = torch.tensor(float(self.lambda_init), device=self.device)
        lambda_if = torch.tensor(float(self.lambda_init), device=self.device)
        self._lambda_lr = self.lr_lambda

        edge_i = edge_j = edge_dists = None
        if self.use_if:
            edge_i, edge_j, edge_dists = self._build_knn_graph(X)

        history = {'train_loss': [], 'val_acc': [], 'val_dp': [], 'val_if': [],
                   'lambda_dp': [], 'lambda_if': [], 'g_dp': [], 'g_if': [],
                   'current_tau': [], 'val_loss': []}

        for epoch in range(self.epochs):
            self.model.train()

            # === TAU WARMUP + λ DECAY ===
            current_tau = self.tau if epoch >= self.tau_warmup_epochs else 1.0
            lambda_lr = self._lambda_lr * (0.95 ** epoch)  # λ decay prevents over-growth

            # === STEP 1: FORWARD PASS (with current θ) ===
            logits = self.model(X_t)
            h_tilde = torch.sigmoid(logits * current_tau)

            # === STEP 2: COMPUTE BASE LOSSES (with current p) ===
            per_sample_loss = F.binary_cross_entropy_with_logits(logits, y_t, reduction='none')
            L_tilt = self._compute_tilted_loss(per_sample_loss)

            g_dp = self._compute_dp_loss_weighted(h_tilde, a_t, p_dp_dict, group_mask_dict) if self.use_dp else torch.tensor(0.0, device=self.device)
            g_if = self._compute_if_loss_weighted(h_tilde, p_if, edge_i, edge_j, edge_dists) if self.use_if else torch.tensor(0.0, device=self.device)

            # === STEP 3a: UPDATE θ (outer minimization) ===
            # Per paper Algorithm 1: θ update happens BEFORE inner max on p
            total_loss = L_tilt + (lambda_dp * g_dp if self.use_dp else 0.0) + (lambda_if * g_if if self.use_if else 0.0)
            opt_theta.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            opt_theta.step()
            lr_scheduler.step()

            # === STEP 3b: DUAL ASCENT on λ ===
            with torch.no_grad():
                if self.use_dp:
                    lambda_dp = torch.clamp(lambda_dp + lambda_lr * g_dp, 0, self.lambda_max)
                if self.use_if:
                    lambda_if = torch.clamp(lambda_if + lambda_lr * g_if, 0, self.lambda_max)

            # === STEP 4: INNER MAXIMIZATION (update p AFTER θ and λ) ===
            # Per paper Algorithm 1: p update happens AFTER θ and λ updates
            # Skip entirely when alpha=0: radii=0 so p never moves, but the loop
            # still advances torch RNG — causing DRO to diverge from Naive at α=0.
            with torch.no_grad():
                h_tilde_detach = h_tilde.detach()

            for _ in range(self.K_inner if self.alpha > 0 else 0):
                if self.use_dp:
                    for j in [0, 1]:
                        p_j = p_dp_dict[j].clone().detach().requires_grad_(True)
                        p_temp = {0: p_dp_dict[0].detach(), 1: p_dp_dict[1].detach()}
                        p_temp[j] = p_j
                        dp_loss = self._compute_dp_loss_weighted(h_tilde_detach, a_t, p_temp, group_mask_dict)
                        dp_loss.backward()
                        if p_j.grad is not None:
                            with torch.no_grad():
                                p_dp_dict[j] = p_j + self.lr_p * p_j.grad
                                p_dp_dict[j] = self._project_dp_weights(p_dp_dict[j], self.p_dp_center[j], self.rho_dp[j])

                if self.use_if:
                    p_if_grad = p_if.clone().detach().requires_grad_(True)
                    if_loss = self._compute_if_loss_weighted(h_tilde_detach, p_if_grad, edge_i, edge_j, edge_dists)
                    if_loss.backward()
                    if p_if_grad.grad is not None:
                        with torch.no_grad():
                            p_if = p_if_grad + self.lr_p * p_if_grad.grad
                            p_if = self._project_dp_weights(p_if, self.p_if_center, self.rho_if)

            history['train_loss'].append(float(total_loss.item()))
            history['lambda_dp'].append(float(lambda_dp.item()) if self.use_dp else 0.0)
            history['lambda_if'].append(float(lambda_if.item()) if self.use_if else 0.0)
            history['g_dp'].append(float(g_dp.item()) if self.use_dp else 0.0)
            history['g_if'].append(float(g_if.item()) if self.use_if else 0.0)
            history['current_tau'].append(float(current_tau))

            # Validation (use same temperature as training for consistent signals)
            # CRITICAL: must use the *epoch's* current_tau (computed from tau_warmup), NOT always self.tau.
            # Verified by test_dro_fair_validation_uses_current_tau and explicit current_tau passing.
            if X_val is not None:
                from src.evaluation.metrics import compute_metrics_torch
                metrics = compute_metrics_torch(
                    self.model, X_val, y_val, a_val,
                    device=self.device, temperature=current_tau, k=self.k, gamma=self.gamma
                )
                history["val_acc"].append(float(metrics["accuracy"]))
                history["val_dp"].append(float(metrics["dp_violation"]))
                history["val_if"].append(float(metrics["if_violation"]))
                # Per-epoch val history improvement: also track val_loss (BCE)
                # when validation data provided. Appended every epoch (per-epoch).
                Xv_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                yv_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    val_logits = self.model(Xv_t)
                    val_loss = F.binary_cross_entropy_with_logits(val_logits, yv_t)
                history['val_loss'].append(float(val_loss.item()))
                if verbose:
                    print(f"Epoch {epoch+1}/{self.epochs}: loss={total_loss.item():.4f}, "
                          f"val_acc={metrics['accuracy']:.4f}, val_dp={metrics['dp_violation']:.4f}, "
                          f"val_if={metrics['if_violation']:.4f}, "
                          f"lambda_dp={lambda_dp.item():.2f}, lambda_if={lambda_if.item():.2f}")

        # Keep original history dict format for backward compatibility (tests rely on hist['train_loss'][-1])
        self.history = history  # Also expose as self.history for Agent C's convergence plots

        if self.history_path:
            import json as _json
            with open(self.history_path, 'w') as _hf:
                _json.dump(history, _hf, indent=2)
        return history

    def predict(self, X):
        """Make binary predictions."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            probs = torch.sigmoid(self.model(X_t))
            return (probs >= 0.5).cpu().numpy()
