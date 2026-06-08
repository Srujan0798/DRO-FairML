"""
Adversarial corruption module.
Replaces random noise with adversarial attacks following:
https://jonathan-hui.medium.com/adversarial-attacks-b58318bb497b

Includes:
- FGSM/PGD-style attacks on numeric features
- Coordinated label flips
- Coordinated protected attribute flips
- RandomCorruptor for baseline comparison

CRITICAL FIX: Use local np.random.RandomState instead of global np.random.seed
to prevent interference between multiple corruptor instances.
"""

import numpy as np
import torch
import torch.nn as nn


class AdversarialCorruptor:
    """
    Applies adversarial corruption to a fraction alpha of the data.
    When model is provided, uses true gradient-based PGD attacks.
    When model is None, uses FGSM-style heuristic attacks.
    """
    
    def __init__(self, alpha=0.2, epsilon=0.1, pgd_steps=5, pgd_step_size=0.02,
                 feature_attack=True, label_flip=True, attr_flip=True,
                 coordinated=True, random_state=None):
        """
        Args:
            alpha: corruption fraction
            epsilon: max perturbation magnitude for feature attacks (as multiple of std)
            pgd_steps: number of PGD steps for feature attack
            pgd_step_size: step size for PGD
            feature_attack: whether to apply FGSM/PGD on features
            label_flip: whether to flip labels
            attr_flip: whether to flip protected attributes
            coordinated: whether attacks are coordinated (target minority group)
            random_state: random seed for local RandomState
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.pgd_step_size = pgd_step_size
        self.feature_attack = feature_attack
        self.label_flip = label_flip
        self.attr_flip = attr_flip
        self.coordinated = coordinated
        # CRITICAL FIX: Use local RandomState, not global seed
        self.rng = np.random.RandomState(random_state)
    
    def corrupt(self, X, y, a, model=None, device='cpu'):
        """
        Apply adversarial corruption to dataset.
        
        Args:
            X: features (numpy array, already standardized)
            y: labels (numpy array)
            a: protected attributes (numpy array)
            model: trained model for gradient-based attacks (optional)
            device: torch device
        
        Returns:
            X_c, y_c, a_c: corrupted data
            corrupt_mask: boolean mask of corrupted samples
        """
        n = len(X)
        n_corrupt = int(self.alpha * n)
        
        # Select samples to corrupt
        if self.coordinated:
            # Target minority group more aggressively (adversarial coordination)
            group_counts = np.bincount(a.astype(int))
            minority_group = int(np.argmin(group_counts))
            minority_indices = np.where(a == minority_group)[0]
            majority_indices = np.where(a != minority_group)[0]
            
            # Corrupt minority group with higher probability
            n_minority_corrupt = min(int(0.7 * n_corrupt), len(minority_indices))
            n_majority_corrupt = n_corrupt - n_minority_corrupt
            
            corrupt_idx = []
            if n_minority_corrupt > 0:
                corrupt_idx.extend(self.rng.choice(minority_indices, n_minority_corrupt, replace=False))
            if n_majority_corrupt > 0 and len(majority_indices) > 0:
                corrupt_idx.extend(self.rng.choice(majority_indices, n_majority_corrupt, replace=False))
            corrupt_idx = np.array(corrupt_idx, dtype=np.int64)
        else:
            if n_corrupt > 0:
                corrupt_idx = self.rng.choice(n, n_corrupt, replace=False).astype(np.int64)
            else:
                corrupt_idx = np.array([], dtype=np.int64)
        
        corrupt_mask = np.zeros(n, dtype=bool)
        if len(corrupt_idx) > 0:
            corrupt_mask[corrupt_idx] = True
        
        X_c = X.copy()
        y_c = y.copy()
        a_c = a.copy()
        
        # 1. Feature perturbation (PGD/FGSM-style)
        if self.feature_attack:
            X_c = self._attack_features(X_c, y_c, a_c, corrupt_idx, model, device)
        
        # 2. Label flips (adversarial: flip to maximize unfairness)
        if self.label_flip:
            y_c = self._attack_labels(y_c, a_c, corrupt_idx)
        
        # 3. Protected attribute flips
        if self.attr_flip:
            a_c = self._attack_attributes(a_c, corrupt_idx)
        
        return X_c, y_c, a_c, corrupt_mask
    
    def _attack_features(self, X, y, a, corrupt_idx, model, device):
        """Apply PGD-style feature perturbation."""
        X_adv = X.copy()
        if len(corrupt_idx) == 0:
            return X_adv
        
        if model is None:
            # FGSM-style perturbation using data statistics
            col_stds = np.std(X, axis=0, keepdims=True)
            col_stds[col_stds == 0] = 1.0
            
            for idx in corrupt_idx:
                # Gradient approximation: perturb towards opposite class
                target_label = 1 - int(y[idx])
                direction = 2 * target_label - 1  # +1 if target=1, -1 if target=0
                
                noise = self.rng.randn(X.shape[1])
                noise = noise / (np.linalg.norm(noise) + 1e-8)
                
                # Scale by column std and epsilon
                perturbation = self.epsilon * direction * col_stds.squeeze() * noise
                X_adv[idx] = X[idx] + perturbation
        else:
            # True gradient-based PGD attack
            model.eval()
            X_batch = torch.tensor(X[corrupt_idx], dtype=torch.float32, device=device, requires_grad=True)
            y_batch = torch.tensor(y[corrupt_idx], dtype=torch.float32, device=device)
            
            with torch.enable_grad():
                for step in range(self.pgd_steps):
                    logits = model(X_batch)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, y_batch)
                    grad = torch.autograd.grad(loss, X_batch)[0]
                    
                    # PGD step: move in direction of gradient
                    X_batch = X_batch + self.pgd_step_size * torch.sign(grad)
                    
                    # Project back to epsilon ball
                    delta = X_batch - torch.tensor(X[corrupt_idx], dtype=torch.float32, device=device)
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                    X_batch = torch.tensor(X[corrupt_idx], dtype=torch.float32, device=device) + delta
                    X_batch = X_batch.detach().requires_grad_(True)
            
            X_adv[corrupt_idx] = X_batch.detach().cpu().numpy()
        
        return X_adv
    
    def _attack_labels(self, y, a, corrupt_idx):
        """Adversarial label flips: greedily flip labels to maximize DP violation."""
        y_adv = y.copy()
        
        for idx in corrupt_idx:
            # Recompute rates after each flip (old code computed once, causing stale heuristic)
            mask0 = (a == 0)
            mask1 = (a == 1)
            group0_pos = np.mean(y_adv[mask0]) if np.any(mask0) else 0.5
            group1_pos = np.mean(y_adv[mask1]) if np.any(mask1) else 0.5
            
            group = int(a[idx])
            current_label = int(y_adv[idx])
            
            # Flip to increase group rate disparity
            if group == 0 and group0_pos <= group1_pos and current_label == 0:
                y_adv[idx] = 1
            elif group == 0 and group0_pos > group1_pos and current_label == 1:
                y_adv[idx] = 0
            elif group == 1 and group1_pos <= group0_pos and current_label == 0:
                y_adv[idx] = 1
            elif group == 1 and group1_pos > group0_pos and current_label == 1:
                y_adv[idx] = 0
            else:
                # No beneficial flip: keep original
                pass
        
        return y_adv
    
    def _attack_attributes(self, a, corrupt_idx):
        """Flip protected attributes."""
        a_adv = a.copy()
        a_adv[corrupt_idx] = 1 - a_adv[corrupt_idx]
        return a_adv


class FairnessTargetedPGD:
    """
    Gradient-based fairness attack targeting DP, IF, or combined metric.

    Unlike AdversarialCorruptor which uses heuristic rules for label flips,
    this class computes the EXACT gradient of the fairness loss with respect
    to each sample's label, then selects the top-k samples whose flip would
    cause the MOST unfairness increase.

    Mathematical basis:
        DP = |P(Y=1|A=0) - P(Y=1|A=1)|
        For binary labels, we can compute d(DP)/d(y_i) analytically.

    Reference: Solans et al. "Poisoning Attacks on Algorithmic Fairness" (2021)
    """

    def __init__(self, alpha=0.2, target_metric='dp', pgd_steps=20,
                 epsilon=0.3, pgd_step_size=0.02,
                 coordinated=True, random_state=None):
        """
        Args:
            alpha: fraction of samples to corrupt
            target_metric: 'dp' (Demographic Parity) or 'if' (Individual Fairness)
                          or 'combined' (weighted sum)
            pgd_steps: number of PGD iterations for feature attack
            epsilon: max perturbation magnitude for feature attack (as multiple of std)
            pgd_step_size: step size for PGD feature attack
            coordinated: if True, target minority group more aggressively
            random_state: random seed for reproducibility
        """
        self.alpha = alpha
        self.target_metric = target_metric
        self.pgd_steps = pgd_steps
        self.epsilon = epsilon
        self.pgd_step_size = pgd_step_size
        self.coordinated = coordinated
        self.rng = np.random.RandomState(random_state)

    def compute_dp_gradient(self, y, a):
        """
        Compute the EXACT marginal gain in DP violation from flipping each sample.

        For group g, flipping a sample changes the group rate by ±1/count_g.
        The marginal gain depends on:
          1. Which group has the higher rate
          2. The current label (flipping 0→1 or 1→0)
          3. The group size (larger impact in smaller groups)

        DP = |P(Y=1|A=0) - P(Y=1|A=1)| = |p0 - p1|

        Marginal gain for flipping sample i in group g:
          If p0 > p1:
            g=0, y_i=0→1:  gain = +1/n0  (increases p0, widens gap)
            g=0, y_i=1→0:  gain = -1/n0  (decreases p0, shrinks gap)
            g=1, y_i=1→0:  gain = +1/n1  (decreases p1, widens gap)
            g=1, y_i=0→1:  gain = -1/n1  (increases p1, shrinks gap)
          If p1 > p0: symmetric

        Returns:
            grad: array where grad[i] is the exact DP increase from flipping y[i]
                  (positive = flip increases unfairness, negative = flip decreases it)
        """
        n = len(y)
        grad = np.full(n, -np.inf)

        mask0 = (a == 0)
        mask1 = (a == 1)
        count0 = np.sum(mask0)
        count1 = np.sum(mask1)

        if count0 == 0 or count1 == 0:
            return grad

        p0 = np.mean(y[mask0])
        p1 = np.mean(y[mask1])

        inv_n0 = 1.0 / count0
        inv_n1 = 1.0 / count1

        if p0 >= p1:
            # Group 0 higher: widen by increasing p0 or decreasing p1
            grad[mask0] = np.where(y[mask0] == 0, +inv_n0, -inv_n0)
            grad[mask1] = np.where(y[mask1] == 1, +inv_n1, -inv_n1)
        else:
            # Group 1 higher: widen by increasing p1 or decreasing p0
            grad[mask1] = np.where(y[mask1] == 0, +inv_n1, -inv_n1)
            grad[mask0] = np.where(y[mask0] == 1, +inv_n0, -inv_n0)

        return grad

    def _precompute_if_neighbors(self, X, a, k=5):
        """Precompute k-NN neighbor indices for IF gradient (computed once, reused)."""
        from sklearn.neighbors import NearestNeighbors
        neighbors = {}
        for g in [0, 1]:
            mask_g = (a == g)
            if mask_g.sum() <= k:
                continue
            idx_g = np.where(mask_g)[0]
            X_g = X[idx_g]
            k_eff = min(k, len(idx_g) - 1)
            nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_g)
            _, neighbor_idx = nbrs.kneighbors(X_g)
            neighbor_idx = neighbor_idx[:, 1:]  # drop self
            neighbors[g] = {'idx_g': idx_g, 'neighbor_idx': neighbor_idx, 'k_eff': k_eff}
        return neighbors

    def compute_if_gradient(self, y, a, X=None, k=5, precomputed_neighbors=None):
        """
        Compute gradient of Individual Fairness w.r.t. each sample's label.

        IF violation = (1/|N_k|) Σ_{(i,j)∈N_k} max(0, |y_i - y_j| - d(x_i,x_j) - γ)
        where N_k are k-NN pairs in feature space.

        Flipping y_i increases IF violation when y_i agrees with most of its
        k-nearest neighbors (because flipping creates new disagreements).
        Conversely, flipping reduces IF violation when y_i disagrees with most
        neighbors (because flipping fixes the disagreement).

        Gradient sign:
            +1: flipping increases IF violation (target for attack)
            -1: flipping decreases IF violation (avoid)

        Args:
            y: labels (n,)
            a: protected attribute (n,) — k-NN is computed WITHIN same group
            X: features (n, d) — required for proper k-NN gradient
            k: number of neighbors
            precomputed_neighbors: dict from _precompute_if_neighbors() for speed

        Returns:
            grad: array where grad[i] > 0 means flipping y[i] INCREASES IF violation
        """
        n = len(y)
        grad = np.zeros(n)

        if X is None:
            return grad

        if precomputed_neighbors is None:
            precomputed_neighbors = self._precompute_if_neighbors(X, a, k)

        # Use precomputed k-NN within each protected group (vectorized)
        for g, info in precomputed_neighbors.items():
            idx_g = info['idx_g']
            neighbor_idx = info['neighbor_idx']
            k_eff = info['k_eff']
            y_g = y[idx_g]

            # Vectorized: y_g[neighbor_idx] has shape (n_g, k_eff)
            # Compare each sample's label to its neighbors' labels
            neighbor_labels = y_g[neighbor_idx]  # shape: (n_g, k_eff)
            agree = np.sum(neighbor_labels == y_g[:, None], axis=1)  # shape: (n_g,)
            disagree = k_eff - agree
            grad[idx_g] = (agree - disagree) / k_eff

        return grad

    def compute_fairness_gradient(self, y, a, X=None, precomputed_neighbors=None):
        """
        Compute gradient for the target fairness metric.

        Returns:
            grad: gradient array where positive = flip to increase unfairness
        """
        if self.target_metric == 'dp':
            return self.compute_dp_gradient(y, a)
        elif self.target_metric == 'if':
            return self.compute_if_gradient(y, a, X=X, precomputed_neighbors=precomputed_neighbors)
        elif self.target_metric == 'combined':
            dp_grad = self.compute_dp_gradient(y, a)
            if_grad = self.compute_if_gradient(y, a, X=X, precomputed_neighbors=precomputed_neighbors)
            # Normalize to comparable scales before mixing (DP grad is ~1/count_g,
            # IF grad is in [-1,1]; without normalization IF dominates completely)
            finite_dp = dp_grad[np.isfinite(dp_grad)]
            dp_max = np.max(np.abs(finite_dp)) + 1e-12 if len(finite_dp) > 0 else 1.0
            if_max = np.max(np.abs(if_grad)) + 1e-12
            return 0.5 * (dp_grad / dp_max) + 0.5 * (if_grad / if_max)
        else:
            raise ValueError(f"Unknown metric: {self.target_metric}")

    def _select_targets(self, grad, n_corrupt, a):
        """
        Select the top-n_corrupt samples to flip based on gradient.

        If coordinated=True, prioritize minority group samples.

        Returns:
            target_idx: indices of samples to flip
        """
        n = len(grad)

        if self.coordinated:
            # Target minority group more aggressively
            group_counts = np.bincount(a.astype(int))
            minority_group = int(np.argmin(group_counts))
            minority_indices = np.where(a == minority_group)[0]
            majority_indices = np.where(a != minority_group)[0]

            # Allocate 70% of corruption budget to minority
            n_minority = min(int(0.7 * n_corrupt), len(minority_indices))
            n_majority = n_corrupt - n_minority

            # Get top-k from each group
            minority_grad = grad[minority_indices]
            majority_grad = grad[majority_indices]

            # Sort by gradient magnitude (descending)
            minority_top = minority_indices[np.argsort(-minority_grad)[:n_minority]]
            majority_top = majority_indices[np.argsort(-majority_grad)[:n_majority]]

            target_idx = np.concatenate([minority_top, majority_top])
        else:
            # Random selection from top gradient samples
            top_k = np.argsort(-grad)[:n_corrupt]
            target_idx = self.rng.choice(top_k, n_corrupt, replace=False)

        return target_idx

    def _attack_labels_fairness(self, y, a, X=None):
        """
        Greedy fairness-targeted label attack.

        At each step, computes the EXACT marginal gain in the fairness
        metric from flipping each unflipped sample. Flips the sample
        with the highest positive gain, then recomputes all gains.

        This is correct for discrete label flips; the old batched-PGD
        approach was wrong because it flipped the same samples multiple
        times and used uniform (not group-size-weighted) gradients.

        Returns:
            y_attacked: corrupted labels
            corrupt_mask: boolean mask of flipped samples
        """
        y_adv = y.copy()
        n = len(y)
        n_corrupt = int(self.alpha * n)
        flipped = np.zeros(n, dtype=bool)

        # Precompute k-NN once for IF attacks (major speedup — was recomputing 3000×)
        precomputed_neighbors = None
        if self.target_metric in ('if', 'combined') and X is not None:
            precomputed_neighbors = self._precompute_if_neighbors(X, a)

        # Coordinated targeting: track minority group budget
        group_counts = np.bincount(a.astype(int))
        minority_group = int(np.argmin(group_counts))
        n_minority_target = int(0.7 * n_corrupt) if self.coordinated else 0

        for _ in range(n_corrupt):
            grad = self.compute_fairness_gradient(y_adv, a, X, precomputed_neighbors=precomputed_neighbors)
            grad[flipped] = -np.inf  # do not re-flip

            if not np.any(grad > 0):
                break  # no beneficial flip left

            if self.coordinated:
                n_minority_done = np.sum(flipped & (a == minority_group))
                minority_mask = (a == minority_group) & ~flipped
                majority_mask = (a != minority_group) & ~flipped

                # Try to allocate 70% to minority group
                if n_minority_done < n_minority_target and np.any(minority_mask):
                    masked_grad = np.full(n, -np.inf)
                    masked_grad[minority_mask] = grad[minority_mask]
                    best = np.argmax(masked_grad)
                elif np.any(majority_mask):
                    masked_grad = np.full(n, -np.inf)
                    masked_grad[majority_mask] = grad[majority_mask]
                    best = np.argmax(masked_grad)
                else:
                    best = np.argmax(grad)
            else:
                best = np.argmax(grad)

            y_adv[best] = 1 - y_adv[best]
            flipped[best] = True

        corrupt_mask = flipped
        return y_adv, corrupt_mask

    def corrupt(self, X, y, a, model=None, device='cpu'):
        """
        Apply fairness-targeted adversarial corruption.

        Total corruption budget is exactly alpha*n samples.
        All three attacks (label, feature, attribute) target the SAME indices.

        Args:
            X: features (numpy array, already standardized)
            y: labels (numpy array)
            a: protected attributes (numpy array)
            model: trained model for gradient-based feature attacks (optional)
            device: torch device

        Returns:
            X_c, y_c, a_c, corrupt_mask
        """
        n = len(y)
        n_corrupt = int(self.alpha * n)

        X_c = X.copy()
        y_c = y.copy()
        a_c = a.copy()

        # 1. Fairness-targeted label attack FIRST to select worst samples
        y_c, corrupt_mask = self._attack_labels_fairness(y_c, a_c, X_c)
        corrupt_idx = np.where(corrupt_mask)[0]

        # 2. Feature perturbation on SAME indices
        if len(corrupt_idx) > 0:
            if model is None:
                # Train surrogate model for gradient-based PGD when no model provided
                model = self._train_surrogate(X, y, device=device)
            X_c = self._attack_features_pgd(X_c, y_c, corrupt_idx, model, device)

        # 3. Attribute flips on SAME indices
        if len(corrupt_idx) > 0:
            a_c = self._attack_attributes(a_c, corrupt_idx)

        return X_c, y_c, a_c, corrupt_mask

    def _train_surrogate(self, X, y, device='cpu'):
        """Train a quick LogisticRegression surrogate for gradient-based PGD."""
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=500, solver='lbfgs')
        lr.fit(X, y)
        W = torch.tensor(lr.coef_, dtype=torch.float32, device=device)
        b = torch.tensor(lr.intercept_, dtype=torch.float32, device=device)

        class LRSurrogate(nn.Module):
            def __init__(s): super().__init__(); s.W = W; s.b = b
            def forward(s, x): return (x @ s.W.T + s.b).squeeze(-1)

        return LRSurrogate().to(device)

    def _attack_features_fgsm(self, X, y, corrupt_idx):
        """FGSM-style feature perturbation (no model needed)."""
        X_adv = X.copy()
        if len(corrupt_idx) == 0:
            return X_adv

        col_stds = np.std(X, axis=0, keepdims=True)
        col_stds[col_stds == 0] = 1.0

        for idx in corrupt_idx:
            target_label = 1 - int(y[idx])
            direction = 2 * target_label - 1
            noise = self.rng.randn(X.shape[1])
            noise = noise / (np.linalg.norm(noise) + 1e-8)
            perturbation = self.epsilon * direction * col_stds.squeeze() * noise
            X_adv[idx] = X[idx] + perturbation

        return X_adv

    def _attack_features_pgd(self, X, y, corrupt_idx, model, device):
        """PGD-style feature perturbation (model required)."""
        X_adv = X.copy()
        if len(corrupt_idx) == 0:
            return X_adv

        model.eval()
        X_batch = torch.tensor(X[corrupt_idx], dtype=torch.float32, device=device, requires_grad=True)
        y_batch = torch.tensor(y[corrupt_idx], dtype=torch.float32, device=device)
        X_orig = torch.tensor(X[corrupt_idx], dtype=torch.float32, device=device)

        for step in range(self.pgd_steps):
            logits = model(X_batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_batch)
            grad = torch.autograd.grad(loss, X_batch)[0]
            X_batch = X_batch + self.pgd_step_size * torch.sign(grad)
            delta = torch.clamp(X_batch - X_orig, -self.epsilon, self.epsilon)
            X_batch = (X_orig + delta).detach().requires_grad_(True)

        X_adv[corrupt_idx] = X_batch.detach().cpu().numpy()
        return X_adv

    def _attack_attributes(self, a, corrupt_idx):
        """Flip protected attributes."""
        a_adv = a.copy()
        a_adv[corrupt_idx] = 1 - a_adv[corrupt_idx]
        return a_adv


class RandomCorruptor:
    """
    Random corruption baseline for comparison with adversarial corruption.
    Uses the same corruption fraction but applies random (non-adversarial) noise.
    """
    
    def __init__(self, alpha=0.2, epsilon=0.1,
                 feature_attack=True, label_flip=True, attr_flip=True,
                 random_state=None):
        """
        Args:
            alpha: corruption fraction
            epsilon: std of Gaussian noise for feature perturbation
            feature_attack: whether to add Gaussian noise to features
            label_flip: whether to flip labels uniformly at random
            attr_flip: whether to flip protected attributes uniformly at random
            random_state: random seed
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.feature_attack = feature_attack
        self.label_flip = label_flip
        self.attr_flip = attr_flip
        self.rng = np.random.RandomState(random_state)
    
    def corrupt(self, X, y, a, model=None, device='cpu'):
        """
        Apply random corruption to dataset.
        
        Args:
            X: features (numpy array)
            y: labels (numpy array)
            a: protected attributes (numpy array)
            model: unused (for API compatibility)
            device: unused (for API compatibility)
        
        Returns:
            X_c, y_c, a_c, corrupt_mask
        """
        n = len(X)
        n_corrupt = int(self.alpha * n)
        
        # Uniform random selection (no coordinated targeting)
        if n_corrupt > 0:
            corrupt_idx = self.rng.choice(n, n_corrupt, replace=False).astype(np.int64)
        else:
            corrupt_idx = np.array([], dtype=np.int64)
        
        corrupt_mask = np.zeros(n, dtype=bool)
        if len(corrupt_idx) > 0:
            corrupt_mask[corrupt_idx] = True
        
        X_c = X.copy()
        y_c = y.copy()
        a_c = a.copy()
        
        # 1. Random Gaussian feature noise
        if self.feature_attack and len(corrupt_idx) > 0:
            col_stds = np.std(X, axis=0, keepdims=True)
            col_stds[col_stds == 0] = 1.0
            for idx in corrupt_idx:
                noise = self.rng.randn(X.shape[1])
                X_c[idx] = X[idx] + self.epsilon * col_stds.squeeze() * noise
        
        # 2. Random label flips (uniform, not coordinated)
        if self.label_flip and len(corrupt_idx) > 0:
            for idx in corrupt_idx:
                y_c[idx] = 1 - int(y[idx])
        
        # 3. Random attribute flips (uniform, not coordinated)
        if self.attr_flip and len(corrupt_idx) > 0:
            a_c[corrupt_idx] = 1 - a_c[corrupt_idx]
        
        return X_c, y_c, a_c, corrupt_mask
