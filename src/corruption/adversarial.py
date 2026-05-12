"""
Adversarial corruption module.
Replaces random noise with adversarial attacks following:
https://jonathan-hui.medium.com/adversarial-attacks-b58318bb497b

Includes:
- FGSM/PGD-style attacks on numeric features
- Coordinated label flips
- Coordinated protected attribute flips
- RandomCorruptor for baseline comparison
"""

import numpy as np
import torch
import torch.nn as nn


class AdversarialCorruptor:
    """
    Applies adversarial corruption to a fraction alpha of the data.
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
            random_state: random seed
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.pgd_step_size = pgd_step_size
        self.feature_attack = feature_attack
        self.label_flip = label_flip
        self.attr_flip = attr_flip
        self.coordinated = coordinated
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
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
                corrupt_idx.extend(np.random.choice(minority_indices, n_minority_corrupt, replace=False))
            if n_majority_corrupt > 0 and len(majority_indices) > 0:
                corrupt_idx.extend(np.random.choice(majority_indices, n_majority_corrupt, replace=False))
            corrupt_idx = np.array(corrupt_idx, dtype=np.int64)
        else:
            if n_corrupt > 0:
                corrupt_idx = np.random.choice(n, n_corrupt, replace=False).astype(np.int64)
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
                
                noise = np.random.randn(X.shape[1])
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
        """Adversarial label flips: flip labels to maximize DP violation."""
        y_adv = y.copy()
        
        # Compute current group-conditional positive rates
        group0_pos = np.mean(y[a == 0])
        group1_pos = np.mean(y[a == 1])
        
        for idx in corrupt_idx:
            group = int(a[idx])
            current_label = int(y[idx])
            
            # Flip to increase group rate disparity
            # If group0 has lower rate, flip group0 negatives to positives
            # and group1 positives to negatives
            if group == 0 and group0_pos <= group1_pos and current_label == 0:
                y_adv[idx] = 1
            elif group == 0 and group0_pos > group1_pos and current_label == 1:
                y_adv[idx] = 0
            elif group == 1 and group1_pos <= group0_pos and current_label == 0:
                y_adv[idx] = 1
            elif group == 1 and group1_pos > group0_pos and current_label == 1:
                y_adv[idx] = 0
            else:
                # Default: random flip
                y_adv[idx] = 1 - current_label
        
        return y_adv
    
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
        if random_state is not None:
            np.random.seed(random_state)
    
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
            corrupt_idx = np.random.choice(n, n_corrupt, replace=False).astype(np.int64)
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
                noise = np.random.randn(X.shape[1])
                X_c[idx] = X[idx] + self.epsilon * col_stds.squeeze() * noise
        
        # 2. Random label flips (uniform, not coordinated)
        if self.label_flip and len(corrupt_idx) > 0:
            for idx in corrupt_idx:
                y_c[idx] = 1 - int(y[idx])
        
        # 3. Random attribute flips (uniform, not coordinated)
        if self.attr_flip and len(corrupt_idx) > 0:
            a_c[corrupt_idx] = 1 - a_c[corrupt_idx]
        
        return X_c, y_c, a_c, corrupt_mask
