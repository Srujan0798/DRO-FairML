# Fairness-Targeted PGD Attack Design

## Why We Need This

The current `_attack_labels()` in `adversarial.py` (lines 165-195) uses **heuristic rules**:
- It looks at current group rates and flips to maximize disparity
- It doesn't actually optimize — it just follows rules
- This is "good enough" but not mathematically optimal

**The proposed approach:** Use gradient-based optimization to find the EXACT samples whose flip would cause the most unfairness increase.

---

## Current Attack vs Proposed Attack

| Aspect | Current (Heuristic) | Proposed (Gradient-Based) |
|--------|---------------------|--------------------------|
| Decision rule | "Flip to increase group rate gap" | "Compute d(DP)/d(y_i) and flip top-k" |
| Optimality | Suboptimal — guess-based | **Optimal** — finds true max |
| Complexity | O(n) | O(n) but vectorized |
| Metric targeting | DP only | DP, IF, or combined |

---

## Mathematical Formulation

### Step 1: Define the Attack Target

We attack **Demographic Parity (DP)**:
```
DP = |P(Y=1|A=0) - P(Y=1|A=1)|
```

We want to maximize DP by flipping labels.

### Step 2: Compute Per-Sample Gradient

For each sample i, compute how flipping its label affects DP:

```
∂DP/∂y_i = 
    +1 if (a_i == 0 and current P0 >= P1 and y_i == 0)  # flip 0→1 increases P0
    -1 if (a_i == 0 and current P0 < P1 and y_i == 1)   # flip 1→0 decreases P0
    +1 if (a_i == 1 and current P1 >= P0 and y_i == 0)  # flip 0→1 increases P1
    -1 if (a_i == 1 and current P1 < P0 and y_i == 1)   # flip 1→0 decreases P1
```

More formally:
```python
def compute_dp_gradient(y, a):
    """
    Vectorized gradient computation for DP attack.
    
    Returns: gradient array of same shape as y
    gradient[i] > 0 means flipping y_i would INCREASE DP
    gradient[i] < 0 means flipping y_i would DECREASE DP
    """
    n = len(y)
    p0 = np.mean(y[a == 0])  # current group 0 rate
    p1 = np.mean(y[a == 1])  # current group 1 rate
    
    # Initialize gradient
    grad = np.zeros(n)
    
    # For group 0 samples (a=0):
    # If P0 >= P1: flipping 0→1 increases P0, increases DP → grad = +|dP0/dy|
    # If P0 < P1: flipping 1→0 decreases P0, decreases DP → grad = -|dP0/dy|
    mask0 = (a == 0)
    grad[mask0] = np.where(
        p0 >= p1,
        # P0 >= P1: flip 0→1 is bad (increases disparity)
        np.where(y[mask0] == 0, +1.0, -1.0),  # 0→1 is +, 1→0 is -
        # P0 < P1: flip 1→0 is bad (decreases P0, increases gap)
        np.where(y[mask0] == 1, +1.0, -1.0)   # 1→0 is +, 0→1 is -
    )
    
    # For group 1 samples (a=1):
    mask1 = (a == 1)
    grad[mask1] = np.where(
        p1 >= p0,
        np.where(y[mask1] == 0, +1.0, -1.0),
        np.where(y[mask1] == 1, +1.0, -1.0)
    )
    
    return grad
```

### Step 3: PGD Attack Loop

```python
def fairness_targeted_pgd(y, a, alpha, n_steps=5):
    """
    Fairness-Targeted PGD Attack
    
    Args:
        y: labels (binary)
        a: protected attributes (binary)
        alpha: fraction to corrupt
        n_steps: PGD steps (we iterate because flipping changes group rates)
    
    Returns:
        y_attacked: corrupted labels
        corrupt_mask: which samples were flipped
    """
    y_adv = y.copy()
    n_corrupt = int(alpha * len(y))
    
    for step in range(n_steps):
        # Compute gradient with CURRENT y_adv
        grad = compute_dp_gradient(y_adv, a)
        
        # Select top-k samples with largest positive gradient
        # These are samples where flipping INCREASES DP
        top_k_idx = np.argsort(-grad)[:n_corrupt]
        
        # Apply the flips
        y_adv[top_k_idx] = 1 - y_adv[top_k_idx]
    
    corrupt_mask = np.zeros(len(y), dtype=bool)
    corrupt_mask[top_k_idx] = True
    
    return y_adv, corrupt_mask
```

### Step 4: Combine with Feature Attack

The label attack works alongside the feature attack from the current code. The key insight is that we can **chain** attacks:

```python
# Complete attack pipeline
def full_fairness_attack(X, y, a, model, alpha=0.3):
    # Step 1: Feature attack (current PGD, keep it)
    X_attacked = pgd_feature_attack(X, y, model)
    
    # Step 2: Label attack (NEW gradient-based)
    y_attacked = fairness_targeted_pgd(y, a, alpha)
    
    # Step 3: Attribute attack (keep current)
    a_attacked = attribute_flip(a, alpha)
    
    return X_attacked, y_attacked, a_attacked
```

---

## API Design: `FairnessTargetedPGD` Class

```python
class FairnessTargetedPGD:
    """
    Gradient-based fairness attack targeting DP, IF, or both.
    
    Unlike the heuristic approach in AdversarialCorruptor,
    this class uses actual gradient computation to find
    the OPTIMAL set of samples to flip for maximum unfairness.
    """
    
    def __init__(self, alpha=0.2, target_metric='dp', 
                 pgd_steps=5, coordinated=True, random_state=None):
        """
        Args:
            alpha: fraction of samples to corrupt
            target_metric: 'dp' (Demographic Parity) or 'if' (Individual Fairness)
                          or 'combined' (weighted sum)
            pgd_steps: number of PGD iterations (flipping changes group rates)
            coordinated: if True, target minority group more aggressively
            random_state: for reproducibility
        """
        self.alpha = alpha
        self.target_metric = target_metric
        self.pgd_steps = pgd_steps
        self.coordinated = coordinated
        self.rng = np.random.RandomState(random_state)
    
    def compute_fairness_gradient(self, y, a):
        """
        Compute gradient of fairness metric w.r.t. each sample's label.
        
        Returns gradient array where:
          - positive value = flipping this label increases unfairness
          - negative value = flipping decreases unfairness
          - zero = flipping has no effect
        """
        if self.target_metric == 'dp':
            return self._compute_dp_gradient(y, a)
        elif self.target_metric == 'if':
            return self._compute_if_gradient(y, a)
        elif self.target_metric == 'combined':
            return 0.5 * self._compute_dp_gradient(y, a) + 0.5 * self._compute_if_gradient(y, a)
    
    def _compute_dp_gradient(self, y, a):
        """Demographic Parity gradient."""
        # [Implementation above]
        pass
    
    def _compute_if_gradient(self, y, a):
        """Individual Fairness gradient."""
        # IF = E[|h(x_i) - h(x_j)||a_i == a_j, similar(x_i, x_j)]
        # More complex - requires k-NN graph
        # For now, approximate with variance heuristic
        pass
    
    def corrupt(self, X, y, a, model=None, device='cpu'):
        """
        Apply fairness-targeted attack.
        
        For label attacks, uses gradient-based selection.
        For feature attacks, falls back to parent's PGD if model provided.
        """
        n = len(y)
        n_corrupt = int(self.alpha * n)
        
        # Feature attack (same as current)
        X_c = self._attack_features(X, y, model, device)
        
        # Fairness-targeted label attack (NEW)
        y_c, corrupt_mask = self._attack_labels_fairness(y, a, n_corrupt)
        
        # Attribute attack (same as current, but can be coordinated)
        a_c = self._attack_attributes(a, corrupt_mask)
        
        return X_c, y_c, a_c, corrupt_mask
    
    def _attack_labels_fairness(self, y, a, n_corrupt):
        """Gradient-based label attack (the core new contribution)."""
        y_adv = y.copy()
        
        for step in range(self.pgd_steps):
            grad = self.compute_fairness_gradient(y_adv, a)
            top_k = np.argsort(-grad)[:n_corrupt]
            y_adv[top_k] = 1 - y_adv[top_k]
        
        return y_adv, top_k
```

---

## Comparison: Heuristic vs Gradient Attack

```python
# HEURISTIC (current)
def _attack_labels_heuristic(y, a, corrupt_idx):
    group0_pos = np.mean(y[a == 0])
    group1_pos = np.mean(y[a == 1])
    for idx in corrupt_idx:
        group = int(a[idx])
        current_label = int(y[idx])
        # Rule-based: flip if it increases group rate gap
        if group == 0 and group0_pos <= group1_pos and current_label == 0:
            y[idx] = 1
        # ... more rules
        
# GRADIENT-BASED (proposed)
def _attack_labels_gradient(y, a, corrupt_idx):
    grad = compute_dp_gradient(y, a)
    # Pick samples with LARGEST positive gradient (most harmful)
    # This is deterministic and optimal
    top_k = np.argsort(-grad)[len(corrupt_idx)]
    y[top_k] = 1 - y[top_k]
```

---

## Why This Matters for Research

1. **More realistic threat model**: Real attackers would use gradient-based attacks, not rules
2. **Stronger baseline**: If our DRO-FAIR can defend against gradient attacks, it's more robust
3. **Publication value**: Novel contribution over current paper's random corruption

---

## Next Steps

1. Implement `compute_dp_gradient()` and `compute_if_gradient()` 
2. Add `FairnessTargetedPGD` class to `src/corruption/adversarial.py`
3. Benchmark: does it cause more fairness violation than the heuristic attack?
4. Does DRO-FAIR trained on heuristic attacks still defend against gradient attacks?

---

## References

1. Madry et al. 2018 — "Towards Deep Learning Models Resistant to Adversarial Attacks" — Section 2 (PGD formulation)
2. Solans et al. 2021 — "Poisoning Attacks on Algorithmic Fairness" — Section 2.2 (fairness gradient derivation)