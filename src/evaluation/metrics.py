"""
Evaluation metrics for fairness and accuracy.
- Accuracy
- Demographic Parity (DP) violation
- Individual Fairness (IF) violation

CRITICAL FIX: For DP and IF, use soft predictions h̃ = σ(τ·f_θ(x)) to match
paper's training objective. Accuracy uses hard binary predictions.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    return np.mean(y_true == y_pred)


def compute_dp_violation(y_pred, a):
    """
    Compute Demographic Parity violation: |P(h=1|A=0) - P(h=1|A=1)|.
    
    Args:
        y_pred: predictions (numpy array, can be soft [0,1] or hard {0,1})
        a: protected attributes (numpy array)
    """
    y_pred = np.asarray(y_pred, dtype=np.float32)
    a = np.asarray(a)
    groups = np.unique(a)
    rates = []
    for g in groups:
        mask = a == g
        if mask.sum() > 0:
            rates.append(np.mean(y_pred[mask]))
        else:
            rates.append(0.0)
    
    if len(rates) >= 2:
        return abs(rates[0] - rates[1])
    return 0.0


def compute_if_violation(X, y_pred, a=None, k=5, gamma=0.0, metric='euclidean'):
    """
    Compute Individual Fairness violation using k-NN approximation.
    IF violation: fraction of neighbor pairs where |h(x_i) - h(x_j)| > d(x_i, x_j) + gamma.
    
    For soft predictions, this measures the average violation magnitude.
    For hard predictions, this counts the fraction of violating pairs.
    
    Args:
        X: features (numpy array)
        y_pred: predictions (numpy array, can be soft [0,1] or hard {0,1})
        a: protected attributes (optional)
        k: number of nearest neighbors
        gamma: slack parameter
        metric: distance metric
    
    Returns:
        IF violation rate (or average violation for soft predictions)
    """
    y_pred = np.asarray(y_pred, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    if n <= 1:
        return 0.0
    
    # Use min(k, n-1) neighbors
    k_eff = min(k, n - 1)
    
    # Fit k-NN
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Skip self (first neighbor)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    violation_count = 0
    total_pairs = 0
    total_magnitude = 0.0
    
    for i in range(n):
        for idx, j in enumerate(indices[i]):
            if j >= n:
                continue
            
            dist = distances[i, idx]
            pred_diff = abs(float(y_pred[i]) - float(y_pred[j]))
            
            # Violation if predictions differ and distance is small
            violation = max(0.0, pred_diff - dist - gamma)
            if violation > 0:
                violation_count += 1
            total_magnitude += violation
            total_pairs += 1
    
    if total_pairs == 0:
        return 0.0
    
    # Return fraction of violating pairs (matches paper's definition)
    return violation_count / total_pairs


def compute_metrics_torch(model, X, y, a, device='cpu', temperature=1.0, k=5, gamma=0.0):
    """Compute all metrics using torch tensors.
    
    CRITICAL FIX: Use soft predictions h̃ = σ(τ·f_θ(x)) for DP and IF,
    matching the paper's training objective. Use hard predictions for accuracy.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_t)
        
        # Hard predictions for accuracy
        hard_preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
        
        # Soft predictions with τ for DP and IF (matches paper)
        soft_probs = torch.sigmoid(logits * temperature).cpu().numpy()
    
    acc = compute_accuracy(y, hard_preds)
    dp = compute_dp_violation(soft_probs, a)
    if_metric = compute_if_violation(X, soft_probs, a, k=k, gamma=gamma)
    
    return {
        'accuracy': acc,
        'dp_violation': dp,
        'if_violation': if_metric
    }
