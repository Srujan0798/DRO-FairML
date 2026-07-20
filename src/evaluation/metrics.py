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
    Compute Demographic Parity violation.

    Binary: |P(h=1|A=0) - P(h=1|A=1)|.
    Multigroup (>=2 groups, e.g. UTKFace race): max_g P(h=1|A=g) - min_g P(h=1|A=g).
        This is the natural generalization — measures the worst-case disparity
        between any two groups, which is what multi-group fairness requires.

    Audit note (per MASTER_PLAN): currently most datasets (Adult, Credit, LSAC) use
    binary protected attr, but implementation + tests (test_metrics.py) support >2
    groups via max-min. Trainers (dro_fair, naive_fair) assume binary [0,1] for their
    internal p/g computations; metrics layer is ready for multi. No hard assert(len==2).

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

    if len(rates) < 2:
        return 0.0
    if len(rates) == 2:
        return abs(rates[0] - rates[1])
    return float(np.max(rates) - np.min(rates))


def compute_if_violation(X, y_pred, a=None, k=5, gamma=0.0, metric='cosine'):
    """
    Compute Individual Fairness violation using k-NN approximation.

    IF violation: (1/(n-1)) * sum_{(i,j) in N_k} max(0, |h(x_i) - h(x_j)| - d(x_i, x_j) - gamma),
    where d(x_i, x_j) is the k-NN distance in feature space.

    CALIBRATION FIX (Agent A): The original implementation used raw Euclidean
    distance on StandardScaler-normalized features. On Adult/Credit/LSAC the
    resulting d_ij was ~0.8-2.8 while soft prediction differences |h_i-h_j| were
    ~0.02, so the ReLU saturated to zero for every pair and every IF value was
    floating-point dust. We now use cosine distance by default, which is bounded
    to [0, 2] and therefore on the same scale as the [0, 1] prediction
    differences. This changes the meaning of IF from an absolute feature-space
    threshold to an angular-similarity threshold; the change is documented and
    consistent across attack, training, and evaluation.

    For soft predictions, this measures the average violation magnitude.
    For hard predictions, this counts the fraction of violating pairs.

    Args:
        X: features (numpy array)
        y_pred: predictions (numpy array, can be soft [0,1] or hard {0,1})
        a: protected attributes (optional; not used in IF k-NN graph because
           IF evaluates similarity across the whole sample, not within groups)
        k: number of nearest neighbors
        gamma: slack parameter
        metric: distance metric. Default 'cosine' after calibration fix.
                Use 'euclidean' only if you explicitly rescale distances.

    Returns:
        IF violation magnitude averaged over (n-1).
    """
    y_pred = np.asarray(y_pred, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    if n <= 1:
        return 0.0

    # Use min(k, n-1) neighbors
    k_eff = min(k, n - 1)

    # Fit k-NN over ALL samples (not within protected groups).  The protected
    # attribute a is accepted for API compatibility but intentionally unused here;
    # IF is a similarity condition, not a group-wise one.
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=1).fit(X)
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

            # Violation if predictions differ more than the feature distance plus slack
            violation = max(0.0, pred_diff - dist - gamma)
            if violation > 0:
                violation_count += 1
            total_magnitude += violation
            total_pairs += 1

    if total_pairs == 0:
        return 0.0

    # Return weighted sum normalized by (n-1) — matches training metric (dro_fair.py:140)
    return total_magnitude / (n - 1) if n > 1 else 0.0


def compute_metrics_torch(model, X, y, a, device='cpu', temperature=1.0, k=5, gamma=0.0,
                          if_metric='cosine'):
    """Compute all metrics using torch tensors.

    CRITICAL FIX: Use soft predictions h̃ = σ(τ·f_θ(x)) for DP and IF,
    matching the paper's training objective. Use hard predictions for accuracy.

    CALIBRATION FIX (Agent A): IF now uses cosine distance by default so the
    feature-space distance is on the same scale as the [0, 1] prediction
    difference.  Use if_metric='euclidean' only if you explicitly rescale
    distances.
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
    if_metric = compute_if_violation(X, soft_probs, a, k=k, gamma=gamma, metric=if_metric)

    return {
        'accuracy': acc,
        'dp_violation': dp,
        'if_violation': if_metric
    }
