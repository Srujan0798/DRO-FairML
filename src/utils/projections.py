"""
Projection utilities for DRO-FAIR.
Projects a vector onto the intersection of the probability simplex and an L1-ball.
Uses Dykstra's alternating projection algorithm.
"""

import numpy as np
import torch


def project_simplex(v):
    """Project v onto the probability simplex (sum=1, all >= 0)."""
    v = np.array(v)
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n) + 1
    cond = u - cssv / ind > 0
    if not np.any(cond):
        # All elements equal, uniform distribution
        return np.ones(n) / n
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    # Ensure numerical stability
    w = np.maximum(w, 0)
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.ones(n) / n
    return w


def project_l1_ball(v, center, radius):
    """Project v onto L1-ball centered at `center` with given radius.

    Uses direct soft-thresholding (Duchi et al. adapted for L1 balls).
    """
    v = np.array(v, dtype=np.float64)
    center = np.array(center, dtype=np.float64)
    if radius <= 1e-12:
        return center.copy()
    u = v - center
    norm_u = np.sum(np.abs(u))
    if norm_u <= radius:
        return v.copy()
    # Direct L1 projection via soft-thresholding
    abs_u = np.abs(u)
    signs = np.sign(u)
    sorted_abs = np.sort(abs_u)[::-1]
    cssv = np.cumsum(sorted_abs) - radius
    ind = np.arange(1, len(u) + 1, dtype=np.float64)
    cond = sorted_abs - cssv / ind > 0
    if not np.any(cond):
        return center.copy()
    rho = int(ind[cond][-1])
    lam = cssv[cond][-1] / rho
    w = signs * np.maximum(abs_u - lam, 0.0)
    return center + w


def project_simplex_l1_ball(v, center, radius, max_iter=100, tol=1e-5):
    """
    Dykstra's alternating projection onto intersection of simplex and L1-ball.

    Args:
        v: vector to project
        center: center of L1 ball
        radius: radius of L1 ball
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        projected vector
    """
    v = np.array(v, dtype=np.float64)
    center = np.array(center, dtype=np.float64)

    x = v.copy()
    p = np.zeros_like(v)
    q = np.zeros_like(v)

    for _ in range(max_iter):
        y = project_simplex(x + p)
        p = x + p - y
        z = project_l1_ball(y + q, center, radius)
        q = y + q - z
        x = z
        if np.linalg.norm(y - z) < tol:
            break

    # Tail loop: alternate simplex ↔ L1 ball until both constraints hold
    for _ in range(50):
        x = project_simplex(x)
        l1_excess = np.abs(x - center).sum() - radius
        if l1_excess <= 1e-9:
            break
        x = project_l1_ball(x, center, radius)
        if np.abs(x - center).sum() <= radius + 1e-9:
            x = project_simplex(x)
            if np.abs(x - center).sum() <= radius + 1e-9:
                break
    return x


def project_simplex_l1_ball_torch(v, center, radius, max_iter=100, tol=1e-6):
    """PyTorch version of Dykstra projection (for gradient compatibility)."""
    # Convert to numpy, project, convert back
    v_np = v.detach().cpu().numpy()
    center_np = center.detach().cpu().numpy() if torch.is_tensor(center) else center
    
    result = project_simplex_l1_ball(v_np, center_np, radius, max_iter, tol)
    return torch.tensor(result, dtype=v.dtype, device=v.device)
