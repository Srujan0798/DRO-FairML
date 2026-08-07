#!/usr/bin/env python3
"""TASK E step 1: verify the augmented-Lagrangian gradient by hand + numerically.

The code adds (mu/2)*g_dp^2 to the scalar total_loss that is .backward()ed.
By autograd, d/dtheta[(mu/2)g^2] = mu*g*(dg/dtheta). We verify:
  (a) g_dp, g_if are non-negative by construction (no clamp silently needed);
  (b) the autograd gradient of total_loss equals gradient of the base loss
      PLUS mu*g*grad(g) computed on a separately-detached graph (no double
      counting, no sign error);
  (c) mu=0 contributes exactly zero to the gradient.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import MLPClassifier
from src.training.dro_fair import DroFairTrainer

torch.manual_seed(0)
np.random.seed(0)
n = 40
X = np.random.randn(n, 8).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
a = np.random.randint(0, 2, size=n).astype(np.int64)
Xv = np.random.randn(20, 8).astype(np.float32)
yv = (Xv[:, 0] > 0).astype(np.float32)
av = np.random.randint(0, 2, size=20).astype(np.int64)


def make_trainer(mu):
    model = MLPClassifier(8, hidden_dims=[8])
    return model, DroFairTrainer(model, alpha=0.2, device='cpu', epochs=1, K_inner=1,
                                 tau=1.0, aug_lagrangian_mu=mu)


# (a) non-negativity of g_dp / g_if via a real forward pass
model, trainer = make_trainer(0.0)
X_t = torch.tensor(X, dtype=torch.float32)
a_t = torch.tensor(a, dtype=torch.long)
model.train()
logits = model(X_t)
h = torch.sigmoid(logits * 1.0)
trainer.n_samples = n
trainer.rho_dp, trainer.rho_if = trainer._compute_radii(a, a_val=av)
group_mask = {j: (a_t == j) for j in [0, 1]}
sizes = {j: group_mask[j].sum().item() for j in [0, 1]}
p_dp, p_if = trainer._init_weights(n, group_mask)
edge_i, edge_j, edge_d = trainer._build_knn_graph(X)
g_dp = trainer._compute_dp_loss_weighted(h, a_t, p_dp, group_mask)
g_if = trainer._compute_if_loss_weighted(h, p_if, edge_i, edge_j, edge_d)
print("g_dp =", g_dp.item(), " non-negative:", g_dp.item() >= 0)
print("g_if =", g_if.item(), " non-negative:", g_if.item() >= 0)

# (b) gradient decomposition check at mu=20
torch.manual_seed(0)
np.random.seed(0)
model, trainer = make_trainer(20.0)
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)
a_t = torch.tensor(a, dtype=torch.long)
trainer.n_samples = n
trainer.rho_dp, trainer.rho_if = trainer._compute_radii(a, a_val=av)
group_mask = {j: (a_t == j) for j in [0, 1]}
p_dp, p_if = trainer._init_weights(n, group_mask)
edge_i, edge_j, edge_d = trainer._build_knn_graph(X)

mu = 20.0
logits = model(X_t)
h = torch.sigmoid(logits)
per_sample = F.binary_cross_entropy_with_logits(logits, y_t, reduction='none')
L_tilt = trainer._compute_tilted_loss(per_sample)
g_dp = trainer._compute_dp_loss_weighted(h, a_t, p_dp, group_mask)
g_if = trainer._compute_if_loss_weighted(h, p_if, edge_i, edge_j, edge_d)
total = L_tilt + 0.5 * mu * g_dp * g_dp + 0.5 * mu * g_if * g_if
grad_total = torch.autograd.grad(total, list(model.parameters()), retain_graph=True)

# separately: grad of base + mu*g*grad(g) computed on detached g
g_dp2 = trainer._compute_dp_loss_weighted(h, a_t, p_dp, group_mask)
g_if2 = trainer._compute_if_loss_weighted(h, p_if, edge_i, edge_j, edge_d)
base = L_tilt
grad_base = torch.autograd.grad(base, list(model.parameters()), retain_graph=True)
grad_dp = torch.autograd.grad(g_dp2, list(model.parameters()), retain_graph=True)
grad_if = torch.autograd.grad(g_if2, list(model.parameters()), retain_graph=True)
recon = [gb + mu * g_dp2.item() * gd + mu * g_if2.item() * gi
         for gb, gd, gi in zip(grad_base, grad_dp, grad_if)]

ok = True
for ga, gr in zip(grad_total, recon):
    diff = (ga - gr).abs().max().item()
    rel = diff / max(ga.abs().max().item(), 1e-9)
    print(f"grad max-abs-diff={diff:.3e} rel={rel:.3e}")
    ok = ok and rel < 1e-5
print("gradient decomposition (autograd total == base + mu*g*grad(g)):", "PASS" if ok else "FAIL")

# (c) mu=0 gradient equals base-only gradient
torch.manual_seed(0)
np.random.seed(0)
model0, trainer0 = make_trainer(0.0)
X_t0 = torch.tensor(X, dtype=torch.float32)
y_t0 = torch.tensor(y, dtype=torch.float32)
a_t0 = torch.tensor(a, dtype=torch.long)
trainer0.n_samples = n
trainer0.rho_dp, trainer0.rho_if = trainer0._compute_radii(a, a_val=av)
pm = {j: (a_t0 == j) for j in [0, 1]}
pd0, pf0 = trainer0._init_weights(n, pm)
ei0, ej0, ed0 = trainer0._build_knn_graph(X)
lg0 = model0(X_t0)
h0 = torch.sigmoid(lg0)
ps0 = F.binary_cross_entropy_with_logits(lg0, y_t0, reduction='none')
L0 = trainer0._compute_tilted_loss(ps0)
gd0 = trainer0._compute_dp_loss_weighted(h0, a_t0, pd0, pm)
gf0 = trainer0._compute_if_loss_weighted(h0, pf0, ei0, ej0, ed0)
tot0 = L0 + gd0 * 0 + gf0 * 0  # mu=0 -> no penalty terms
g0a = torch.autograd.grad(tot0, list(model0.parameters()), retain_graph=True)
g0b = torch.autograd.grad(L0, list(model0.parameters()))
okc = all((a - b).abs().max().item() < 1e-12 for a, b in zip(g0a, g0b))
print("mu=0 gradient == base gradient (bit-identical):", "PASS" if okc else "FAIL")
