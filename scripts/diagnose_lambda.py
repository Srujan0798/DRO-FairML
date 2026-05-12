"""
D8 diagnostic — log λ_DP and g_DP trajectory while training DRO-FAIR on
Adult α=0.2 seed=0. Purpose: confirm whether λ_DP runs to lambda_max=10
early and whether g_DP collapses thereafter (over-correction hypothesis).

Reads the live training loop via monkey-patching: wraps DroFairTrainer.fit
to capture (epoch, λ_DP, λ_IF, g_DP, g_IF, val_dp, val_acc) per step.

Usage:
    python3 scripts/diagnose_lambda.py
    python3 scripts/diagnose_lambda.py --dataset credit --alpha 0.3 --seed 0
    python3 scripts/diagnose_lambda.py --lambda_max 3.0 --lr_lambda 2.5e-3

Output:
    - prints per-epoch trajectory
    - saves CSV to results/lambda_trace_<dataset>_<alpha>_<seed>.csv
    - if matplotlib available, saves figures/lambda_trace_<...>.png
"""

import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.standard_ml import StandardMLTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch


def get_temperature(alpha):
    return 1.0 if alpha >= 0.4 else 100.0


def diagnose(dataset, alpha, seed, lr_theta, lr_lambda, lambda_max, K_inner,
             epochs, tau, beta, k, gamma, device='cpu'):
    """Train DRO-FAIR with per-epoch λ/g logging."""
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = \
        get_dataset(dataset, random_state=seed)

    # Warm-start for PGD attacks
    warm = MLPClassifier(X_train.shape[1], hidden_dims=[128, 64], dropout=0.1)
    StandardMLTrainer(warm, device=device, epochs=10, lr=1e-3).fit(X_train, y_train, verbose=False)

    # Corrupt training data
    c = AdversarialCorruptor(alpha=alpha, epsilon=0.1, pgd_steps=5, pgd_step_size=0.02,
                             coordinated=True, random_state=seed)
    X_tr, y_tr, a_tr, _ = c.corrupt(X_train, y_train, a_train, model=warm, device=device)

    # Pretrain
    model_pre = MLPClassifier(X_train.shape[1], hidden_dims=[128, 64], dropout=0.1)
    StandardMLTrainer(model_pre, device=device, epochs=15, lr=1e-3).fit(X_tr, y_tr, verbose=False)

    # Build DRO trainer
    model = MLPClassifier(X_train.shape[1], hidden_dims=[128, 64], dropout=0.1)
    model.load_state_dict(model_pre.state_dict())
    trainer = DroFairTrainer(
        model, alpha=alpha, device=device,
        lr_theta=lr_theta, lr_lambda=lr_lambda, lr_p=5e-3, lambda_max=lambda_max,
        tau=tau, beta=beta, k=k, gamma=gamma,
        K_inner=K_inner, epochs=epochs, weight_decay=1e-4,
    )

    # Replicate fit() but expose λ and g per epoch. Pulled verbatim from
    # DroFairTrainer.fit so we observe the exact dynamics.
    X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    a_t = torch.tensor(a_tr, dtype=torch.long, device=device)
    n = len(X_tr)
    trainer.n_samples = n
    trainer.rho_dp, trainer.rho_if = trainer._compute_radii(a_tr)

    group_mask = {j: (a_t == j) for j in [0, 1]}
    group_sizes = {j: group_mask[j].sum().item() for j in [0, 1]}
    p_dp_dict, p_if = trainer._init_weights(n, group_mask)
    trainer.p_dp_center = {j: torch.ones(group_sizes[j], device=device) / group_sizes[j] for j in [0, 1]}
    trainer.p_if_center = torch.ones(n, device=device) / n

    opt_theta = torch.optim.AdamW(trainer.model.parameters(),
                                  lr=trainer.lr_theta, weight_decay=trainer.weight_decay)
    lambda_dp = torch.tensor(1.0, device=device)
    lambda_if = torch.tensor(1.0, device=device)
    edge_i, edge_j, edge_dists = trainer._build_knn_graph(X_tr)

    trace = []
    print(f"{'ep':>3s} {'L_tilt':>8s} {'g_DP':>8s} {'g_IF':>8s} "
          f"{'λ_DP':>7s} {'λ_IF':>7s} {'val_DP':>8s} {'val_acc':>8s}")

    for epoch in range(trainer.epochs):
        trainer.model.train()
        logits = trainer.model(X_t)
        h_tilde = torch.sigmoid(logits * trainer.tau)
        per_sample_loss = F.binary_cross_entropy_with_logits(logits, y_t, reduction='none')
        L_tilt = trainer._compute_tilted_loss(per_sample_loss)
        g_dp = trainer._compute_dp_loss_weighted(h_tilde, a_t, p_dp_dict, group_mask)
        g_if = trainer._compute_if_loss_weighted(h_tilde, p_if, edge_i, edge_j, edge_dists)
        total = L_tilt + lambda_dp * g_dp + lambda_if * g_if

        opt_theta.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        opt_theta.step()

        with torch.no_grad():
            lambda_dp = torch.clamp(lambda_dp + trainer.lr_lambda * g_dp, 0, trainer.lambda_max)
            lambda_if = torch.clamp(lambda_if + trainer.lr_lambda * g_if, 0, trainer.lambda_max)

        # Inner max (K steps)
        with torch.no_grad():
            h_for_p = h_tilde.detach()
        for _ in range(trainer.K_inner):
            for j in [0, 1]:
                p_j = p_dp_dict[j].clone().detach().requires_grad_(True)
                p_tmp = {0: p_dp_dict[0].detach(), 1: p_dp_dict[1].detach()}
                p_tmp[j] = p_j
                dpl = trainer._compute_dp_loss_weighted(h_for_p, a_t, p_tmp, group_mask)
                dpl.backward()
                if p_j.grad is not None:
                    with torch.no_grad():
                        p_dp_dict[j] = p_j + trainer.lr_p * p_j.grad
                        p_dp_dict[j] = trainer._project_dp_weights(
                            p_dp_dict[j], trainer.p_dp_center[j], trainer.rho_dp[j]
                        )
            p_grad = p_if.clone().detach().requires_grad_(True)
            ifl = trainer._compute_if_loss_weighted(h_for_p, p_grad, edge_i, edge_j, edge_dists)
            ifl.backward()
            if p_grad.grad is not None:
                with torch.no_grad():
                    p_if = p_grad + trainer.lr_p * p_grad.grad
                    p_if = trainer._project_if_weights(p_if, trainer.p_if_center, trainer.rho_if)

        # Validation
        m = compute_metrics_torch(trainer.model, X_val, y_val, a_val,
                                  device=device, temperature=trainer.tau,
                                  k=trainer.k, gamma=trainer.gamma)
        row = {
            'epoch': epoch + 1,
            'L_tilt': float(L_tilt),
            'g_DP': float(g_dp),
            'g_IF': float(g_if),
            'lambda_DP': float(lambda_dp),
            'lambda_IF': float(lambda_if),
            'val_DP': m['dp_violation'],
            'val_acc': m['accuracy'],
        }
        trace.append(row)
        print(f"{row['epoch']:>3d} {row['L_tilt']:>8.4f} {row['g_DP']:>8.4f} "
              f"{row['g_IF']:>8.4f} {row['lambda_DP']:>7.3f} {row['lambda_IF']:>7.3f} "
              f"{row['val_DP']:>8.4f} {row['val_acc']:>8.4f}")

    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='adult')
    ap.add_argument('--alpha', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr_theta', type=float, default=1e-3)
    ap.add_argument('--lr_lambda', type=float, default=5e-3)
    ap.add_argument('--lambda_max', type=float, default=10.0)
    ap.add_argument('--K_inner', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--tau', type=float, default=None,
                    help='If None, uses get_temperature(alpha)')
    ap.add_argument('--beta', type=float, default=5.0)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--gamma', type=float, default=0.0)
    args = ap.parse_args()

    tau = args.tau if args.tau is not None else get_temperature(args.alpha)
    print(f"# DRO-FAIR diagnostic  dataset={args.dataset} α={args.alpha} seed={args.seed}")
    print(f"# lr_θ={args.lr_theta} lr_λ={args.lr_lambda} λ_max={args.lambda_max} "
          f"K={args.K_inner} epochs={args.epochs} τ={tau} β={args.beta} k={args.k}\n")

    trace = diagnose(args.dataset, args.alpha, args.seed,
                     args.lr_theta, args.lr_lambda, args.lambda_max,
                     args.K_inner, args.epochs, tau, args.beta, args.k, args.gamma)

    # Save CSV
    os.makedirs('results', exist_ok=True)
    csv_path = f"results/lambda_trace_{args.dataset}_{args.alpha}_{args.seed}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        w.writeheader()
        w.writerows(trace)
    print(f"\nCSV → {csv_path}")

    # Optional figure
    try:
        import matplotlib.pyplot as plt
        os.makedirs('figures', exist_ok=True)
        eps = [r['epoch'] for r in trace]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(eps, [r['lambda_DP'] for r in trace], label='λ_DP', marker='o')
        axes[0].plot(eps, [r['lambda_IF'] for r in trace], label='λ_IF', marker='s')
        axes[0].axhline(args.lambda_max, ls=':', c='r', alpha=0.5, label=f'λ_max={args.lambda_max}')
        axes[0].set_title(f'λ trajectory  ({args.dataset} α={args.alpha})')
        axes[0].set_xlabel('epoch'); axes[0].legend()
        axes[1].plot(eps, [r['g_DP'] for r in trace], label='g_DP', marker='o')
        axes[1].plot(eps, [r['g_IF'] for r in trace], label='g_IF', marker='s')
        axes[1].set_title('Fairness gap (training)')
        axes[1].set_xlabel('epoch'); axes[1].legend()
        axes[2].plot(eps, [r['val_DP'] for r in trace], label='val_DP', marker='o')
        axes[2].plot(eps, [r['val_acc'] for r in trace], label='val_acc', marker='s')
        axes[2].set_title('Validation')
        axes[2].set_xlabel('epoch'); axes[2].legend()
        plt.tight_layout()
        png_path = f"figures/lambda_trace_{args.dataset}_{args.alpha}_{args.seed}.png"
        plt.savefig(png_path, dpi=120)
        print(f"PNG → {png_path}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == '__main__':
    main()
