#!/usr/bin/env python3
"""
Stream B diagnostics: Lambda trajectory and group-rate analysis.
Runs a single (dataset, alpha, seed) with logging enabled to diagnose DRO-FAIR behavior.

Usage:
    python3 experiments/diagnostics.py --dataset adult --alpha 0.2 --seed 42
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.standard_ml import StandardMLTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch
from experiments.run_experiments import get_temperature


def run_diagnostic(dataset_name, alpha, seed, output_dir='figures/diagnostics'):
    """Run one experiment with full epoch-level logging."""
    os.makedirs(output_dir, exist_ok=True)

    # Reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = \
        get_dataset(dataset_name, random_state=seed)

    tau = get_temperature(alpha)
    input_dim = X_train.shape[1]

    # Pretrain + corrupt
    model_pre = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    pretrainer = StandardMLTrainer(model_pre, device='cpu', epochs=15, lr=1e-3)
    pretrainer.fit(X_train, y_train, verbose=False)

    corruptor = AdversarialCorruptor(
        alpha=alpha, epsilon=0.1,
        feature_attack=True, label_flip=True, attr_flip=True,
        coordinated=True, random_state=seed
    )
    X_c, y_c, a_c, _ = corruptor.corrupt(X_train, y_train, a_train, model=model_pre, device='cpu')

    # Patch DroFairTrainer to log lambda and group rates per epoch
    log = {'epoch': [], 'lambda_dp': [], 'lambda_if': [],
           'train_dp': [], 'train_if': [], 'val_dp': [], 'val_if': [],
           'val_acc': [], 'train_loss': [],
           'group0_rate': [], 'group1_rate': []}

    original_fit = DroFairTrainer.fit

    def logging_fit(self, X, y, a, X_val=None, y_val=None, a_val=None, verbose=False):
        """Wrapper that captures per-epoch internals."""
        n = len(X)
        self.n_samples = n
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device)

        self.rho_dp, self.rho_if = self._compute_radii(a)
        group_mask_dict = {j: (a_t == j) for j in [0, 1]}
        group_sizes = {j: group_mask_dict[j].sum().item() for j in [0, 1]}

        p_dp_dict, p_if = self._init_weights(n, group_mask_dict)
        self.p_dp_center = {j: torch.ones(group_sizes[j], device=self.device) / group_sizes[j] for j in [0, 1]}
        self.p_if_center = torch.ones(n, device=self.device) / n

        opt_theta = torch.optim.AdamW(self.model.parameters(), lr=self.lr_theta, weight_decay=self.weight_decay)
        lambda_dp = torch.tensor(0.0, device=self.device)
        lambda_if = torch.tensor(0.0, device=self.device)

        edge_i, edge_j, edge_dists = self._build_knn_graph(X)

        for epoch in range(self.epochs):
            self.model.train()
            current_tau = self.tau if epoch >= self.tau_warmup_epochs else 1.0

            logits = self.model(X_t)
            h_tilde = torch.sigmoid(logits * current_tau)

            per_sample_loss = F.binary_cross_entropy_with_logits(logits, y_t, reduction='none')
            L_tilt = self._compute_tilted_loss(per_sample_loss)
            g_dp = self._compute_dp_loss_weighted(h_tilde, a_t, p_dp_dict, group_mask_dict)
            g_if = self._compute_if_loss_weighted(h_tilde, p_if, edge_i, edge_j, edge_dists)

            total_loss = L_tilt + lambda_dp * g_dp + lambda_if * g_if
            opt_theta.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt_theta.step()

            with torch.no_grad():
                lambda_dp = torch.clamp(lambda_dp + self.lr_lambda * g_dp, 0, self.lambda_max)
                lambda_if = torch.clamp(lambda_if + self.lr_lambda * g_if, 0, self.lambda_max)

            # Inner max
            with torch.no_grad():
                h_tilde_d = h_tilde.detach()
            for _ in range(self.K_inner):
                for j in [0, 1]:
                    p_j = p_dp_dict[j].clone().detach().requires_grad_(True)
                    p_temp = {0: p_dp_dict[0].detach(), 1: p_dp_dict[1].detach()}
                    p_temp[j] = p_j
                    dp_loss = self._compute_dp_loss_weighted(h_tilde_d, a_t, p_temp, group_mask_dict)
                    dp_loss.backward()
                    if p_j.grad is not None:
                        with torch.no_grad():
                            p_dp_dict[j] = p_j + self.lr_p * p_j.grad
                            p_dp_dict[j] = self._project_dp_weights(p_dp_dict[j], self.p_dp_center[j], self.rho_dp[j])

                p_if_grad = p_if.clone().detach().requires_grad_(True)
                if_loss = self._compute_if_loss_weighted(h_tilde_d, p_if_grad, edge_i, edge_j, edge_dists)
                if_loss.backward()
                if p_if_grad.grad is not None:
                    with torch.no_grad():
                        p_if = p_if_grad + self.lr_p * p_if_grad.grad
                        p_if = self._project_if_weights(p_if, self.p_if_center, self.rho_if)

            # Log
            with torch.no_grad():
                g0 = h_tilde[group_mask_dict[0]].mean().item()
                g1 = h_tilde[group_mask_dict[1]].mean().item()

            log['epoch'].append(epoch)
            log['lambda_dp'].append(lambda_dp.item())
            log['lambda_if'].append(lambda_if.item())
            log['train_dp'].append(g_dp.item())
            log['train_if'].append(g_if.item())
            log['train_loss'].append(total_loss.item())
            log['group0_rate'].append(g0)
            log['group1_rate'].append(g1)

            if X_val is not None:
                metrics = compute_metrics_torch(
                    self.model, X_val, y_val, a_val,
                    device=self.device, temperature=self.tau, k=self.k, gamma=self.gamma
                )
                log['val_dp'].append(metrics['dp_violation'])
                log['val_if'].append(metrics['if_violation'])
                log['val_acc'].append(metrics['accuracy'])

    # Run with logging
    model_dro = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer = DroFairTrainer(
        model_dro, alpha=alpha, device='cpu',
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=2.0,
        tau=tau, beta=5.0, k=5, gamma=0.0,
        K_inner=10, epochs=60, weight_decay=1e-4, tau_warmup_epochs=5
    )
    logging_fit(trainer, X_c, y_c, a_c, X_val=X_val, y_val=y_val, a_val=a_val)

    # === Plot 1: Lambda trajectories ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{dataset_name.upper()} | alpha={alpha} | seed={seed}', fontsize=14)

    axes[0, 0].plot(log['epoch'], log['lambda_dp'], label='lambda_DP', color='blue')
    axes[0, 0].plot(log['epoch'], log['lambda_if'], label='lambda_IF', color='red')
    axes[0, 0].set_ylabel('Lambda')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].set_title('Lagrange Multipliers')

    axes[0, 1].plot(log['epoch'], log['train_dp'], label='Train DP', color='blue')
    if log['val_dp']:
        axes[0, 1].plot(log['epoch'], log['val_dp'], label='Val DP', color='blue', linestyle='--')
    axes[0, 1].plot(log['epoch'], log['train_if'], label='Train IF', color='red')
    if log['val_if']:
        axes[0, 1].plot(log['epoch'], log['val_if'], label='Val IF', color='red', linestyle='--')
    axes[0, 1].set_ylabel('Violation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].set_title('Fairness Violations')

    axes[1, 0].plot(log['epoch'], log['group0_rate'], label='Group 0 rate')
    axes[1, 0].plot(log['epoch'], log['group1_rate'], label='Group 1 rate')
    axes[1, 0].set_ylabel('Mean Prediction')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].set_title('Group Rates (h_tilde)')

    axes[1, 1].plot(log['epoch'], log['train_loss'], label='Train Loss', color='green')
    if log['val_acc']:
        ax2 = axes[1, 1].twinx()
        ax2.plot(log['epoch'], log['val_acc'], label='Val Acc', color='orange', linestyle='--')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='lower right')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].set_title('Loss & Accuracy')

    plt.tight_layout()
    fname = f'{dataset_name}_a{alpha}_s{seed}_diagnostic.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    print(f"Saved {output_dir}/{fname}")

    # Save log data (convert numpy types for JSON serialization)
    import numpy as _np
    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        if isinstance(obj, (_np.floating, _np.integer)):
            return obj.item()
        return obj
    log_path = os.path.join(output_dir, f'{dataset_name}_a{alpha}_s{seed}_log.json')
    with open(log_path, 'w') as f:
        json.dump(_to_python(log), f, indent=2)
    print(f"Saved {log_path}")

    return log


def compare_group_rates(output_dir='figures/diagnostics'):
    """B2: Compare group rates across datasets at alpha=0.2."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, ds in enumerate(['adult', 'credit', 'lsac']):
        log_path = os.path.join(output_dir, f'{ds}_a0.2_s42_log.json')
        if not os.path.exists(log_path):
            print(f"Missing {log_path}, running diagnostic first...")
            run_diagnostic(ds, 0.2, 42, output_dir)
            if not os.path.exists(log_path):
                continue

        log = json.load(open(log_path))
        ax = axes[idx]
        ax.plot(log['epoch'], log['group0_rate'], label='Group 0')
        ax.plot(log['epoch'], log['group1_rate'], label='Group 1')
        gap = [abs(g1 - g0) for g0, g1 in zip(log['group0_rate'], log['group1_rate'])]
        ax.fill_between(log['epoch'], 0, gap, alpha=0.2, color='red', label='DP gap')
        ax.set_title(f'{ds.upper()} (alpha=0.2)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Group Rate')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'group_rates_comparison.png'), dpi=150)
    print(f"Saved {output_dir}/group_rates_comparison.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='adult')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compare', action='store_true', help='Run B2: group rate comparison across datasets')
    args = parser.parse_args()

    if args.compare:
        compare_group_rates()
    else:
        run_diagnostic(args.dataset, args.alpha, args.seed)
