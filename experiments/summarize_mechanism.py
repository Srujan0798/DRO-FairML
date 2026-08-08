#!/usr/bin/env python3
"""
TASK B mechanism study: why does accuracy go UP under the augmented Lagrangian?

Hypothesis (the "denoising" story): under a DP-targeted attack, the corrupted
points are exactly the ones driving the group-rate gap, so penalising that gap
hard (via the AL quadratic penalty) suppresses the attack's influence — AL is
acting as attack-specific denoising.

Prediction: AL should fit the corrupted points LESS than canonical DRO does
(lower accuracy on the corrupted subset), while maintaining or improving accuracy
on the clean subset.

Cell (pre-registered in TASKS_AL_VALIDATION.md): Adult, alpha=0.2, seed=0,
attack=dp. Canonical mu=0 vs AL mu=5. Epochs=60, K_inner=10, PGD steps=20.

Writes:
  - results/history_mechanism_canonical.json
  - results/history_mechanism_al.json
  - results/al_mechanism_summary.md
  - figures/al_mechanism_*.png
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import FairnessTargetedPGD
from src.training.dro_fair import DroFairTrainer
from src.temperature import get_temperature
from experiments.run_ablation_parallel import _AblationLock


DATASET = 'adult'
ALPHA = 0.2
SEED = 0
ATTACK = 'dp'
MU_CANONICAL = 0.0
MU_AL = 5.0
EPOCHS = 60
K_INNER = 10
PGD_STEPS = 20


def run_dro(mu):
    """Run DRO with given mu. Returns (model, history, corrupt_mask, corrupted_data, test_data)."""
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, _ = \
        get_dataset(DATASET, random_state=SEED)

    tau = get_temperature(ALPHA)
    input_dim = X_train.shape[1]

    attack_obj = FairnessTargetedPGD(
        alpha=ALPHA, target_metric=ATTACK, pgd_steps=PGD_STEPS,
        epsilon=0.3, pgd_step_size=0.02, coordinated=False,
        random_state=SEED, k=5
    )
    X_att, y_att, a_att, corrupt_mask = attack_obj.corrupt(X_train, y_train, a_train)

    model = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer = DroFairTrainer(
        model, alpha=ALPHA, device='cpu',
        lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=1.5,
        tau=tau, beta=5.0, k=5, gamma=0.0,
        K_inner=K_INNER, epochs=EPOCHS, weight_decay=1e-4, tau_warmup_epochs=15,
        lambda_init=0.0, radii_mode='uniform',
        radii_scale=1.0, radii_clamp=None, pi_shrinkage_k=0.0,
        aug_lagrangian_mu=mu
    )
    history = trainer.fit(X_att, y_att, a_att,
                           X_val=X_val, y_val=y_val, a_val=a_val, verbose=False)

    data = {'X_att': X_att, 'y_att': y_att, 'a_att': a_att, 'tau': tau,
            'X_test': X_test, 'y_test': y_test, 'a_test': a_test}
    return model, history, corrupt_mask, data


def compute_subset_accuracies(model, X, y, corrupt_mask, tau):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        probs = torch.sigmoid(model(X_t) * tau)
        preds = (probs >= 0.5).numpy().astype(np.float32)
    y_np = np.asarray(y, dtype=np.float32)
    clean_mask = ~corrupt_mask
    out = {'n_all': len(y_np), 'n_corrupted': int(corrupt_mask.sum()),
           'n_clean': int(clean_mask.sum())}
    out['acc_all'] = float(np.mean(preds == y_np))
    out['acc_corrupted'] = float(np.mean(preds[corrupt_mask] == y_np[corrupt_mask])) \
        if corrupt_mask.sum() > 0 else float('nan')
    out['acc_clean'] = float(np.mean(preds[clean_mask] == y_np[clean_mask])) \
        if clean_mask.sum() > 0 else float('nan')
    return out


def compute_group_fractions(model, X, a, tau):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        probs = torch.sigmoid(model(X_t) * tau)
        preds = (probs >= 0.5).numpy().astype(np.float32)
    a_np = np.asarray(a)
    fracs = {}
    for g in [0, 1]:
        mask = (a_np == g)
        fracs[g] = float(np.mean(preds[mask])) if mask.sum() > 0 else float('nan')
    return fracs


def plot_g_dp(hist_can, hist_al):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(hist_can['g_dp']) + 1)
    ax.plot(epochs, hist_can['g_dp'], label='Canonical DRO (mu=0)', alpha=0.8)
    ax.plot(epochs, hist_al['g_dp'], label='AL DRO (mu=5)', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('g_dp (DP constraint violation)')
    ax.set_title('Plot 1: g_dp trajectory — does AL suppress g faster?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/al_mechanism_g_dp.png', dpi=150)
    plt.close(fig)


def plot_lambda_dp(hist_can, hist_al):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(hist_can['lambda_dp']) + 1)
    ax.plot(epochs, hist_can['lambda_dp'], label='Canonical DRO (mu=0)', alpha=0.8)
    ax.plot(epochs, hist_al['lambda_dp'], label='AL DRO (mu=5)', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('lambda_dp')
    ax.set_title('Plot 2: lambda_dp trajectory — does AL lambda grow larger?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/al_mechanism_lambda_dp.png', dpi=150)
    plt.close(fig)


def plot_val_acc(hist_can, hist_al):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(hist_can['val_acc']) + 1)
    ax.plot(epochs, hist_can['val_acc'], label='Canonical DRO (mu=0)', alpha=0.8)
    ax.plot(epochs, hist_al['val_acc'], label='AL DRO (mu=5)', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('val_acc')
    ax.set_title('Plot 3: val_acc trajectory — when does AL pull ahead?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/al_mechanism_val_acc.png', dpi=150)
    plt.close(fig)


def plot_corrupted_vs_clean(acc_can, acc_al):
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['Corrupted\nsubset', 'Clean\nsubset', 'Overall']
    canon_vals = [acc_can['acc_corrupted'], acc_can['acc_clean'], acc_can['acc_all']]
    al_vals = [acc_al['acc_corrupted'], acc_al['acc_clean'], acc_al['acc_all']]
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, canon_vals, width, label='Canonical DRO (mu=0)', alpha=0.8)
    bars2 = ax.bar(x + width/2, al_vals, width, label='AL DRO (mu=5)', alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Plot 4: corrupted vs clean accuracy — does AL fit attacked points less?')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig('figures/al_mechanism_corrupted_vs_clean.png', dpi=150)
    plt.close(fig)


def plot_group_fractions(grp_can_train, grp_al_train, grp_can_test, grp_al_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    groups = ['Group 0\n(Female)', 'Group 1\n(Male)']
    width = 0.35

    canon_train = [grp_can_train[0], grp_can_train[1]]
    al_train = [grp_al_train[0], grp_al_train[1]]
    x = np.arange(len(groups))
    axes[0].bar(x - width / 2, canon_train, width, label='Canonical DRO', alpha=0.8)
    axes[0].bar(x + width / 2, al_train, width, label='AL DRO', alpha=0.8)
    axes[0].set_ylabel('Fraction predicted positive')
    axes[0].set_title('Training (corrupted) data')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    canon_test = [grp_can_test[0], grp_can_test[1]]
    al_test = [grp_al_test[0], grp_al_test[1]]
    axes[1].bar(x - width / 2, canon_test, width, label='Canonical DRO', alpha=0.8)
    axes[1].bar(x + width / 2, al_test, width, label='AL DRO', alpha=0.8)
    axes[1].set_ylabel('Fraction predicted positive')
    axes[1].set_title('Test (clean) data')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Plot 5: prediction distribution — fraction predicted positive per group', y=1.02)
    fig.tight_layout()
    fig.savefig('figures/al_mechanism_group_fractions.png', dpi=150)
    plt.close(fig)


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    with _AblationLock():
        print("[mechanism] Running canonical DRO (mu=0)...", flush=True)
        t0 = time.time()
        model_can, hist_can, mask, data = run_dro(MU_CANONICAL)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)

        with open('results/history_mechanism_canonical.json', 'w') as f:
            json.dump(hist_can, f, indent=2)
        print("  wrote results/history_mechanism_canonical.json", flush=True)

        print("[mechanism] Running AL DRO (mu=5)...", flush=True)
        t0 = time.time()
        model_al, hist_al, _, _ = run_dro(MU_AL)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)

        with open('results/history_mechanism_al.json', 'w') as f:
            json.dump(hist_al, f, indent=2)
        print("  wrote results/history_mechanism_al.json", flush=True)

    tau = data['tau']
    acc_can = compute_subset_accuracies(model_can, data['X_att'], data['y_att'], mask, tau)
    acc_al = compute_subset_accuracies(model_al, data['X_att'], data['y_att'], mask, tau)

    grp_can_train = compute_group_fractions(model_can, data['X_att'], data['a_att'], tau)
    grp_al_train = compute_group_fractions(model_al, data['X_att'], data['a_att'], tau)
    grp_can_test = compute_group_fractions(model_can, data['X_test'], data['a_test'], tau)
    grp_al_test = compute_group_fractions(model_al, data['X_test'], data['a_test'], tau)

    plot_g_dp(hist_can, hist_al)
    plot_lambda_dp(hist_can, hist_al)
    plot_val_acc(hist_can, hist_al)
    plot_corrupted_vs_clean(acc_can, acc_al)
    plot_group_fractions(grp_can_train, grp_al_train, grp_can_test, grp_al_test)

    corrupted_diff = acc_al['acc_corrupted'] - acc_can['acc_corrupted']
    clean_diff = acc_al['acc_clean'] - acc_can['acc_clean']

    if corrupted_diff < -0.01:
        verdict = ("SUPPORTED: AL fits the corrupted points LESS than canonical DRO "
                   "(Delta acc_corrupted = {dc:+.4f}), consistent with 'AL suppresses "
                   "attacked points' denoising.").format(dc=corrupted_diff)
    elif corrupted_diff > 0.01:
        verdict = ("CONTRADICTED: AL fits the corrupted points MORE than canonical DRO "
                   "(Delta acc_corrupted = {dc:+.4f}), opposite to the denoising "
                   "prediction.").format(dc=corrupted_diff)
    else:
        verdict = ("UNCLEAR: AL's accuracy on the corrupted subset is essentially equal "
                   "to canonical DRO (Delta acc_corrupted = {dc:+.4f}), neither supporting "
                   "nor contradicting the denoising hypothesis.").format(dc=corrupted_diff)

    canon_peak_lam = max(hist_can['lambda_dp'])
    al_peak_lam = max(hist_al['lambda_dp'])

    lines = [
        "# TASK B — mechanism study: why does accuracy go UP under AL?",
        "",
        "**Cell:** {ds}, alpha={a}, seed={s}, attack={atk} | "
        "canonical mu={mc} vs AL mu={ma}".format(
            ds=DATASET, a=ALPHA, s=SEED, atk=ATTACK, mc=MU_CANONICAL, ma=MU_AL),
        "**Epochs:** {ep} | K_inner={ki} | PGD steps={pgd}".format(
            ep=EPOCHS, ki=K_INNER, pgd=PGD_STEPS),
        "",
        "## 1. Corrupted-vs-clean accuracy (the key measurement)",
        "",
        "Corrupted samples: {nc}/{nt} ({pct:.1f}% of training data)".format(
            nc=mask.sum(), nt=len(mask), pct=100 * mask.sum() / len(mask)),
        "",
        "| subset | canonical DRO | AL DRO | Delta (AL - canonical) |",
        "|---|---|---|---|",
        "| **corrupted** | {ac:.4f} | {aa:.4f} | **{d:+.4f}** |".format(
            ac=acc_can['acc_corrupted'], aa=acc_al['acc_corrupted'], d=corrupted_diff),
        "| **clean** | {ac:.4f} | {aa:.4f} | {d:+.4f} |".format(
            ac=acc_can['acc_clean'], aa=acc_al['acc_clean'], d=clean_diff),
        "| overall | {ac:.4f} | {aa:.4f} | {d:+.4f} |".format(
            ac=acc_can['acc_all'], aa=acc_al['acc_all'],
            d=acc_al['acc_all'] - acc_can['acc_all']),
        "",
        "**Interpretation:** {interp} (Delta = {d:+.4f}), while clean-subset accuracy "
        "{clean_desc} (Delta = {cd:+.4f}).".format(
            interp=('AL fits corrupted points LESS' if corrupted_diff < -0.01
                    else 'AL fits corrupted points MORE' if corrupted_diff > 0.01
                    else 'AL fits corrupted points EQUALLY'),
            d=corrupted_diff,
            clean_desc=('improves' if clean_diff > 0.01
                        else 'decreases' if clean_diff < -0.01
                        else 'is unchanged'),
            cd=clean_diff),
        "",
        "## 2. g_dp trajectory (does AL suppress g faster?)",
        "",
        "- canonical final g_dp: {v:.4f}".format(v=hist_can['g_dp'][-1]),
        "- AL final g_dp: {v:.4f}".format(v=hist_al['g_dp'][-1]),
        "- AL suppresses g_dp by: {d:.4f}".format(
            d=hist_can['g_dp'][-1] - hist_al['g_dp'][-1]),
        "",
        "![g_dp trajectory](figures/al_mechanism_g_dp.png)",
        "",
        "## 3. lambda_dp trajectory (does AL lambda grow larger?)",
        "",
        "- canonical max lambda_dp: {v:.6f}".format(v=canon_peak_lam),
        "- AL max lambda_dp: {v:.6f}".format(v=al_peak_lam),
        "  (ceiling = 1.5; both starved — lambda*g peak ~ {c:.4f} vs {a:.4f})".format(
            c=canon_peak_lam * np.mean(hist_can['g_dp']),
            a=al_peak_lam * np.mean(hist_al['g_dp'])),
        "",
        "![lambda_dp trajectory](figures/al_mechanism_lambda_dp.png)",
        "",
        "## 4. val_acc trajectory (when does AL pull ahead?)",
        "",
        "- canonical final val_acc: {v:.4f}".format(v=hist_can['val_acc'][-1]),
        "- AL final val_acc: {v:.4f}".format(v=hist_al['val_acc'][-1]),
        "- AL improves val_acc by: {d:+.4f}".format(
            d=hist_al['val_acc'][-1] - hist_can['val_acc'][-1]),
        "",
        "![val_acc trajectory](figures/al_mechanism_val_acc.png)",
        "",
        "## 5. Prediction distribution shift (fraction predicted positive per group)",
        "",
        "### Training (corrupted) data",
        "",
        "| group | canonical DRO | AL DRO | Delta |",
        "|---|---|---|---|",
        "| Group 0 | {c:.4f} | {a:.4f} | {d:+.4f} |".format(
            c=grp_can_train[0], a=grp_al_train[0],
            d=grp_al_train[0] - grp_can_train[0]),
        "| Group 1 | {c:.4f} | {a:.4f} | {d:+.4f} |".format(
            c=grp_can_train[1], a=grp_al_train[1],
            d=grp_al_train[1] - grp_can_train[1]),
        "",
        "### Test (clean) data",
        "",
        "| group | canonical DRO | AL DRO | Delta |",
        "|---|---|---|---|",
        "| Group 0 | {c:.4f} | {a:.4f} | {d:+.4f} |".format(
            c=grp_can_test[0], a=grp_al_test[0],
            d=grp_al_test[0] - grp_can_test[0]),
        "| Group 1 | {c:.4f} | {a:.4f} | {d:+.4f} |".format(
            c=grp_can_test[1], a=grp_al_test[1],
            d=grp_al_test[1] - grp_can_test[1]),
        "",
        "![group fractions](figures/al_mechanism_group_fractions.png)",
        "",
        "## 6. Verdict",
        "",
        "**{v}**".format(v=verdict),
        "",
        "### Reasoning",
        "",
        "The denoising hypothesis predicts AL should fit corrupted points less. "
        "Measured Delta acc_corrupted = {d:+.4f}.".format(d=corrupted_diff),
        "",
        "- Delta acc_corrupted < 0: AL denoises the attack (fits corrupted points less).",
        "- Delta acc_corrupted ~ 0: mechanism unclear.",
        "- Delta acc_corrupted > 0: opposite to denoising.",
        "",
        "**Clean-subset accuracy Delta = {d:+.4f}**: {desc}.".format(
            d=clean_diff,
            desc=('AL improves clean accuracy, suggesting it does not merely sacrifice '
                  'corrupted points.' if clean_diff > 0
                  else 'AL does not improve clean accuracy — the overall gain comes from '
                       'elsewhere.')),
        "",
        "---",
        "",
        "*Generated by experiments/summarize_mechanism.py | {ts}*".format(
            ts=time.strftime('%Y-%m-%d %H:%M:%S')),
    ]

    with open('results/al_mechanism_summary.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Corrupted accuracy: canonical={:.4f}, AL={:.4f}, Delta={:+.4f}".format(
        acc_can['acc_corrupted'], acc_al['acc_corrupted'], corrupted_diff))
    print("Clean accuracy:     canonical={:.4f}, AL={:.4f}, Delta={:+.4f}".format(
        acc_can['acc_clean'], acc_al['acc_clean'], clean_diff))
    print("Verdict: {}".format(verdict))
    print("\nOutputs:")
    print("  results/history_mechanism_canonical.json")
    print("  results/history_mechanism_al.json")
    print("  results/al_mechanism_summary.md")
    print("  figures/al_mechanism_*.png")


if __name__ == '__main__':
    main()
