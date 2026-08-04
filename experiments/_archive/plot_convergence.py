"""
Generate convergence curves showing training dynamics for Naive-FAIR vs DRO-FAIR.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.datasets import get_dataset
from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer

sns.set_style('whitegrid')


def plot_convergence(dataset='adult', alpha=0.2, seed=42, output_dir='figures'):
    """Plot training convergence for one setting."""
    os.makedirs(output_dir, exist_ok=True)
    
    X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname = \
        get_dataset(dataset, random_state=seed)
    
    corruptor = AdversarialCorruptor(alpha=alpha, coordinated=True, random_state=seed)
    X_tr, y_tr, a_tr, _ = corruptor.corrupt(X_train, y_train, a_train)
    X_va, y_va, a_va, _ = corruptor.corrupt(X_val, y_val, a_val)
    
    input_dim = X_train.shape[1]
    
    # Naive-FAIR
    model_naive = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer_naive = NaiveFairTrainer(model_naive, device='cpu', epochs=30, tau=1.0, k=5,
                                     lambda_max=2.0, tau_warmup_epochs=0)
    hist_naive = trainer_naive.fit(X_tr, y_tr, a_tr, X_va, y_va, a_va, verbose=False)

    # DRO-FAIR
    model_dro = MLPClassifier(input_dim, hidden_dims=[128, 64], dropout=0.1)
    trainer_dro = DroFairTrainer(model_dro, alpha=alpha, device='cpu', epochs=30, tau=1.0, k=5,
                                 lambda_max=2.0, K_inner=10, lr_p=5e-3, tau_warmup_epochs=0)
    hist_dro = trainer_dro.fit(X_tr, y_tr, a_tr, X_va, y_va, a_va, verbose=False)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Train loss
    ax = axes[0]
    ax.plot(hist_naive['train_loss'], label='Naive-FAIR', color='C0')
    ax.plot(hist_dro['train_loss'], label='DRO-FAIR', color='C1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Val accuracy
    ax = axes[1]
    val_epochs = list(range(5, 31, 5))
    ax.plot(val_epochs, hist_naive['val_acc'], marker='o', label='Naive-FAIR', color='C0')
    ax.plot(val_epochs, hist_dro['val_acc'], marker='s', label='DRO-FAIR', color='C1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Val DP
    ax = axes[2]
    ax.plot(val_epochs, hist_naive['val_dp'], marker='o', label='Naive-FAIR', color='C0')
    ax.plot(val_epochs, hist_dro['val_dp'], marker='s', label='DRO-FAIR', color='C1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation DP Violation')
    ax.set_title('Validation DP Violation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset.upper()} α={alpha} (seed={seed})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'convergence_{dataset}_a{alpha}.png'), dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir}/convergence_{dataset}_a{alpha}.png")
    plt.close()


if __name__ == '__main__':
    for dataset in ['adult', 'credit']:
        for alpha in [0.2, 0.3]:
            plot_convergence(dataset, alpha, seed=42, output_dir='figures')
