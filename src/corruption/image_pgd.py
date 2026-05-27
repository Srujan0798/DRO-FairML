"""
Image PGD attack module for adversarial corruption of image data.
Standard PGD attack on classification loss for images.
"""

import torch
import torch.nn as nn


class ImagePGD:
    """
    Projected Gradient Descent attack on image classifiers.

    Applies adversarial perturbations to images within an epsilon ball,
    maximizing classification loss.

    Args:
        epsilon: maximum perturbation magnitude (default 4/255 ≈ 0.0157)
        steps: number of PGD steps (default 10)
        step_size: step size for each PGD update (default 1/255 ≈ 0.0039)
        random_start: whether to start from random point within epsilon ball
    """

    def __init__(self, epsilon=4/255, steps=10, step_size=1/255, random_start=True):
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.random_start = random_start

    def attack(self, model, X, y, device='cpu'):
        """
        Apply PGD attack to images.

        Args:
            model: classifier that takes (N, C, H, W) input and returns logits
            X: images tensor of shape (N, C, H, W) in range [0, 1]
            y: labels tensor of shape (N,)
            device: torch device

        Returns:
            X_adv: adversarial images in range [0, 1]
        """
        model.eval()

        X_adv = X.clone().detach().to(device)
        X_min = X_adv - self.epsilon
        X_max = X_adv + self.epsilon

        if self.random_start:
            X_adv = X_adv + torch.zeros_like(X_adv).uniform_(-self.epsilon, self.epsilon)
            X_adv = torch.clamp(X_adv, 0, 1)

        for _ in range(self.steps):
            X_adv = X_adv.detach().requires_grad_(True)

            logits = model(X_adv)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)

            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                X_adv = X_adv + self.step_size * torch.sign(X_adv.grad)
                X_adv = torch.maximum(torch.minimum(X_adv, torch.tensor(X_max)), torch.tensor(X_min))
                X_adv = torch.clamp(X_adv, 0, 1)

        return X_adv

    def __call__(self, model, X, y, device='cpu'):
        """Alias for attack method."""
        return self.attack(model, X, y, device)