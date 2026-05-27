"""CNN classifier using ResNet18 backbone for image classification."""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNClassifier(nn.Module):
    """
    CNN classifier using ResNet18 backbone with custom FC head.

    Args:
        input_channels: number of input channels (3 for RGB images)
        hidden_dim: dimension of hidden FC layer (default 512, matches ResNet18 output)
        num_classes: number of output classes (default 1 for binary)
    """

    def __init__(self, input_channels=3, hidden_dim=512, num_classes=1):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input tensor of shape (N, C, H, W) where C=input_channels

        Returns:
            logits: raw scores of shape (N,) (squeeze last dim)
        """
        return self.resnet(x).squeeze(-1)

    def predict_proba(self, x, temperature=1.0):
        """Return soft predictions using sigmoid with temperature."""
        logits = self.forward(x)
        return torch.sigmoid(logits * temperature)

    def predict(self, x):
        """Return hard binary predictions (threshold at 0.5)."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        probs = torch.sigmoid(self.forward(x))
        return (probs >= 0.5).cpu().numpy()