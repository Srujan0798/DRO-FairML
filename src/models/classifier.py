"""Simple MLP classifier for binary classification."""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def predict_proba(self, x, temperature=1.0):
        """Return soft predictions using sigmoid with temperature."""
        logits = self.forward(x)
        return torch.sigmoid(logits / temperature)
    
    def predict(self, x):
        """Return hard binary predictions (threshold at 0.5)."""
        probs = torch.sigmoid(self.forward(x))
        return (probs >= 0.5).float()
