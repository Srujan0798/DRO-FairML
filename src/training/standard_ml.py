"""
Standard ML baseline: trains a classifier with NO fairness constraints.
Useful for measuring the accuracy-fairness tradeoff.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardMLTrainer:
    """Standard neural network trainer (no fairness constraints)."""
    
    def __init__(self, model, device='cpu', lr=1e-3, epochs=30,
                 weight_decay=1e-4, batch_size=256):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        
    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        """Train standard classifier."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        history = {'train_loss': [], 'val_acc': []}
        
        for epoch in range(self.epochs):
            self.model.train()
            logits = self.model(X_t)
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            history['train_loss'].append(loss.item())
            
            if X_val is not None and (epoch + 1) % 5 == 0:
                from src.evaluation.metrics import compute_accuracy
                preds = self.predict(X_val)
                acc = compute_accuracy(y_val, preds)
                history['val_acc'].append(acc)
                if verbose:
                    print(f"Epoch {epoch+1}/{self.epochs}: loss={loss.item():.4f}, val_acc={acc:.4f}")
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            probs = torch.sigmoid(self.model(X_t))
            return (probs >= 0.5).cpu().numpy()
