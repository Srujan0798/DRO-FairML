#!/usr/bin/env python3
"""
Random-init backbone UTKFace experiment — server-ready (tests Hypothesis H1).

H1: ImageNet-pretrained ResNet18 features are gender-agnostic, so DRO's
worst-case reweighting has no demographic axis to anchor on. If we instead
train a ResNet18 *from scratch* on the UTKFace gender task, the resulting
features should carry strong demographic signal, and DRO should stop
inverting.

Pipeline (per seed):
  1. Train ResNet18 backbone end-to-end on gender prediction (no ImageNet init).
  2. Freeze backbone, extract 512-d features for all images.
  3. Apply AdversarialCorruptor at alpha in {0.1, 0.2}.
  4. Train Naive + DRO heads on attacked features, evaluate on clean test set.
  5. Compare: does DRO still invert?

Run on flair2.iitgn.ac.in:
    cd /data/srujan.sai/DRO-FairML && git pull
    venv/bin/python3 experiments/run_utkface_randinit.py \
        --data_dir /data/srujan.sai/UTKFace --n_seeds 5 --alphas 0.1 0.2

Expected wall time: ~30 min/seed on L40S (backbone training dominates).
Use --backbone_epochs 5 --max_images 8000 for a faster sanity check.
"""
import os
import sys
import json
import time
import glob
import argparse
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from src.models.classifier import MLPClassifier
from src.corruption.adversarial import AdversarialCorruptor
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.temperature import get_temperature


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_utkface_meta(data_dir, max_images=None, seed=42):
    files = sorted(glob.glob(os.path.join(data_dir, '*.jpg.chip.jpg')))
    if not files:
        raise RuntimeError(f"No UTKFace images in {data_dir}")
    rng = np.random.RandomState(seed)
    if max_images and len(files) > max_images:
        files = list(rng.choice(files, size=max_images, replace=False))
    keep, y_list, a_list = [], [], []
    for f in files:
        parts = os.path.basename(f).split('_')
        if len(parts) < 3:
            continue
        try:
            gender = int(parts[1])
        except ValueError:
            continue
        keep.append(f)
        y_list.append(gender)
        a_list.append(gender)  # use gender as protected attribute (matches v1)
    return keep, np.array(y_list, dtype=np.float32), np.array(a_list, dtype=np.int64)


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, files, labels, transform):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.transform(Image.open(self.files[i]).convert('RGB'))
        return img, float(self.labels[i])


class ResNet18FromScratch(nn.Module):
    """ResNet18 with random init, single sigmoid head for binary gender."""
    def __init__(self):
        super().__init__()
        # weights=None => random init (no ImageNet pretraining)
        self.backbone = models.resnet18(weights=None)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_feat, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)

    def features(self, x):
        return self.backbone(x)


def train_backbone(net, files_tr, y_tr, files_v, y_v, transform,
                   epochs, batch_size, lr, device):
    train_ds = FaceDataset(files_tr, y_tr, transform)
    val_ds = FaceDataset(files_v, y_v, transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                           num_workers=4, pin_memory=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()

    net.train()
    for ep in range(epochs):
        tot, correct, loss_sum = 0, 0, 0.0
        for X, y in train_dl:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = net(X)
            loss = bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * X.size(0)
            correct += ((torch.sigmoid(logits) >= 0.5).float() == y).sum().item()
            tot += X.size(0)
        sched.step()

        net.eval()
        v_tot, v_correct = 0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = net(X)
                v_correct += ((torch.sigmoid(logits) >= 0.5).float() == y).sum().item()
                v_tot += X.size(0)
        net.train()
        print(f"    backbone epoch {ep+1}/{epochs} "
              f"train_loss={loss_sum/tot:.4f} train_acc={correct/tot:.3f} "
              f"val_acc={v_correct/v_tot:.3f}")


def extract_features(net, files, transform, batch_size, device):
    ds = FaceDataset(files, np.zeros(len(files), dtype=np.float32), transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=4, pin_memory=True)
    feats = []
    net.eval()
    with torch.no_grad():
        for X, _ in dl:
            X = X.to(device, non_blocking=True)
            feats.append(net.features(X).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/data/srujan.sai/UTKFace')
    p.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.2])
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--max_images', type=int, default=None)
    p.add_argument('--backbone_epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--backbone_lr', type=float, default=1e-3)
    p.add_argument('--out', default='results/utkface_randinit.json')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  data_dir={args.data_dir}")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    files, y_all, a_all = load_utkface_meta(args.data_dir, args.max_images, seed=42)
    print(f"Loaded {len(files)} UTKFace images")

    runs = []
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    for s in range(args.n_seeds):
        set_seed(s)
        print(f"\n=== seed={s} ===")
        idx_tv, idx_te = train_test_split(np.arange(len(files)), test_size=0.2,
                                          random_state=s, stratify=y_all)
        idx_tr, idx_v = train_test_split(idx_tv, test_size=0.1875,
                                         random_state=s, stratify=y_all[idx_tv])

        # 1) train from-scratch ResNet18 on the gender task
        net = ResNet18FromScratch().to(device)
        t0 = time.time()
        train_backbone(
            net,
            files_tr=[files[i] for i in idx_tr], y_tr=y_all[idx_tr],
            files_v=[files[i] for i in idx_v],   y_v=y_all[idx_v],
            transform=transform_train,
            epochs=args.backbone_epochs, batch_size=args.batch_size,
            lr=args.backbone_lr, device=device)
        print(f"  backbone trained in {time.time()-t0:.0f}s")

        # 2) extract 512-d features for all splits
        t0 = time.time()
        X_tr = extract_features(net, [files[i] for i in idx_tr], transform_eval,
                                args.batch_size, device)
        X_v  = extract_features(net, [files[i] for i in idx_v],  transform_eval,
                                args.batch_size, device)
        X_te = extract_features(net, [files[i] for i in idx_te], transform_eval,
                                args.batch_size, device)
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr).astype(np.float32)
        X_v  = scaler.transform(X_v).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)
        y_tr = y_all[idx_tr]; a_tr = a_all[idx_tr]
        y_v  = y_all[idx_v];  a_v  = a_all[idx_v]
        y_te = y_all[idx_te]; a_te = a_all[idx_te]
        print(f"  features extracted in {time.time()-t0:.0f}s "
              f"(dim={X_tr.shape[1]}, train n={X_tr.shape[0]})")

        # 3) per-alpha attack + Naive vs DRO heads
        for alpha in args.alphas:
            tau = get_temperature(alpha)
            corr = AdversarialCorruptor(
                alpha=alpha, epsilon=0.1,
                feature_attack=True, label_flip=True, attr_flip=True,
                coordinated=True, random_state=s)
            X_tr_c, y_tr_c, a_tr_c, _ = corr.corrupt(X_tr, y_tr, a_tr,
                                                     model=None, device='cpu')

            m_n = MLPClassifier(X_tr.shape[1], hidden_dims=[128, 64], dropout=0.1)
            t_n = NaiveFairTrainer(m_n, device=device, lr_theta=1e-3, lr_lambda=5e-3,
                                   lambda_max=1.5, tau=tau, k=5, gamma=0.0,
                                   epochs=60, weight_decay=1e-4, tau_warmup_epochs=15)
            t_n.fit(X_tr_c, y_tr_c, a_tr_c, X_val=X_v, y_val=y_v, a_val=a_v, verbose=False)
            naive = compute_metrics_torch(t_n.model, X_te, y_te, a_te,
                                          device=device, temperature=tau, k=5, gamma=0.0)

            m_d = MLPClassifier(X_tr.shape[1], hidden_dims=[128, 64], dropout=0.1)
            t_d = DroFairTrainer(m_d, alpha=alpha, device=device,
                                 lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=1.5,
                                 tau=tau, beta=5.0, k=5, gamma=0.0,
                                 K_inner=10, epochs=60, weight_decay=1e-4,
                                 tau_warmup_epochs=15, lambda_warmstart=0.01)
            hist = t_d.fit(X_tr_c, y_tr_c, a_tr_c, X_val=X_v, y_val=y_v, a_val=a_v, verbose=False)
            dro = compute_metrics_torch(t_d.model, X_te, y_te, a_te,
                                        device=device, temperature=tau, k=5, gamma=0.0)

            runs.append({
                'seed': s, 'alpha': alpha,
                'backbone': 'resnet18_randinit',
                'naive': {k: float(v) for k, v in naive.items()},
                'dro':   {k: float(v) for k, v in dro.items()},
                'dro_lambda_dp_history': hist['lambda_dp'],
            })
            print(f"  alpha={alpha}  naive dp={naive['dp_violation']:.4f}  "
                  f"dro dp={dro['dp_violation']:.4f}")
            with open(args.out, 'w') as f:
                json.dump(runs, f, indent=2)

    print(f"\nDone. {len(runs)} runs -> {args.out}")


if __name__ == '__main__':
    main()
