#!/usr/bin/env python3
"""
Pixel-space PGD UTKFace experiment — server-ready (tests Hypothesis H2).

H2: feature-space attacks on cached 512-d ResNet outputs don't simulate
realistic corruption. This script attacks RAW PIXELS through ResNet18,
then refeeds the perturbed-pixel features through ResNet to get the
adversarial features used by the downstream Naive/DRO trainers.

Comparison vs feature-space (run_utkface.py): same alpha, same seeds,
different attack surface.

Run on flair2.iitgn.ac.in:
    cd /data/srujan.sai/DRO-FairML
    venv/bin/python3 experiments/run_utkface_pixel_pgd.py \
        --data_dir /data/srujan.sai/UTKFace \
        --n_seeds 5 --alphas 0.1 0.2

Expected wall time: ~25 min/alpha/seed on L40S (PGD over 24k images is
the bottleneck). Use --max_images 8000 for a faster sanity check.
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


def build_resnet18_backbone(device):
    m = models.resnet18(pretrained=True)
    m = nn.Sequential(*list(m.children())[:-1])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device)


class PixelToLogit(nn.Module):
    """Wrap ResNet18 + MLP head for end-to-end gradient through pixels."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.head(feat)


def load_utkface_pixels(data_dir, max_images=None, seed=42):
    files = sorted(glob.glob(os.path.join(data_dir, '*.jpg.chip.jpg')))
    if not files:
        raise RuntimeError(f"No UTKFace images in {data_dir}")
    rng = np.random.RandomState(seed)
    if max_images and len(files) > max_images:
        files = list(rng.choice(files, size=max_images, replace=False))
    metas = []
    keep = []
    for f in files:
        parts = os.path.basename(f).split('_')
        if len(parts) < 3:
            continue
        try:
            gender = int(parts[1])
            race = int(parts[2])
        except ValueError:
            continue
        keep.append(f)
        metas.append((gender, race))
    y = np.array([m[0] for m in metas], dtype=np.float32)
    a = np.array([m[0] for m in metas], dtype=np.int64)  # gender as protected, matches v1
    return keep, y, a


def pgd_attack_pixels(model, X_pix, y, eps=4/255, steps=10, step_size=1/255, device='cpu'):
    """Untargeted PGD on classification loss in pixel space, with epsilon-ball + [0,1] clipping."""
    X_adv = X_pix.clone().detach().to(device)
    X_min = (X_pix - eps).clamp(0, 1).to(device)
    X_max = (X_pix + eps).clamp(0, 1).to(device)
    X_adv = (X_adv + torch.empty_like(X_adv).uniform_(-eps, eps)).clamp(X_min, X_max)
    for _ in range(steps):
        X_adv = X_adv.detach().requires_grad_(True)
        logits = model(X_adv).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + step_size * X_adv.grad.sign()
            X_adv = torch.max(torch.min(X_adv, X_max), X_min)
    return X_adv.detach()


def to_pixel_tensor(paths, transform, batch=64, device='cpu'):
    out = []
    for i in range(0, len(paths), batch):
        chunk = [transform(Image.open(p).convert('RGB')) for p in paths[i:i+batch]]
        out.append(torch.stack(chunk))
    return torch.cat(out, dim=0).to(device)


def features_from_pixels(backbone, X_pix, batch=128, device='cpu'):
    feats = []
    with torch.no_grad():
        for i in range(0, X_pix.shape[0], batch):
            f = backbone(X_pix[i:i+batch].to(device)).flatten(1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/data/srujan.sai/UTKFace')
    p.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.2])
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--max_images', type=int, default=None)
    p.add_argument('--pgd_steps', type=int, default=10)
    p.add_argument('--pgd_eps', type=float, default=4/255)
    p.add_argument('--out', default='results/utkface_pixel_pgd.json')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  data_dir={args.data_dir}  eps={args.pgd_eps}  steps={args.pgd_steps}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    files, y_all, a_all = load_utkface_pixels(args.data_dir, args.max_images, seed=42)
    print(f"Loaded {len(files)} UTKFace images")
    backbone = build_resnet18_backbone(device)

    runs = []
    os.makedirs('results', exist_ok=True)

    for s in range(args.n_seeds):
        set_seed(s)
        idx_tv, idx_te = train_test_split(np.arange(len(files)), test_size=0.2,
                                          random_state=s, stratify=y_all)
        idx_tr, idx_v = train_test_split(idx_tv, test_size=0.1875,
                                         random_state=s, stratify=y_all[idx_tv])

        for alpha in args.alphas:
            tau = get_temperature(alpha)
            print(f"\n[pixel_pgd] seed={s} alpha={alpha}")

            # 1) Train a clean-feature classifier head to define the PGD target loss.
            t0 = time.time()
            X_tr_pix = to_pixel_tensor([files[i] for i in idx_tr], transform, device=device)
            X_v_pix  = to_pixel_tensor([files[i] for i in idx_v],  transform, device=device)
            X_te_pix = to_pixel_tensor([files[i] for i in idx_te], transform, device=device)
            X_tr_pix_n = normalize(X_tr_pix)
            X_v_pix_n  = normalize(X_v_pix)
            X_te_pix_n = normalize(X_te_pix)

            X_tr = features_from_pixels(backbone, X_tr_pix_n, device=device)
            X_v  = features_from_pixels(backbone, X_v_pix_n,  device=device)
            X_te = features_from_pixels(backbone, X_te_pix_n, device=device)
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr).astype(np.float32)
            X_v_s  = scaler.transform(X_v).astype(np.float32)
            X_te_s = scaler.transform(X_te).astype(np.float32)

            head = MLPClassifier(X_tr_s.shape[1], hidden_dims=[128, 64], dropout=0.1).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
            y_tr_t = torch.tensor(y_all[idx_tr], dtype=torch.float32, device=device)
            X_tr_st = torch.tensor(X_tr_s, device=device)
            for _ in range(30):
                logits = head(X_tr_st).squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y_tr_t)
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            print(f"  head trained in {time.time()-t0:.0f}s")

            # 2) Pick top alpha*n train images by classification margin → attack their pixels.
            n_corrupt = int(alpha * len(idx_tr))
            with torch.no_grad():
                margins = head(torch.tensor(X_tr_s, device=device)).squeeze(-1).cpu().numpy()
            confidence = np.abs(margins)
            target_local = np.argsort(-confidence)[:n_corrupt]

            tp = time.time()
            full_model = PixelToLogit(backbone, head).to(device).eval()
            X_attacked_pix = X_tr_pix.clone()
            y_tr_arr = y_all[idx_tr]
            BATCH = 64
            for start in range(0, len(target_local), BATCH):
                idx = target_local[start:start+BATCH]
                X_chunk = X_tr_pix[idx]
                y_chunk = torch.tensor(y_tr_arr[idx], dtype=torch.float32, device=device)
                X_chunk_n_adv = pgd_attack_pixels(
                    PixelToLogit(backbone, head).to(device).eval(),
                    normalize(X_chunk),
                    y_chunk,
                    eps=args.pgd_eps, steps=args.pgd_steps,
                    step_size=args.pgd_eps / 4, device=device)
                # unnormalize back to [0,1] pixel space for re-featurization
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                X_chunk_adv = (X_chunk_n_adv * std + mean).clamp(0, 1)
                X_attacked_pix[idx] = X_chunk_adv.cpu()
            print(f"  pixel PGD done in {time.time()-tp:.0f}s")

            # 3) Re-extract features from attacked pixels.
            X_tr_adv = features_from_pixels(backbone, normalize(X_attacked_pix), device=device)
            X_tr_adv_s = scaler.transform(X_tr_adv).astype(np.float32)

            # 4) Train naive + DRO on pixel-attacked features.
            y_tr_a = y_all[idx_tr].copy()
            a_tr_a = a_all[idx_tr].copy()
            y_v = y_all[idx_v]; a_v = a_all[idx_v]
            y_te = y_all[idx_te]; a_te = a_all[idx_te]

            m_n = MLPClassifier(X_tr_adv_s.shape[1], hidden_dims=[128, 64], dropout=0.1)
            t_n = NaiveFairTrainer(m_n, device=device, lr_theta=1e-3, lr_lambda=5e-3,
                                   lambda_max=1.5, tau=tau, k=5, gamma=0.0,
                                   epochs=60, weight_decay=1e-4, tau_warmup_epochs=15)
            t_n.fit(X_tr_adv_s, y_tr_a, a_tr_a, X_val=X_v_s, y_val=y_v, a_val=a_v, verbose=False)
            naive = compute_metrics_torch(t_n.model, X_te_s, y_te, a_te,
                                          device=device, temperature=tau, k=5, gamma=0.0)

            m_d = MLPClassifier(X_tr_adv_s.shape[1], hidden_dims=[128, 64], dropout=0.1)
            t_d = DroFairTrainer(m_d, alpha=alpha, device=device,
                                 lr_theta=1e-3, lr_lambda=5e-3, lr_p=5e-3, lambda_max=1.5,
                                 tau=tau, beta=5.0, k=5, gamma=0.0,
                                 K_inner=10, epochs=60, weight_decay=1e-4,
                                 tau_warmup_epochs=15, lambda_warmstart=0.01)
            hist = t_d.fit(X_tr_adv_s, y_tr_a, a_tr_a, X_val=X_v_s, y_val=y_v, a_val=a_v, verbose=False)
            dro = compute_metrics_torch(t_d.model, X_te_s, y_te, a_te,
                                        device=device, temperature=tau, k=5, gamma=0.0)

            runs.append({
                'seed': s, 'alpha': alpha, 'attack': 'pixel_pgd',
                'naive': {k: float(v) for k, v in naive.items()},
                'dro':   {k: float(v) for k, v in dro.items()},
                'dro_lambda_dp_history': hist['lambda_dp'],
                'elapsed_total': time.time() - t0,
            })
            print(f"  naive dp={naive['dp_violation']:.4f}  dro dp={dro['dp_violation']:.4f}  "
                  f"(total {time.time()-t0:.0f}s)")
            with open(args.out, 'w') as f:
                json.dump(runs, f, indent=2)

    print(f"\nDone. {len(runs)} runs -> {args.out}")


if __name__ == '__main__':
    main()
