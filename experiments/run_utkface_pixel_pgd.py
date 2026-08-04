#!/usr/bin/env python3
"""
U3 — Pixel-space PGD on raw UTKFace JPEGs (not cached 512-d features).

Restored/adapted from the server-ready H2 script (commit 4c76c9f) with
canonical lock-in:
  - tau=1.0 fixed (no stepped τ=100)
  - K_inner=10, epochs=60
  - protected attr = race binary (White vs non-White), label = gender
  - resume-safe JSON; full provenance on every row
  - default n_seeds=6, alphas 0.1 0.2

flair2 (after U1/U2 free a GPU; images linked):
  CONFIRM=1 bash scripts/flair2_link_utkface_images.sh
  CUDA_VISIBLE_DEVICES=0 ./venv_gpu/bin/python experiments/run_utkface_pixel_pgd.py \\
    --data_dir /data/srujan.sai/UTKFace --n_seeds 6 --alphas 0.1 0.2 \\
    --out results/utkface_pixel_pgd.json

Smoke (few images):
  python experiments/run_utkface_pixel_pgd.py --max_images 256 --n_seeds 1 \\
    --alphas 0.1 --epochs 3 --pgd_steps 2 --device cpu
"""
from __future__ import annotations

import os
import sys
import json
import time
import glob
import argparse
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.classifier import MLPClassifier
from src.training.naive_fair import NaiveFairTrainer
from src.training.dro_fair import DroFairTrainer
from src.evaluation.metrics import compute_metrics_torch


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_resnet18_backbone(device: str):
    """Frozen ResNet18 trunk. Prefer cached ImageNet weights; offline → random init warning."""
    try:
        weights = models.ResNet18_Weights.DEFAULT
        m = models.resnet18(weights=weights)
    except Exception as e:
        print(f"WARNING: could not load ImageNet weights ({e}); using random init", flush=True)
        m = models.resnet18(weights=None)
    m = nn.Sequential(*list(m.children())[:-1])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device)


class PixelToLogit(nn.Module):
    """ResNet18 trunk + MLP head for end-to-end gradient through pixels."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.head(feat)


def load_utkface_pixels(data_dir, max_images=None, seed=42):
    """Load JPEG paths + gender labels + race-binary protected attr."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.jpg.chip.jpg")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    if not files:
        raise RuntimeError(f"No UTKFace images in {data_dir}")
    rng = np.random.RandomState(seed)
    if max_images and len(files) > max_images:
        files = list(rng.choice(files, size=max_images, replace=False))
    keep, y_list, a_list, race_list = [], [], [], []
    for f in files:
        parts = os.path.basename(f).split("_")
        if len(parts) < 3:
            continue
        try:
            gender = int(parts[1])
            race = int(parts[2])
        except ValueError:
            continue
        if race < 0 or race > 4:
            continue
        keep.append(f)
        y_list.append(gender)
        a_list.append(0 if race == 0 else 1)  # White vs non-White
        race_list.append(race)
    y = np.array(y_list, dtype=np.float32)
    a = np.array(a_list, dtype=np.int64)
    race = np.array(race_list, dtype=np.int64)
    return keep, y, a, race


def pgd_attack_pixels(model, X_pix, y, eps=4 / 255, steps=10, step_size=1 / 255, device="cpu"):
    """Untargeted PGD on BCE logits in normalized pixel space, clip to eps-ball ∩ [0,1] preimage."""
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


def to_pixel_tensor(paths, transform, batch=64, device="cpu"):
    out = []
    for i in range(0, len(paths), batch):
        chunk = [transform(Image.open(p).convert("RGB")) for p in paths[i : i + batch]]
        out.append(torch.stack(chunk))
    return torch.cat(out, dim=0).to(device)


def features_from_pixels(backbone, X_pix, batch=128, device="cpu"):
    feats = []
    with torch.no_grad():
        for i in range(0, X_pix.shape[0], batch):
            f = backbone(X_pix[i : i + batch].to(device)).flatten(1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="U3 pixel-space PGD UTKFace (canonical τ=1)")
    p.add_argument("--data_dir", default="/data/srujan.sai/UTKFace")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2])
    p.add_argument("--n_seeds", type=int, default=6)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--pgd_eps", type=float, default=4 / 255)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--k_inner", type=int, default=10)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/utkface_pixel_pgd.json")
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(
        f"U3 pixel_pgd device={device} data_dir={args.data_dir} "
        f"tau={args.tau} k_inner={args.k_inner} epochs={args.epochs} "
        f"eps={args.pgd_eps} pgd_steps={args.pgd_steps}",
        flush=True,
    )

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def normalize(x):
        return (x - mean_t.to(x.device)) / std_t.to(x.device)

    files, y_all, a_all, race_all = load_utkface_pixels(
        args.data_dir, args.max_images, seed=42
    )
    print(f"Loaded {len(files)} images; race counts={np.bincount(race_all, minlength=5).tolist()}", flush=True)
    backbone = build_resnet18_backbone(device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    done = set()
    if out_path.exists():
        runs = json.loads(out_path.read_text())
        done = {(int(r["seed"]), float(r["alpha"])) for r in runs}
        print(f"resume: {len(runs)} existing rows", flush=True)

    for s in range(args.n_seeds):
        set_seed(s)
        idx_tv, idx_te = train_test_split(
            np.arange(len(files)), test_size=0.2, random_state=s, stratify=y_all
        )
        idx_tr, idx_v = train_test_split(
            idx_tv, test_size=0.1875, random_state=s, stratify=y_all[idx_tv]
        )

        for alpha in args.alphas:
            key = (s, float(alpha))
            if key in done:
                continue
            print(f"\n[pixel_pgd] seed={s} alpha={alpha}", flush=True)
            t0 = time.time()

            X_tr_pix = to_pixel_tensor([files[i] for i in idx_tr], transform, device=device)
            X_v_pix = to_pixel_tensor([files[i] for i in idx_v], transform, device=device)
            X_te_pix = to_pixel_tensor([files[i] for i in idx_te], transform, device=device)

            X_tr = features_from_pixels(backbone, normalize(X_tr_pix), device=device)
            X_v = features_from_pixels(backbone, normalize(X_v_pix), device=device)
            X_te = features_from_pixels(backbone, normalize(X_te_pix), device=device)
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr).astype(np.float32)
            X_v_s = scaler.transform(X_v).astype(np.float32)
            X_te_s = scaler.transform(X_te).astype(np.float32)

            head = MLPClassifier(X_tr_s.shape[1], hidden_dims=[128, 64], dropout=0.1).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
            y_tr_t = torch.tensor(y_all[idx_tr], dtype=torch.float32, device=device)
            X_tr_st = torch.tensor(X_tr_s, device=device)
            for _ in range(30):
                logits = head(X_tr_st).squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y_tr_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
            head.eval()

            n_corrupt = max(1, int(alpha * len(idx_tr)))
            with torch.no_grad():
                margins = head(torch.tensor(X_tr_s, device=device)).squeeze(-1).cpu().numpy()
            target_local = np.argsort(-np.abs(margins))[:n_corrupt]

            X_attacked_pix = X_tr_pix.clone()
            y_tr_arr = y_all[idx_tr]
            BATCH = 64
            for start in range(0, len(target_local), BATCH):
                idx = target_local[start : start + BATCH]
                X_chunk = X_tr_pix[idx]
                y_chunk = torch.tensor(y_tr_arr[idx], dtype=torch.float32, device=device)
                X_chunk_n_adv = pgd_attack_pixels(
                    PixelToLogit(backbone, head).to(device).eval(),
                    normalize(X_chunk),
                    y_chunk,
                    eps=args.pgd_eps,
                    steps=args.pgd_steps,
                    step_size=args.pgd_eps / 4,
                    device=device,
                )
                X_chunk_adv = (X_chunk_n_adv * std_t.to(device) + mean_t.to(device)).clamp(0, 1)
                X_attacked_pix[idx] = X_chunk_adv.cpu() if X_attacked_pix.device.type == "cpu" else X_chunk_adv

            X_tr_adv = features_from_pixels(backbone, normalize(X_attacked_pix), device=device)
            X_tr_adv_s = scaler.transform(X_tr_adv).astype(np.float32)

            y_tr_a = y_all[idx_tr].copy()
            a_tr_a = a_all[idx_tr].copy()
            y_v, a_v = y_all[idx_v], a_all[idx_v]
            y_te, a_te = y_all[idx_te], a_all[idx_te]

            m_n = MLPClassifier(X_tr_adv_s.shape[1], hidden_dims=[128, 64], dropout=0.1)
            t_n = NaiveFairTrainer(
                m_n,
                device=device,
                lr_theta=1e-3,
                lr_lambda=5e-3,
                lambda_max=1.5,
                tau=args.tau,
                k=5,
                gamma=0.0,
                epochs=args.epochs,
                weight_decay=1e-4,
                tau_warmup_epochs=15,
            )
            t_n.fit(X_tr_adv_s, y_tr_a, a_tr_a, X_val=X_v_s, y_val=y_v, a_val=a_v, verbose=False)
            naive = compute_metrics_torch(
                t_n.model, X_te_s, y_te, a_te, device=device, temperature=args.tau, k=5, gamma=0.0
            )

            m_d = MLPClassifier(X_tr_adv_s.shape[1], hidden_dims=[128, 64], dropout=0.1)
            t_d = DroFairTrainer(
                m_d,
                alpha=alpha,
                device=device,
                lr_theta=1e-3,
                lr_lambda=5e-3,
                lr_p=5e-3,
                lambda_max=1.5,
                tau=args.tau,
                beta=5.0,
                k=5,
                gamma=0.0,
                K_inner=args.k_inner,
                epochs=args.epochs,
                weight_decay=1e-4,
                tau_warmup_epochs=15,
                lambda_warmstart=0.01,
            )
            t_d.fit(X_tr_adv_s, y_tr_a, a_tr_a, X_val=X_v_s, y_val=y_v, a_val=a_v, verbose=False)
            dro = compute_metrics_torch(
                t_d.model, X_te_s, y_te, a_te, device=device, temperature=args.tau, k=5, gamma=0.0
            )

            row = {
                "dataset": "utkface",
                "seed": s,
                "alpha": float(alpha),
                "attack": "pixel_pgd",
                "device": device,
                "tau": float(args.tau),
                "k_inner": int(args.k_inner),
                "epochs": int(args.epochs),
                "pgd_steps": int(args.pgd_steps),
                "pgd_eps": float(args.pgd_eps),
                "n_seeds_planned": int(args.n_seeds),
                "n_images": len(files),
                "data_dir": str(args.data_dir),
                "data_provenance": "REAL_PIXELS",
                "protected_def": "race_binary_White_vs_nonWhite",
                "label_def": "gender",
                "naive": {k: float(v) for k, v in naive.items()},
                "dro": {k: float(v) for k, v in dro.items()},
                "total_time": time.time() - t0,
            }
            runs.append(row)
            done.add(key)
            out_path.write_text(json.dumps(runs, indent=2))
            print(
                f"  naive dp={naive.get('dp_violation', float('nan')):.4f} "
                f"dro dp={dro.get('dp_violation', float('nan')):.4f} "
                f"({row['total_time']:.0f}s) saved {len(runs)}",
                flush=True,
            )

    print(f"\nDone. {len(runs)} runs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
