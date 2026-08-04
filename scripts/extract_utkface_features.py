#!/usr/bin/env python3
"""
Extract ResNet18 features from UTKFace images.

Usage:
    python scripts/extract_utkface_features.py \
        --data-dir data/raw/utkface/UTKFace \
        --output data/raw/utkface_features.npz

Supports CUDA, MPS (Apple Silicon), and CPU.
"""

import argparse
import json
import os
import glob
import time
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class UTKFeatureExtractor:
    """ResNet18 ImageNet features (512-dim, pre-FC)."""

    def __init__(self, device='cpu'):
        self.device = device
        try:
            from torchvision.models import ResNet18_Weights
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            weight_tag = 'IMAGENET1K_V1'
        except Exception:
            backbone = models.resnet18(pretrained=True)
            weight_tag = 'pretrained=True (legacy)'
        self.weight_tag = weight_tag
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval()
        self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"Initialized ResNet18 ({weight_tag}) on {device}")

    @torch.no_grad()
    def extract_batch(self, image_paths, batch_size=64):
        all_features = []
        n = len(image_paths)
        for i in range(0, n, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    batch_tensors.append(self.transform(img))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    batch_tensors.append(torch.zeros(3, 224, 224))

            batch = torch.stack(batch_tensors).to(self.device)
            features = self.model(batch)
            # [B, 512, 1, 1] -> [B, 512]
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu().numpy().astype(np.float32))

            done = min(i + batch_size, n)
            if done % 1000 == 0 or done == n:
                print(f"  Processed {done}/{n} images...")

        return np.vstack(all_features)


def parse_utkface_filename(fname):
    """Parse UTKFace filename: {age}_{gender}_{race}_{date}.jpg[.chip.jpg]"""
    parts = os.path.basename(fname).split('_')
    if len(parts) >= 3:
        try:
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
            if gender not in (0, 1) or race not in (0, 1, 2, 3, 4):
                return None, None, None
            if age < 0 or age > 116:
                return None, None, None
            return age, gender, race
        except Exception:
            pass
    return None, None, None


def pick_device(requested):
    if requested == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return requested


def main():
    parser = argparse.ArgumentParser(description='Extract ResNet18 features from UTKFace')
    parser.add_argument('--data-dir', type=str, default='data/raw/utkface/UTKFace',
                        help='Directory containing UTKFace images')
    parser.add_argument('--output', type=str, default='data/raw/utkface_features.npz',
                        help='Output .npz file path')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for extraction')
    parser.add_argument('--max-images', type=int, default=None, help='Max images (None = all)')
    parser.add_argument('--device', type=str, default='auto',
                        help="Device: auto|cuda|mps|cpu")
    parser.add_argument('--source', type=str, default='',
                        help='Provenance note for metadata (e.g. kaggle:jangedoo/utkface-new)')
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")

    # Collect images (flat dir or nested)
    patterns = [
        os.path.join(args.data_dir, '*jpg.chip.jpg'),
        os.path.join(args.data_dir, '*.jpg'),
        os.path.join(args.data_dir, '**', '*jpg.chip.jpg'),
        os.path.join(args.data_dir, '**', '*.jpg'),
    ]
    image_files = []
    for pat in patterns:
        image_files.extend(glob.glob(pat, recursive=True))
    # unique, sorted for determinism
    image_files = sorted(set(image_files))

    if len(image_files) == 0:
        raise RuntimeError(f"No UTKFace images found in {args.data_dir}")

    print(f"Found {len(image_files)} image paths")

    if args.max_images:
        image_files = image_files[:args.max_images]
        print(f"Processing {len(image_files)} images (max-images limit)")

    ages, genders, races, valid_files = [], [], [], []
    for fpath in image_files:
        age, gender, race = parse_utkface_filename(fpath)
        if age is not None:
            ages.append(age)
            genders.append(gender)
            races.append(race)
            valid_files.append(fpath)

    print(f"Valid parseable images: {len(valid_files)}/{len(image_files)}")
    if len(valid_files) == 0:
        raise RuntimeError("No valid UTKFace filenames parsed")

    ages = np.array(ages, dtype=np.int64)
    genders = np.array(genders, dtype=np.float32)
    races = np.array(races, dtype=np.int64)

    extractor = UTKFeatureExtractor(device=device)
    start_time = time.time()
    features = extractor.extract_batch(valid_files, batch_size=args.batch_size)
    elapsed = time.time() - start_time
    print(f"Extraction took {elapsed:.1f}s ({len(valid_files) / max(elapsed, 1e-6):.1f} img/s)")

    # Provenance metadata (JSON string arrays for npz compatibility)
    meta = {
        'provenance': 'REAL_UTKFACE_IMAGES',
        'source': args.source or f'local:{os.path.abspath(args.data_dir)}',
        'backbone': 'resnet18',
        'weights': extractor.weight_tag,
        'feature_dim': int(features.shape[1]),
        'n_images': int(features.shape[0]),
        'device': device,
        'data_dir': os.path.abspath(args.data_dir),
        'synthetic': False,
        'note': 'X=ResNet18 pre-FC ImageNet features; gender 0=F 1=M; race 0-4',
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    np.savez_compressed(
        args.output,
        X=features,
        age=ages,
        gender=genders,
        race=races,
        meta_json=np.array(json.dumps(meta)),
    )

    print(f"Saved features: X={features.shape}")
    print(f"Gender: Female={int(np.sum(genders==0))}, Male={int(np.sum(genders==1))}")
    print(f"Race bincount: {np.bincount(races, minlength=5).tolist()}")
    print(f"Age range: {ages.min()}-{ages.max()}")
    print(f"Output: {args.output}")
    print(f"Meta: {meta}")


if __name__ == '__main__':
    main()
