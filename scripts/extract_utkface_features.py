#!/usr/bin/env python3
"""
Extract ResNet18 features from UTKFace images.

Run this on the GPU server (flair2.iitgn.ac.in) after copying UTKFace data.

Usage:
    python scripts/extract_utkface_features.py --data-dir /data/srujan.sai/UTKFace --output /data/srujan.sai/utkface_features.npz

Expected time: ~45 min for 200K images on L40S GPU.
"""

import argparse
import os
import glob
import time
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class UTKFeatureExtractor:
    """
    Extract features from UTKFace images using pretrained ResNet18.

    Uses the layer before the final classification layer (512-dim features).
    ImageNet pretrained weights are used as-is (no fine-tuning).
    """

    def __init__(self, device='cuda'):
        self.device = device

        # Load pretrained ResNet18 (remove final FC layer)
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(device)

        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Initialized ResNet18 on {device}")

    def extract_batch(self, image_paths, batch_size=64):
        """Extract features from a batch of images."""
        all_features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    batch_tensors.append(self.transform(img))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    # Use zeros for failed images
                    batch_tensors.append(torch.zeros(3, 224, 224))

            batch = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                features = self.model(batch).squeeze(-1)  # Shape: [batch, 512]

            all_features.append(features.cpu().numpy())

            if (i + batch_size) % 5000 == 0:
                print(f"  Processed {i + batch_size}/{len(image_paths)} images...")

        return np.vstack(all_features)


def parse_utkface_filename(fname):
    """Parse UTKFace filename: {age}_{gender}_{race}_{date}.jpg.chip.jpg"""
    parts = os.path.basename(fname).split('_')
    if len(parts) >= 3:
        try:
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
            return age, gender, race
        except:
            pass
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Extract ResNet18 features from UTKFace')
    parser.add_argument('--data-dir', type=str, default='/data/srujan.sai/UTKFace',
                        help='Directory containing UTKFace images')
    parser.add_argument('--output', type=str, default='/data/srujan.sai/utkface_features.npz',
                        help='Output .npz file path')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for extraction')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to process (None = all)')
    args = parser.parse_args()

    # Find all image files
    image_pattern = os.path.join(args.data_dir, '*', 'jpg.chip.jpg')
    image_files = glob.glob(image_pattern)

    # Also check if images are directly in data_dir
    direct_pattern = os.path.join(args.data_dir, '*jpg.chip.jpg')
    direct_files = glob.glob(direct_pattern)

    if direct_files:
        image_files = direct_files

    if len(image_files) == 0:
        raise RuntimeError(f"No UTKFace images found in {args.data_dir}")

    print(f"Found {len(image_files)} images")

    # Limit if requested
    if args.max_images:
        image_files = image_files[:args.max_images]
        print(f"Processing {len(image_files)} images (maxImages limit)")

    # Initialize extractor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = UTKFeatureExtractor(device=device)

    # Parse metadata
    print("Parsing filenames...")
    ages = []
    genders = []
    races = []
    valid_indices = []

    for i, fpath in enumerate(image_files):
        age, gender, race = parse_utkface_filename(fpath)
        if age is not None:
            ages.append(age)
            genders.append(gender)
            races.append(race)
            valid_indices.append(i)

    print(f"Valid images: {len(valid_indices)}/{len(image_files)}")

    # Filter to valid images
    image_files = [image_files[i] for i in valid_indices]
    ages = np.array(ages)
    genders = np.array(genders)
    races = np.array(races)

    # Extract features
    print("Extracting features...")
    start_time = time.time()

    features = extractor.extract_batch(image_files, batch_size=args.batch_size)

    elapsed = time.time() - start_time
    print(f"Extraction took {elapsed:.1f}s ({len(image_files) / elapsed:.1f} img/s)")

    # Save
    print(f"Saving to {args.output}...")
    np.savez_compressed(
        args.output,
        X=features,
        age=ages,
        gender=genders,
        race=races
    )

    print(f"Saved features: X={features.shape}, age={ages.shape}, gender={genders.shape}, race={races.shape}")

    # Print summary
    print("\n=== UTKFace Feature Extraction Summary ===")
    print(f"Total images: {len(image_files)}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Gender distribution: Male={np.sum(genders==1)}, Female={np.sum(genders==0)}")
    print(f"Race distribution: {np.bincount(races)}")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()