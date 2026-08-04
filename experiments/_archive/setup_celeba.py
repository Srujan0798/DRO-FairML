"""Download and prepare CelebA dataset for fairness experiments.

CelebA Attributes:
- label: Attractive (or Smiling)
- protected: Male (gender)

We use the aligned & cropped images.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('/data/srujan.sai/CelebA')  # server path

def download_celeba():
    """Download CelebA dataset using torchvision."""
    import torchvision.datasets as datasets
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download with attributes
    dataset = datasets.CelebA(
        root=str(DATA_DIR),
        split='all',
        target_type='attr',
        download=True
    )
    print(f"CelebA downloaded: {len(dataset)} images")
    return dataset

def prepare_celeba_labels():
    """Extract label and protected attribute from CelebA attributes CSV."""
    attr_path = DATA_DIR / 'celeba' / 'list_attr_celeba.csv'
    if not attr_path.exists():
        print(f"Attributes not found at {attr_path}")
        print("Need to download CelebA first.")
        return None
    
    df = pd.read_csv(attr_path)
    
    # Use 'Attractive' as label (20th attribute, index 2)
    # Use 'Male' as protected attribute (index 20)
    # CelebA uses -1/1 encoding
    
    labels = {
        'filenames': df['image_id'].values,
        'label': ((df['Attractive'].values + 1) // 2).astype(np.float32),  # -1->0, 1->1
        'gender': ((df['Male'].values + 1) // 2).astype(np.int64),         # -1->0 (Female), 1->1 (Male)
    }
    
    print(f"CelebA labels prepared: {len(labels['filenames'])} samples")
    print(f"  Label (Attractive): mean={labels['label'].mean():.3f}")
    print(f"  Gender (Male): mean={labels['gender'].mean():.3f}")
    
    np.savez(DATA_DIR / 'celeba_labels.npz', **labels)
    return labels

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--download':
        download_celeba()
    prepare_celeba_labels()
