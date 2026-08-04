"""Download and prepare FairFace dataset for fairness experiments.

FairFace: 108,501 images across 7 race groups.
Label: gender (Male/Female)
Protected: race (White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern)
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('/data/srujan.sai/FairFace')

def prepare_fairface_labels():
    """FairFace provides train.csv and val.csv with labels."""
    train_csv = DATA_DIR / 'fairface_label_train.csv'
    val_csv = DATA_DIR / 'fairface_label_val.csv'
    
    if not train_csv.exists():
        print(f"FairFace not found at {DATA_DIR}")
        print("Download from: https://github.com/joojs/fairface")
        return None
    
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df = pd.concat([df_train, df_val], ignore_index=True)
    
    # Gender as label: Male=1, Female=0
    label = (df['gender'] == 'Male').astype(np.float32).values
    
    # Race as protected attribute (binarize: White=0, Non-White=1 for simplicity)
    race = df['race'].values
    protected = (race != 'White').astype(np.int64).values
    
    print(f"FairFace prepared: {len(df)} samples")
    print(f"  Gender (Male): {label.mean():.3f}")
    print(f"  Race (Non-White): {protected.mean():.3f}")
    
    np.savez(DATA_DIR / 'fairface_labels.npz',
             filenames=df['file'].values.astype(str),
             label=label, race=protected)
    return df

if __name__ == '__main__':
    prepare_fairface_labels()
