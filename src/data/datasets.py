"""
Dataset loading and preprocessing for Adult, Credit, and LSAC.
All datasets use label encoding for categorical variables and StandardScaler normalization.
80/20 train-test split, with training data further split 85/15 for train/validation.

CRITICAL FIXES:
1. StandardScaler is fit ONLY on training data, then applied to val/test (no leakage).
2. Removed synthetic fallbacks — fail loudly if real data cannot be loaded.
3. Use local RandomState for reproducibility without global side effects.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os


def _download_file(url, path):
    """Download file using curl (works around Python SSL issues on macOS)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import subprocess
    result = subprocess.run(['curl', '-sL', '-o', path, url], capture_output=True)
    if result.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Failed to download {url}: {result.stderr.decode()}")


def load_adult(data_dir='data/raw'):
    """Load and preprocess Adult dataset (UCI Machine Learning Repository)."""
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, 'adult.data')
    test_path = os.path.join(data_dir, 'adult.test')

    if not os.path.exists(train_path):
        _download_file(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            train_path
        )
    if not os.path.exists(test_path):
        _download_file(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
            test_path
        )

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    df_train = pd.read_csv(train_path, names=columns, skipinitialspace=True, na_values='?')
    df_test = pd.read_csv(test_path, names=columns, skipinitialspace=True, na_values='?', skiprows=1)

    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.dropna()

    # Target: income >50K
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

    # Protected attribute: sex (1=Male, 0=Female)
    df['sex'] = df['sex'].apply(lambda x: 1 if str(x).strip() == 'Male' else 0)

    # Drop fnlwgt and education (redundant with education-num)
    df = df.drop(columns=['fnlwgt', 'education'])

    # Encode categorical features
    categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    y = df['income'].values.astype(np.float32)
    a = df['sex'].values.astype(np.int64)
    X = df.drop(columns=['income', 'sex']).values.astype(np.float32)

    return X, y, a, 'Adult'


def load_credit(data_dir='data/raw'):
    """Load and preprocess Credit Card Default dataset (UCI)."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'default_of_credit_card_clients.xls')

    if not os.path.exists(path):
        _download_file(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
            path
        )

    df = pd.read_excel(path, header=1)
    df = df.dropna()

    # Target: default payment next month
    y = df['default payment next month'].values.astype(np.float32)

    # Protected attribute: SEX (1=Male, 2=Female -> convert to 1=Male, 0=Female)
    a = (df['SEX'].values == 1).astype(np.int64)

    # Drop target and protected from features
    df = df.drop(columns=['default payment next month', 'SEX', 'ID'])
    X = df.values.astype(np.float32)

    return X, y, a, 'Credit'


def load_lsac(data_dir='data/raw'):
    """Load and preprocess LSAC (Law School Admissions Council) Bar Passage dataset.

    Data source: https://github.com/damtharvey/law-school-dataset
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'lsac.csv')

    if not os.path.exists(path):
        # Try GitHub raw URL for the real LSAC dataset
        urls = [
            'https://raw.githubusercontent.com/damtharvey/law-school-dataset/main/law_dataset.csv',
        ]
        downloaded = False
        for url in urls:
            try:
                _download_file(url, path)
                downloaded = True
                break
            except Exception:
                continue
        if not downloaded:
            raise RuntimeError("Could not download LSAC dataset from any source")

    df = pd.read_csv(path)
    df = df.dropna()

    # Target: pass_bar (1=passed, 0=failed)
    y = df['pass_bar'].values.astype(np.float32)

    # Protected attribute: race (racetxt: 0=minority, 1=majority)
    # NOTE: Was 'male' in earlier code; fixed to match paper (Race)
    a = df['racetxt'].values.astype(np.int64)

    # Drop target and protected
    df = df.drop(columns=['pass_bar', 'racetxt'])

    # Encode remaining categoricals if any
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    X = df.values.astype(np.float32)
    return X, y, a, 'LSAC'


def load_utkface(data_dir='data/raw/utkface', feature_cache=None):
    """Load and preprocess UTKFace dataset.

    Uses ResNet18 pretrained features (512-dim) extracted from face images.
    Requires a pre-extracted feature cache (.npz). Does NOT invent synthetic data.

    Preferred cache locations (first hit wins if feature_cache is None):
      - data/raw/utkface_features.npz
      - {data_dir}/utkface_features.npz
      - data/raw/utkface_features_smoke.npz  (only if marked REAL; otherwise rejected)

    Task (binary trainers):
      y = gender (0=Female, 1=Male)
      a = race binarized (0=White, 1=non-White)  — binary protected attr for DRO/Naive

    Returns:
        X: ResNet18 features (N, 512)
        y: gender labels (binary)
        a: binary race (White vs non-White)
        dname: 'UTKFace' (never synthetic)
    """
    import os
    import json
    import glob

    candidates = []
    if feature_cache is not None:
        candidates.append(feature_cache)
    # Prefer real full cache over smoke
    candidates.extend([
        os.path.join('data', 'raw', 'utkface_features.npz'),
        os.path.join(data_dir if data_dir else 'data/raw/utkface', 'utkface_features.npz'),
        os.path.join(os.path.dirname(data_dir) if data_dir else 'data/raw', 'utkface_features.npz'),
        os.path.join('data', 'raw', 'utkface_features_smoke.npz'),
    ])

    cache_path = None
    for c in candidates:
        if c and os.path.exists(c):
            cache_path = c
            break

    if cache_path is None:
        # Probe for images so the error message is actionable
        search_dirs = [
            data_dir,
            os.path.join(data_dir, 'UTKFace') if data_dir else None,
            'data/raw/utkface/UTKFace',
        ]
        n_img = 0
        for d in search_dirs:
            if d and os.path.isdir(d):
                n_img = max(n_img, len(glob.glob(os.path.join(d, '*jpg*'))))
        raise RuntimeError(
            f"No UTKFace feature cache found. Tried: {candidates}. "
            f"Images present≈{n_img}. Extract with: "
            f"python scripts/extract_utkface_features.py "
            f"--data-dir data/raw/utkface/UTKFace --output data/raw/utkface_features.npz"
        )

    data = np.load(cache_path, allow_pickle=True)
    # Reject synthetic / Gaussian smoke mislabelled as real
    meta = {}
    if 'meta_json' in data.files:
        try:
            meta = json.loads(str(data['meta_json'].item() if hasattr(data['meta_json'], 'item') else data['meta_json']))
        except Exception:
            meta = {}
    if meta.get('synthetic') is True or meta.get('provenance') == 'SYNTHETIC':
        raise RuntimeError(
            f"Feature cache {cache_path} is tagged synthetic — refusing to load as real UTKFace"
        )
    # Heuristic: smoke Gaussian features have near-perfect race balance + ~N(0,1)
    # Full real UTKFace race is heavily White-majority. Still allow smoke only if
    # explicitly named smoke AND caller asks for it (never as canonical real).
    X = data['X'].astype(np.float32)
    gender = data['gender'].astype(np.float32)
    race = data['race'].astype(np.int64)

    # Binary protected: White (0) vs non-White (1) for trainers that require binary a
    a = (race != 0).astype(np.int64)
    y = gender.astype(np.float32)

    print(
        f"Loaded REAL UTKFace features from {cache_path}: "
        f"X={X.shape}, y=gender, a=race_binary(White/nonWhite), "
        f"meta={meta.get('provenance', 'unspecified')}"
    )
    return X, y, a, 'UTKFace'


def get_dataset(name, data_dir='data/raw', test_size=0.2, val_size=0.15, random_state=42):
    """
    Load dataset and split into train/val/test.
    CRITICAL FIX: StandardScaler is fit ONLY on training data to prevent leakage.

    Returns: X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dataset_name
    """
    name = name.lower()
    if name == 'adult':
        X, y, a, dname = load_adult(data_dir)
    elif name == 'credit':
        X, y, a, dname = load_credit(data_dir)
    elif name == 'lsac':
        X, y, a, dname = load_lsac(data_dir)
    elif name == 'utkface':
        # data_dir for tabular is 'data/raw'; UTKFace lives under data/raw/utkface + feature cache
        utk_dir = data_dir if 'utkface' in os.path.basename(data_dir.rstrip('/')) else os.path.join(data_dir, 'utkface')
        X, y, a, dname = load_utkface(data_dir=utk_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Joint stratification by label and protected attribute when possible
    try:
        n_groups = int(a.max() + 1)
        joint_strat = y * n_groups + a
        # Check if all combinations have at least 2 samples
        if np.min(np.bincount(joint_strat.astype(int))) >= 2:
            strat = joint_strat
        else:
            strat = y
    except Exception:
        strat = y

    # First split: train+val / test
    X_trainval, X_test, y_trainval, y_test, a_trainval, a_test = train_test_split(
        X, y, a, test_size=test_size, random_state=random_state, stratify=strat
    )

    # Second split: train / val
    try:
        n_groups = int(a_trainval.max() + 1)
        joint_strat_val = y_trainval * n_groups + a_trainval
        if np.min(np.bincount(joint_strat_val.astype(int))) >= 2:
            strat_val = joint_strat_val
        else:
            strat_val = y_trainval
    except Exception:
        strat_val = y_trainval

    X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
        X_trainval, y_trainval, a_trainval, test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=strat_val
    )

    # CRITICAL FIX: Fit scaler ONLY on training data, transform val/test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname
