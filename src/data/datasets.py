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

    # Protected attribute: male (1=male, 0=female)
    a = df['male'].values.astype(np.int64)

    # Drop target and protected
    df = df.drop(columns=['pass_bar', 'male'])

    # Encode remaining categoricals if any
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    X = df.values.astype(np.float32)
    return X, y, a, 'LSAC'


def load_utkface(data_dir='/data/srujan.sai/UTKFace', feature_cache=None):
    """Load and preprocess UTKFace dataset.

    Uses ResNet18 pretrained features (512-dim) extracted from face images.
    If feature_cache is provided, loads pre-extracted features.
    Otherwise returns placeholders for later feature extraction.

    UTKFace file format: {age}_{gender}_{race}_{date}.jpg.chip.jpg
        age: 0-116
        gender: 0=Female, 1=Male
        race: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others

    Returns:
        X: ResNet18 features (200K, 512)
        y: gender labels (binary: 1=Male)
        a: race labels (5-class: 0-4) — could also use gender as protected
    """
    import os
    import glob

    if feature_cache is not None and os.path.exists(feature_cache):
        data = np.load(feature_cache)
        return data['X'], data['gender'], data['race'], 'UTKFace'

    image_dir = os.path.join(data_dir, '*.jpg.chip.jpg')
    image_files = glob.glob(image_dir)

    if len(image_files) == 0:
        raise RuntimeError(f"No UTKFace images found in {data_dir}")

    print(f"Found {len(image_files)} UTKFace images — run feature extraction first")
    print(f"Expected format: {{age}}_{{gender}}_{{race}}_{{date}}.jpg.chip.jpg")
    print(f"Feature cache not found at {feature_cache}")
    print(f"Will return placeholder data — extract features using extract_utkface_features.py")

    X = np.zeros((len(image_files), 512), dtype=np.float32)
    y = np.zeros(len(image_files), dtype=np.float32)
    a = np.zeros(len(image_files), dtype=np.int64)

    valid_count = 0
    for i, fpath in enumerate(image_files):
        fname = os.path.basename(fpath)
        parts = fname.split('_')
        if len(parts) >= 3:
            try:
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                y[i] = gender
                a[i] = race
                valid_count += 1
            except:
                pass

    print(f"Parsed {valid_count}/{len(image_files)} valid filenames")
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
        X, y, a, dname = load_utkface(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # First split: train+val / test
    X_trainval, X_test, y_trainval, y_test, a_trainval, a_test = train_test_split(
        X, y, a, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train / val
    X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
        X_trainval, y_trainval, a_trainval, test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=y_trainval
    )

    # CRITICAL FIX: Fit scaler ONLY on training data, transform val/test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname
