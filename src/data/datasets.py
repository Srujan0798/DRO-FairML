"""
Dataset loading and preprocessing for Adult, Credit, and LSAC.
All datasets use label encoding for categorical variables and StandardScaler normalization.
80/20 train-test split, with training data further split 85/15 for train/validation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import warnings


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

    try:
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

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y, a, 'Adult'

    except Exception as e:
        warnings.warn(f"Failed to load Adult dataset: {e}. Using synthetic data.")
        return _generate_synthetic_adult()


def load_credit(data_dir='data/raw'):
    """Load and preprocess Credit Card Default dataset (UCI)."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'default_of_credit_card_clients.xls')

    try:
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

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y, a, 'Credit'

    except Exception as e:
        warnings.warn(f"Failed to load Credit dataset: {e}. Using synthetic data.")
        return _generate_synthetic_credit()


def load_lsac(data_dir='data/raw'):
    """Load and preprocess LSAC (Law School Admissions Council) Bar Passage dataset.

    Data source: https://github.com/damtharvey/law-school-dataset
    Columns: decile1b, decile3, lsat, ugpa, zfygpa, zgpa, fulltime, fam_inc,
             male, racetxt, tier, pass_bar
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'lsac.csv')

    try:
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
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y, a, 'LSAC'

    except Exception as e:
        warnings.warn(f"Failed to load LSAC dataset: {e}. Using synthetic data.")
        return _generate_synthetic_lsac()


def _generate_synthetic_adult(n_samples=45222, n_features=12, random_state=42):
    """Generate synthetic Adult-like data with realistic group disparities."""
    rng = np.random.RandomState(random_state)
    a = (rng.rand(n_samples) > 0.67).astype(np.int64)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    for i in range(n_features):
        X[a == 0, i] = rng.normal(loc=0.0, scale=1.0, size=(a == 0).sum())
        X[a == 1, i] = rng.normal(loc=0.3, scale=1.0, size=(a == 1).sum())
    logit = (0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.4 * X[:, 2] +
             0.2 * a + 0.15 * rng.randn(n_samples))
    y = (logit > 0.3).astype(np.float32)
    return X, y, a, 'Adult (Synthetic)'


def _generate_synthetic_credit(n_samples=30000, n_features=22, random_state=42):
    """Generate synthetic Credit-like data with realistic group disparities."""
    rng = np.random.RandomState(random_state)
    a = (rng.rand(n_samples) > 0.4).astype(np.int64)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    for i in range(n_features):
        X[a == 0, i] = rng.normal(loc=0.0, scale=1.0, size=(a == 0).sum())
        X[a == 1, i] = rng.normal(loc=0.2, scale=1.0, size=(a == 1).sum())
    logit = (0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.3 * X[:, 2] +
             0.15 * a + 0.1 * rng.randn(n_samples))
    y = (logit > 0.4).astype(np.float32)
    return X, y, a, 'Credit (Synthetic)'


def _generate_synthetic_lsac(n_samples=18692, n_features=10, random_state=42):
    """Generate synthetic LSAC-like data with realistic group disparities."""
    rng = np.random.RandomState(random_state)
    a = (rng.rand(n_samples) > 0.45).astype(np.int64)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    for i in range(n_features):
        X[a == 0, i] = rng.normal(loc=0.0, scale=1.0, size=(a == 0).sum())
        X[a == 1, i] = rng.normal(loc=0.25, scale=1.0, size=(a == 1).sum())
    logit = (0.4 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] +
             0.18 * a + 0.12 * rng.randn(n_samples))
    y = (logit > 0.35).astype(np.float32)
    return X, y, a, 'LSAC (Synthetic)'


def get_dataset(name, data_dir='data/raw', test_size=0.2, val_size=0.15, random_state=42):
    """
    Load dataset and split into train/val/test.
    Returns: X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dataset_name
    """
    name = name.lower()
    if name == 'adult':
        X, y, a, dname = load_adult(data_dir)
    elif name == 'credit':
        X, y, a, dname = load_credit(data_dir)
    elif name == 'lsac':
        X, y, a, dname = load_lsac(data_dir)
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

    return X_train, y_train, a_train, X_val, y_val, a_val, X_test, y_test, a_test, dname
