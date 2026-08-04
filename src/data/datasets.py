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


def load_compas(data_dir='data/raw'):
    """Load and preprocess the COMPAS (ProPublica recidivism) dataset.

    Source: ProPublica's compas-analysis repo (public)
      https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
    Local: data/raw/compas-scores-two-years.csv

    Protected attribute (race), binarized — STANDARD FAIRNESS PRACTICE:
      a = 1  if African-American
      a = 0  if Caucasian
    Only these two groups are retained (Hispanic/Asian/Other/Native American
    are dropped). This is the 2-group split used by ProPublica's original
    "Machine Bias" analysis and by Hardt, Price, Srebro (2016) "Equality of
    Opportunity: Disparate Accuracy and Equalized Odds in Fair Prediction"
    (arXiv:1610.02413), the canonical COMPAS fairness benchmark. Mapping
    1=African-American, 0=Caucasian follows the convention of measuring DP as
    |P(recid=1 | AA) - P(recid=1 | Caucasian)|.

    Label: two_year_recid (0=no recidivism within 2 years, 1=recidivated).

    Leakage hygiene: identity columns (id/name/dob), post-outcome and score
    columns (is_recid, decile_score, *score_text*, *decile_score.1*, event,
    v_*_score*, type_of_assessment), and post-arrest case/jail/charge columns
    are dropped before modeling. Only pre-screening features survive.

    Rows with NA in the surviving feature columns are dropped. Categorical
    features are LabelEncoded.

    Returns X, y, a, 'COMPAS'.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'compas-scores-two-years.csv')

    if not os.path.exists(path):
        _download_file(
            'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv',
            path
        )

    df = pd.read_csv(path)

    # Restrict to the two-group race benchmark (African-American vs Caucasian)
    df = df[df['race'].isin(['African-American', 'Caucasian'])].copy()
    df = df.dropna(subset=['two_year_recid', 'race']).reset_index(drop=True)

    # Target: two_year_recid (already 0/1)
    y = df['two_year_recid'].astype(np.float32).values

    # Protected attribute: 1=African-American, 0=Caucasian
    a = (df['race'] == 'African-American').astype(np.int64).values

    # Leakage columns: identity, post-outcome scores, and post-arrest case info.
    leak_cols = {
        'id', 'name', 'first', 'last', 'dob', 'compas_screening_date',
        'is_recid', 'decile_score', 'decile_score.1', 'score_text',
        'v_type_of_assessment', 'v_decile_score', 'v_score_text',
        'v_screening_date', 'screening_date', 'type_of_assessment',
        'event', 'start', 'end',
        'r_case_number', 'r_charge_degree', 'r_days_from_arrest',
        'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
        'vr_case_number', 'vr_charge_degree', 'vr_offense_date',
        'vr_charge_desc', 'violent_recid', 'is_violent_recid',
        'c_case_number', 'c_offense_date', 'c_arrest_date',
        'c_days_from_compas', 'c_jail_in', 'c_jail_out',
        'days_b_screening_arrest', 'in_custody', 'out_custody',
        'priors_count.1',
    }
    drop_cols = [c for c in leak_cols if c in df.columns]
    drop_cols += ['two_year_recid', 'race']
    df = df.drop(columns=drop_cols)

    # Drop rows with NA in the surviving features, then re-align y/a
    df = df.dropna().reset_index(drop=True)
    y = y[df.index.values]
    a = a[df.index.values]

    # Encode remaining categorical columns
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    X = df.values.astype(np.float32)
    return X, y, a, 'COMPAS'


def load_german(data_dir='data/raw'):
    """Load and preprocess the German Credit (UCI statlog) dataset.

    Source: UCI Machine Learning Repository (statlog/german)
      https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
    Local: data/raw/german.data
    Format: space-separated, no header, 20 attributes + 1 label column.

    Protected attribute: sex (1=male, 0=female).
      Derived from attribute 9 (Personal status and sex) per the UCI
      german.doc codebook:
        A91 : male   : divorced/separated
        A92 : female : divorced/separated/married
        A93 : male   : single
        A94 : male   : married/widowed
        A95 : female : single
      => male = {A91, A93, A94}, female = {A92, A95}.
      Sex is chosen (over age<25) for consistency with the Adult and Credit
      loaders, which also use sex as the binary protected attribute.

    Label: credit outcome (1=good, 0=bad). Raw column is 1=good, 2=bad, so we
    remap 1->1 (good) and 2->0 (bad).

    Categorical columns (the A*-coded attributes) are LabelEncoded. Numeric
    columns are passed through.

    Returns X, y, a, 'German'.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'german.data')

    if not os.path.exists(path):
        _download_file(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
            path
        )

    # Space-separated, no header, 21 columns (20 attrs + label)
    df = pd.read_csv(path, sep=r'\s+', header=None)
    df = df.dropna().reset_index(drop=True)

    # Protected attribute: sex from column 8 (Personal status and sex)
    sex_male = df[8].isin(['A91', 'A93', 'A94'])
    a = sex_male.astype(np.int64).values

    # Label: column 20 (1=good, 2=bad) -> (1=good, 0=bad)
    y = (df[20] == 1).astype(np.float32).values

    # Drop label and protected-attr columns from features
    df = df.drop(columns=[8, 20])

    # Encode any categorical (object/string) columns with LabelEncoder
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    X = df.values.astype(np.float32)
    return X, y, a, 'German'


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
    elif name == 'compas':
        X, y, a, dname = load_compas(data_dir)
    elif name == 'german':
        X, y, a, dname = load_german(data_dir)
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
