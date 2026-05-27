# Guide for Friend - Running DRO-FAIR on FLAIR2 Server

## IMPORTANT REPO INFO
- **GitHub URL**: https://github.com/Srujan0798/DRO-FairML.git
- **Repo Owner**: Srujan0798 (Srujan Sai)
- **Main Branch**: main

---

## Quick Setup (One Time)

### 1. Login to JupyterHub
- URL: http://flair2.iitgn.ac.in:8000/hub/login
- Username: `srujan.sai`
- Password: `ss#081`

### 2. Open Terminal
- In JupyterHub: `New → Terminal`

### 3. Clone the Repo (First Time Only)
```bash
cd /data/srujan.sai

# Clone the repo:
git clone https://github.com/Srujan0798/DRO-FairML.git

# Navigate into it:
cd DRO-FairML
```

### 4. Update Repo (When Srujan Makes Changes)
```bash
cd /data/srujan.sai/DRO-FairML
git pull origin main
```

### 5. Install Python Packages
```bash
pip install numpy torch scikit-learn pandas scikit-image
```

### 6. Verify GPU Access
```bash
nvidia-smi
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Daily Workflow

### When Srujan pushes new code to GitHub:
```bash
cd /data/srujan.sai/DRO-FairML
git pull origin main
```

### To run experiments:
```bash
cd /data/srujan.sai/DRO-FairML
export PYTHONPATH=/data/srujan.sai/DRO-FairML
```

#### Option 1: Test Fairness Attack (FAST - No GPU needed)
```bash
python3 scripts/test_fairness_pgd.py
```
- Tests gradient-based fairness attack on Adult/Credit/LSAC datasets
- Takes ~5-10 minutes on CPU
- Compares: Clean vs Random vs Heuristic vs Grad-PGD attacks

#### Option 2: Extract UTKFace Features (GPU Required - 45 min)
```bash
# First download UTKFace images, then extract features
python3 scripts/extract_utkface_features.py \
    --data-dir /data/srujan.sai/UTKFace \
    --output /data/srujan.sai/utkface_features.npz
```

#### Option 3: Run DRO-FAIR Training
```bash
python3 experiments/run_fairness_pgd.py --dataset adult --epochs 50
```

---

## Current Code Status

### Ready to Run NOW (No GPU):
| Script | What it does | Time |
|--------|--------------|------|
| `scripts/test_fairness_pgd.py` | Tests FairnessTargetedPGD on Adult/Credit/LSAC | ~5 min |

### GPU Required (Run later):
| Script | What it does | Time |
|--------|--------------|------|
| `scripts/extract_utkface_features.py` | Extract ResNet18 features from UTKFace images | ~45 min |
| `experiments/run_fairness_pgd.py` | Main DRO-FAIR training | varies |

---

## File Locations on Server

| Path | Description |
|------|-------------|
| `/data/srujan.sai/DRO-FairML/` | Main code directory |
| `/data/srujan.sai/DRO-FairML/scripts/` | Executable scripts |
| `/data/srujan.sai/DRO-FairML/src/` | Source code (adversarial.py, datasets.py, classifier.py) |
| `/data/srujan.sai/DRO-FairML/data/raw/` | Tabular datasets (Adult, Credit, LSAC) |
| `/data/srujan.sai/UTKFace/` | UTKFace images (download separately) |
| `/data/srujan.sai/utkface_features.npz` | Extracted CNN features (after running extract script) |

---

## Key Classes (What's in the code)

### FairnessTargetedPGD (src/corruption/adversarial.py)
Gradient-based attack that increases Demographic Parity (DP) violation.
- `alpha`: fraction of samples to corrupt (0.2 = 20%)
- `target_metric`: 'dp' (Demographic Parity) or 'if' (Individual Fairness)
- `coordinated`: if True, targets minority group more aggressively

### load_utkface (src/data/datasets.py)
Loads UTKFace dataset. Needs feature cache file (`utkface_features.npz`).

### MLPClassifier (src/models/classifier.py)
Simple MLP neural network for binary classification.

---

## If Something Goes Wrong

### Package errors:
```bash
pip install --upgrade pip
pip install numpy torch scikit-learn pandas
```

### GPU not found:
```bash
nvidia-smi  # Should show GPU info
python3 -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### Git issues:
```bash
cd /data/srujan.sai/DRO-FairML
git status           # See what changed locally
git pull origin main  # Get latest code
```

---

## Communicate with Srujan

WhatsApp/Discord Srujan when:
- Experiments finish → send output/screenshot
- Errors occur → paste error message
- Questions about what to run next

Check GitHub for updates: https://github.com/Srujan0798/DRO-FairML/commits/main

---

## Srujan's Work (What We've Built)

1. **FairnessTargetedPGD** - Gradient-based attack that strategically flips labels to MAXIMIZE unfairness (DP violation). Verified working on local Mac.

2. **UTKFace Pipeline** - ResNet18 feature extraction + DRO-FAIR training on face images.

3. **Tabular Experiments** - Adult, Credit, LSAC datasets tested with all attack types.

---

## Quick Command Reference

```bash
# Setup (one time)
git clone https://github.com/Srujan0798/DRO-FairML.git
cd DRO-FairML
pip install numpy torch scikit-learn pandas

# Daily: Get latest code
cd /data/srujan.sai/DRO-FairML
git pull origin main

# Run fairness test
export PYTHONPATH=/data/srujan.sai/DRO-FairML
python3 scripts/test_fairness_pgd.py

# Check GPU
nvidia-smi
```