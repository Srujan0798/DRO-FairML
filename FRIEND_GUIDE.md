# Guide for Friend - Running DRO-FAIR on FLAIR2 Server

## Quick Setup (One Time)

### 1. Login to JupyterHub
- URL: http://flair2.iitgn.ac.in:8000/hub/login
- Username: `srujan.sai`
- Password: `ss#081`

### 2. Open Terminal
- In JupyterHub: `New → Terminal`

### 3. Clone/Download the Repo
```bash
cd /data/srujan.sai

# If DRO-FairML doesn't exist, clone it:
git clone https://github.com/YOUR_GITHUB_USERNAME/DRO-FairML.git
# OR download as zip from GitHub

# If it exists, pull latest:
cd DRO-FairML
git pull origin main
```

### 4. Install Python Packages
```bash
pip install numpy torch scikit-learn pandas scikit-image
```

### 5. Verify GPU Access
```bash
nvidia-smi
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Daily Workflow

### When Srujan updates code on GitHub:
```bash
cd /data/srujan.sai/DRO-FairML
git pull origin main
```

### To run experiments:
```bash
cd /data/srujan.sai/DRO-FairML

# Set PYTHONPATH
export PYTHONPATH=/data/srujan.sai/DRO-FairML

# Run fairness PGD test on tabular datasets (fast, no GPU needed):
python3 scripts/test_fairness_pgd.py

# Run UTKFace feature extraction (GPU required):
python3 scripts/extract_utkface_features.py \
    --data-dir /data/srujan.sai/UTKFace \
    --output /data/srujan.sai/utkface_features.npz

# Run main training with DRO-FAIR:
python3 experiments/run_fairness_pgd.py --dataset adult --epochs 50
```

---

## Current Code Status

### Ready to Run (No GPU):
- `scripts/test_fairness_pgd.py` - Tests FairnessTargetedPGD on Adult/Credit/LSAC
  - Compares gradient attack vs random vs heuristic
  - Takes ~5-10 minutes on CPU

### GPU Required:
- `scripts/extract_utkface_features.py` - Extract ResNet18 features from UTKFace images
  - Takes ~45 minutes for 200K images
  - After this, training is fast

---

## File Locations on Server

| Path | Description |
|------|-------------|
| `/data/srujan.sai/DRO-FairML/` | Main code directory |
| `/data/srujan.sai/DRO-FairML/scripts/` | Executable scripts |
| `/data/srujan.sai/DRO-FairML/src/` | Source code |
| `/data/srujan.sai/DRO-FairML/data/raw/` | Tabular datasets (Adult, Credit, LSAC) |
| `/data/srujan.sai/UTKFace/` | UTKFace images (when downloaded) |
| `/data/srujan.sai/utkface_features.npz` | Extracted CNN features |

---

## Key Classes/Functions

### FairnessTargetedPGD (src/corruption/adversarial.py)
Gradient-based attack that increases DP violation by flipping labels strategically.
- `alpha`: fraction of samples to corrupt (0.2 = 20%)
- `target_metric`: 'dp' (Demographic Parity) or 'if' (Individual Fairness)
- `coordinated`: if True, targets minority group more aggressively

### load_utkface (src/data/datasets.py)
Loads UTKFace dataset with ResNet18 features. Requires feature cache file.

---

## If Something Goes Wrong

### Package errors:
```bash
pip install --upgrade pip
pip install numpy torch scikit-learn pandas
```

### GPU not found:
```bash
nvidia-smi  # Check if GPU is visible
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Git issues:
```bash
cd /data/srujan.sai/DRO-FairML
git status  # See what changed
git stash   # Save local changes if needed
git pull    # Get latest from GitHub
```

---

## Srujan's Notes for Friend

Contact Srujan via WhatsApp/Discord when:
- Experiments finish (send output)
- Errors occur (paste error message)
- Questions about what to run next

Check GitHub for new commits: https://github.com/YOUR_GITHUB_USERNAME/DRO-FairML/commits/main