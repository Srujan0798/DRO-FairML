# UTKFace RESULTS — Week 2

## Setup

**Dataset:** UTKFace (200K face images, age/gender/race annotated)

**Features:** ResNet18 pretrained features (512-dim) extracted from face images

**Protected attribute:** Race (5-class: White, Black, Asian, Indian, Others)
**Target:** Gender (binary: Male/Female)

**Pipeline:**
1. Extract ResNet18 features from face images → cache as `.npz`
2. Train MLP on features with Naive-FAIR and DRO-FAIR
3. Evaluate fairness (DP, IF) on test set

**Current status:** UTKFace images not available locally. Pipeline tested with synthetic data fallback.

---

## Pipeline Components

| Component | File | Status |
|-----------|------|--------|
| Data loader | `src/data/datasets.py::load_utkface()` | Working (synthetic fallback) |
| Feature extractor | `scripts/extract_utkface_features.py` | Needs UTKFace images |
| CNN classifier | `src/models/cnn_classifier.py` | Committed (ResNet18 backbone) |
| Image PGD | `src/corruption/image_pgd.py` | Committed (epsilon=4/255) |
| Experiment driver | `experiments/run_utkface.py` | Committed (synthetic fallback works) |

---

## Smoke Test Results (Synthetic Data)

```
UTKFace not available (GPU server blocked) — using synthetic data
DRO clean: acc=0.520 dp=0.053 if=0.000
Naive clean: acc=0.510 dp=0.077 if=0.000
```

DRO-FAIR shows lower DP violation (0.053 vs 0.077) on synthetic data with 512-dim Gaussian features.

---

## GPU Server Status

**Hostname:** `gpu-server` — not resolvable from local machine
**Action:** Need to contact sysadmin for correct hostname or IP

**Workaround:** Running CPU-only synthetic experiments. Full UTKFace requires:
1. Download UTKFace to `/data/srujan.sai/UTKFace/`
2. Extract ResNet18 features → `utkface_features.npz`
3. Run full experiment: 4 alphas × 3 seeds × 2 methods

---

## Limitations

1. **No real UTKFace results yet:** Pipeline works but no images available
2. **GPU server unresolved:** Cannot run CNN on 200K images without GPU
3. **Synthetic data is not conclusive:** 512-dim Gaussian noise doesn't reflect real face image structure

---

## Next Steps

1. **Resolve GPU access** — get server hostname/IP from sysadmin
2. **Download UTKFace** — ~2GB for full dataset
3. **Extract features** — ~30 min on GPU, run once, cache forever
4. **Run full ablation** — 4 alphas × 3 seeds × 2 methods on real features
5. **Generate fig10** — DP/IF curves vs α for UTKFace