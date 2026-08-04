# UTKFace Pipeline Design

## Why UTKFace?

We've done tabular (Adult, Credit, LSAC) — image data is the natural extension:
- **200K+ face images** with age, gender, race labels
- **Multi-class protected attributes**: gender (binary), race (5-class), age (0-116)
- **GPU-ready**: CNN features + DRO-FAIR training
- **Research gap**: Fairness on image data is less explored

## Dataset Structure

### UTKFace Files
Images named: `{age}_{gender}_{race}_{date}.jpg.chip.jpg`

| Field | Range | Example |
|-------|-------|---------|
| `age` | 0-116 | `25` |
| `gender` | 0=Female, 1=Male | `0` |
| `race` | 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others | `2` |
| `date` | YYYYMMDD | `201701161845` |

**Full filename**: `25_0_2_201701161845.jpg.chip.jpg` = 25-year-old Asian female

### Download
```bash
# Option 1: UTKFace official (Kaggle)
# https://www.kaggle.com/datasets/jangedoo/utkface-new

# Option 2: Direct download
wget https://github.com/m垂o/simple-unet/releases/download/v1.0/UTKFace.tar.gz

# Store in:
/data/srujan.sai/UTKFace/
```

---

## Pipeline Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Images    │ ──► │  Resize      │ ──► │   CNN       │ ──► │   DRO-FAIR   │
│  (200K)    │     │  224x224     │     │  (ResNet18)│     │  Training   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                            │
                                        Pretrained
                                        (ImageNet)
```

### Step 1: Feature Extraction (GPU)

```python
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

class UTKFeatureExtractor:
    """
    Extract features from UTKFace images using pretrained CNN.
    
    Uses ResNet18 (lighter than ResNet50, faster for 200K images).
    Features from the last convolutional layer → flattened.
    """
    
    def __init__(self, device='cuda'):
        # Load pretrained ResNet18 (remove final classifier)
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove FC
        self.model.eval()
        self.device = device
        self.model.to(device)
        
        # ImageNet normalization (applied to all images)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract(self, image_path):
        """Extract features from single image."""
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_t)  # Shape: [1, 512]
        
        return features.squeeze().cpu().numpy()
    
    def extract_batch(self, image_paths, batch_size=64):
        """Extract features from batch of images (GPU accelerated)."""
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                batch_tensors.append(self.transform(img))
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                features = self.model(batch).squeeze(-1)  # Shape: [batch, 512]
            
            all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)
```

### Step 2: UTKFace Dataset Loader

```python
import os
import numpy as np
from PIL import Image
import pandas as pd

class UTKFaceDataset:
    """
    UTKFace dataset loader for fairness research.
    
    Protected attributes available:
    - gender: 0=Female, 1=Male (binary)
    - race: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others (multi-class)
    - age: 0-116 (continuous → bin into categories)
    
    For binary fairness: use gender
    For multi-class fairness: use race
    """
    
    def __init__(self, data_dir='/data/srujan.sai/UTKFace'):
        self.data_dir = data_dir
        
        # Find all image files
        self.image_files = [
            f for f in os.listdir(data_dir) 
            if f.endswith('.jpg.chip.jpg')
        ]
        
        # Parse filenames into structured data
        self.data = self._parse_filenames()
    
    def _parse_filenames(self):
        """
        Parse: {age}_{gender}_{race}_{date}.jpg.chip.jpg
        Returns: DataFrame with parsed columns
        """
        records = []
        
        for fname in self.image_files:
            try:
                parts = fname.split('_')
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                
                records.append({
                    'filename': fname,
                    'age': age,
                    'gender': gender,  # 0=F, 1=M
                    'race': race,      # 0=W, 1=B, 2=A, 3=I, 4=O
                    'path': os.path.join(self.data_dir, fname)
                })
            except:
                continue
        
        df = pd.DataFrame(records)
        return df
    
    def get_binary_fairness_data(self, protected_attr='gender'):
        """
        Get data formatted for DRO-FAIR training.
        
        Args:
            protected_attr: 'gender' (binary) or 'race' (multi-class)
        
        Returns:
            X: image features (after CNN extraction)
            y: labels (binary: income/employment/etc — we'll use gender prediction)
            a: protected attributes (binary or multi-class)
            
        Note: For UTKFace, we typically predict age or gender.
        The "label" y depends on the fairness task.
        """
        pass
    
    def to_pytorch_dataset(self, X, y, a):
        """
        Convert to PyTorch dataset for training.
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        a_tensor = torch.tensor(a, dtype=torch.float32)
        
        dataset = TensorDataset(X_tensor, y_tensor, a_tensor)
        return DataLoader(dataset, batch_size=64, shuffle=True)
```

### Step 3: ResNet18 Feature Extraction Pipeline

```python
def extract_utkface_features(
    data_dir='/data/srujan.sai/UTKFace',
    output_path='/data/srujan.sai/utkface_features.npz',
    batch_size=64
):
    """
    Extract CNN features from all UTKFace images.
    
    This takes ~30-60 min for 200K images on L40S GPU.
    We do it ONCE and cache the features.
    """
    
    extractor = UTKFeatureExtractor(device='cuda')
    dataset = UTKFaceDataset(data_dir)
    
    print(f"Extracting features from {len(dataset.data)} images...")
    
    all_features = []
    all_ages = []
    all_genders = []
    all_races = []
    
    image_paths = dataset.data['path'].tolist()
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        features = extractor.extract_batch(batch_paths, batch_size)
        all_features.append(features)
        
        if i % 5000 == 0:
            print(f"  Processed {i}/{len(image_paths)}...")
    
    # Combine all features
    X_all = np.vstack(all_features)
    
    # Get labels and protected attributes
    y_age = dataset.data['age'].values
    a_gender = dataset.data['gender'].values
    a_race = dataset.data['race'].values
    
    # Save cached features
    np.savez(
        output_path,
        X=X_all,
        age=y_age,
        gender=a_gender,
        race=a_race
    )
    
    print(f"Saved features to {output_path}")
    print(f"Shape: {X_all.shape}")  # (200K, 512)
    
    return X_all, y_age, a_gender, a_race
```

---

## Fairness Tasks on UTKFace

### Task 1: Gender Prediction (Binary — like our current datasets)
```
Label y: Is this person male? (1) or female? (0)
Protected a: Actual gender (0=F, 1=M)

Fairness metric: |P(Y=1|A=0) - P(Y=1|A=1)| → DP
```

**Problem:** Model might learn facial features → use gender stereotype.

### Task 2: Age Prediction (Regression → Binned)
```
Label y: Young (0) / Middle (1) / Old (2)
Protected a: Gender or Race

Fairness: |P(Y=y|A=a) - P(Y=y|A=a')| for each y → DP
```

### Task 3: Race Prediction (Multi-class — harder!)
```
Label y: White/Black/Asian/Indian/Others
Protected a: Gender

Fairness: Group-wise DP (harder with 5 groups)
```

---

## What to Implement First

### Day 1 Priority: Binary Gender Prediction

```python
# Baseline pipeline for gender prediction
# Works like our current Adult/Credit/LSAC datasets

Task: Predict gender from face image
Label y: predicted gender (binary)
Protected a: actual gender (0/1)

1. Extract ResNet18 features (already have pretrained)
2. Train MLPClassifier on features (same as tabular)
3. Measure DP violation
4. Compare DRO-FAIR vs Naive-FAIR
```

This is the **simplest first task** — reuses the same DRO-FAIR code, just with image-derived features.

---

## GPU Memory Planning

| Operation | GPU Memory | Time |
|-----------|-----------|------|
| ResNet18 feature extraction (200K images, batch 64) | ~4GB | ~45 min |
| DRO-FAIR training on extracted features | ~2GB | ~20 min |
| Adversarial corruption (label + feature attacks) | ~1GB | ~5 min |
| **Total** | **~8GB** | **~70 min** |

**With 96GB available, we have headroom.** We could even:
- Fine-tune ResNet18 on UTKFace (additional 4GB)
- Run multiple experiments in parallel

---

## Implementation Checklist

- [ ] SSH into flair2 server
- [ ] Download UTKFace dataset to `/data/srujan.sai/UTKFace`
- [ ] Run feature extraction (45 min, do overnight)
- [ ] Add `load_utkface()` to `src/data/datasets.py`
- [ ] Replace `MLPClassifier` with ResNet18-based classifier
- [ ] Run first gender prediction experiment
- [ ] Compare with tabular results

---

## Next Steps After Feature Extraction

Once we have `utkface_features.npz`:

```python
# Load cached features
data = np.load('/data/srujan.sai/utkface_features.npz')
X = data['X']  # (200K, 512)
y = data['gender']  # 0/1 gender label
a = data['race']  # 0-4 race label

# Use gender as protected for binary fairness
# Train DRO-FAIR on gender prediction
```

---

## Files to Create

1. `src/data/utkface.py` — UTKFace loader + feature extractor
2. `src/models/cnn_classifier.py` — ResNet18-based classifier  
3. `experiments/run_utkface.py` — UTKFace experiment runner

---

## Questions for Professor

1. **Which task?** Gender prediction (easiest) or something else?
2. **Which protected attribute?** Gender (binary) or Race (5-class)?
3. **Pre-extract features or fine-tune end-to-end?** Pre-extract is faster but less powerful