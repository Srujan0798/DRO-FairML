# ALL REMAINING TASKS — DRO-FAIR Week 2
**Date:** May 27, 2026  
**Meeting:** May 29, 3 PM with Madam  

---

## RUNNING NOW (Do Not Touch)

| Task | Status | ETA | Location |
|------|--------|-----|----------|
| Fairness-PGD 264 experiments | 🔄 RUNNING | ~4 hours | PID 67831, logs/batch_fpgd.log |

**What this does:** 3 datasets × 3 alphas × 5 seeds × 3 attacks × 2 methods = 270 total − 6 smoke = 264 new experiments. Saves after each one to `results/fairness_pgd_results.json`.

**DO NOT:** Launch another run, edit the JSON, or kill PID 67831.

**Monitor:** `tail -f logs/batch_fpgd.log` (will show output once stdout buffer flushes)

---

## TASK 1 — Analyze Results + Generate Figures (Blocked until experiments finish)

**Owner:** Anyone with Python  
**Time:** 30 minutes  
**Blocked by:** Task above (needs `results/fairness_pgd_results.json` with 200+ rows)

**Exact commands:**
```bash
cd /Users/srujansai/Desktop/DRO-FairML
venv/bin/python3 experiments/analyze_fairness_pgd.py
```

**Outputs:**
- `results/fairness_pgd_summary.csv` — mean ± SE table
- `results/fairness_pgd_wilcoxon.csv` — statistical tests
- `figures/fig8_fairness_pgd_comparison.png/pdf` — bar chart
- `figures/fig9_fairness_pgd_curves.png/pdf` — line chart

**Deliverable for Madam:** Figures + 1-page table of Wilcoxon results.

---

## TASK 2 — UTKFace GPU Server (Critical for Madam)

**Owner:** Srujan (needs server credentials)  
**Time:** 2–3 hours  
**Blocked by:** GPU server hostname + password

### Step 2a: Confirm server access (15 min)
```bash
ssh srujan.sai@<gpu-server-hostname>
# If fails, email sysadmin again
```

### Step 2b: Download UTKFace data (30 min)
Options:
1. Kaggle: `kaggle datasets download -d jangedoo/utkface-new`
2. Mirror: `wget https://github.com/moo-simple-unet/releases/download/v1.0/UTKFace.tar.gz`
3. Manual upload via `scp`

Target: `/data/srujan.sai/UTKFace/` (create if needed)

### Step 2c: Extract ResNet18 features (45 min GPU)
```bash
ssh srujan.sai@<gpu-server>
cd /home/srujan.sai/DRO-FairML
python3 scripts/extract_utkface_features.py \
  --data-dir /data/srujan.sai/UTKFace \
  --output /data/srujan.sai/utkface_features.npz \
  --batch-size 128
```

Output: `utkface_features.npz` (X: 200K×512, gender: 200K, race: 200K)

### Step 2d: Run UTKFace experiments (30 min)
```bash
python3 experiments/run_utkface.py --alphas 0.0 0.1 0.2 --n_seeds 3
```

Output: `results/utkface_results.json`

### Step 2e: Generate UTKFace figure (15 min)
```bash
# Add analyze_utkface.py if needed, or adapt analyze_fairness_pgd.py
```

**Deliverable for Madam:** `results/utkface_results.json` + one figure showing DRO vs Naive on 200K images.

---

## TASK 3 — Commit + Push Everything

**Owner:** Srujan  
**Time:** 10 minutes  
**When:** After all results are in

```bash
cd /Users/srujansai/Desktop/DRO-FairML
git add -A
git status
git commit -m "Week 2 complete: Fairness-PGD 270 experiments + UTKFace pipeline + figures"
git push origin main
```

---

## TASK 4 — Update Report for Madam Meeting

**Owner:** Srujan (or whoever writes)  
**Time:** 1 hour  
**When:** May 29 morning

**File:** `docs/ADVERSARIAL_FAIRNESS_REPORT.md`

**Replace placeholders with:**
1. Actual Wilcoxon p-values from `results/fairness_pgd_wilcoxon.csv`
2. Actual figure paths
3. UTKFace results (if available)
4. Key sentence with real numbers: "DRO reduces DP by X% under DP-attack"

**For screen-share:** Open the markdown file + figures in split view.

---

## TASK 5 — Run Tests (Optional but good)

**Owner:** Anyone  
**Time:** 5 minutes

```bash
cd /Users/srujansai/Desktop/DRO-FairML
venv/bin/python3 -m pytest tests/test_fairness_pgd.py -v
```

Expected: 8/8 tests pass.

---

## PRIORITY ORDER FOR MADAM MEETING

**Must have (Friday 3 PM):**
1. ✅ FairnessTargetedPGD code working
2. ✅ Preliminary figure (`figures/fig8_fairness_pgd_smoke.png`)
3. 🔄 Full results + figures (Task 1)
4. ⏳ UTKFace pipeline status (Task 2 — at minimum, show server setup script)

**Nice to have:**
- UTKFace actual results
- 10 seeds instead of 5
- Credit + LSAC full ablations

---

## WHAT TO TELL MADAM IF TASK 2 FAILS

> "UTKFace pipeline is implemented and tested locally with synthetic data. GPU server access is confirmed [or: pending confirmation]. Real feature extraction and experiments will run as soon as server is accessible — full results by next Friday."

---

## FILE REFERENCE

| File | What it is | Status |
|------|-----------|--------|
| `src/corruption/adversarial.py` | FairnessTargetedPGD class | ✅ Committed |
| `experiments/run_fairness_pgd_batch.py` | Robust batch runner | ✅ Committed |
| `experiments/analyze_fairness_pgd.py` | Analysis + figures | ✅ Committed |
| `tests/test_fairness_pgd.py` | 8 pytest tests | ✅ Committed |
| `scripts/setup_server.sh` | GPU server automation | ✅ Committed |
| `results/fairness_pgd_results.json` | Raw results | 🔄 13/270 rows |
| `figures/fig8_fairness_pgd_smoke.png` | Preliminary figure | ✅ Ready |
| `docs/ADVERSARIAL_FAIRNESS_REPORT.md` | Report template | ✅ Ready |
