# COMPLETE ASSIGNMENT LIST — Week 2
**Deadline:** May 29, 2026, 3:00 PM (Madam meeting)

---

## ASSIGNMENT 1: Monitor Experiments (NO ACTION NEEDED)
**Owner:** Running automatically  
**Status:** PID 81903 alive, 87/270 done  
**ETA:** ~2.5 hours (finishes ~2 AM)  
**Check:** `python3 scripts/check_progress.py`

---

## ASSIGNMENT 2: Generate Full Figures + Analysis (Do AFTER experiments finish)
**Owner:** You or anyone with Python  
**Time:** 15 minutes  
**When:** Tomorrow morning (after ~2 AM)

```bash
cd /Users/srujansai/Desktop/DRO-FairML
venv/bin/python3 experiments/analyze_fairness_pgd.py
```

**Outputs:**
- `figures/fig8_fairness_pgd_comparison.png/pdf`
- `figures/fig9_fairness_pgd_curves.png/pdf`
- `results/fairness_pgd_wilcoxon.csv`
- `results/fairness_pgd_summary.csv`

**Then run:**
```bash
venv/bin/python3 scripts/auto_generate_deliverables.py
```

---

## ASSIGNMENT 3: GPU Server Access (CRITICAL)
**Owner:** Srujan  
**Time:** 30 minutes  
**When:** ASAP (today/tomorrow morning)

1. Get hostname + password from sysadmin
2. Test: `ssh srujan.sai@<hostname>`
3. Download UTKFace data
4. Run feature extraction:
```bash
ssh srujan.sai@<hostname>
cd /home/srujan.sai/DRO-FairML
python3 scripts/extract_utkface_features.py \
  --data-dir /data/srujan.sai/UTKFace \
  --output /data/srujan.sai/utkface_features.npz
```

5. Run experiments:
```bash
python3 experiments/run_utkface.py --alphas 0.0 0.1 0.2 --n_seeds 3
```

---

## ASSIGNMENT 4: Git Push
**Owner:** Srujan  
**Time:** 2 minutes  
**When:** After Assignment 2

```bash
cd /Users/srujansai/Desktop/DRO-FairML
git push origin main
```

---

## ASSIGNMENT 5: Final Report for Madam
**Owner:** Srujan  
**Time:** 1 hour  
**When:** Friday morning (May 29)

**File:** `docs/ADVERSARIAL_FAIRNESS_REPORT.md`

**Replace placeholders with:**
1. Actual numbers from `results/fairness_pgd_wilcoxon.csv`
2. Actual figures from `figures/fig8_*.png` and `figures/fig9_*.png`
3. UTKFace status (if server access done)

**For screen-share:** Open report + figures side by side.

---

## ASSIGNMENT 6: What to Tell Madam
**Owner:** Srujan  
**When:** Friday 3 PM

**If full results ready:**
> "Madam, we implemented gradient-based attacks targeting DP, IF, and joint metrics. Under DP-targeted attacks, DRO [reduces/increases] DP by X%. Full results on 3 datasets, 5 seeds, 3 alphas."

**If UTKFace ready:**
> "We also tested on UTKFace — 200K images. DRO [result]."

**If NOT ready:**
> "Pipeline implemented and smoke-tested. Full ablation running, complete results by next Friday. No submitted code was modified."

---

## PRIORITY ORDER

| Priority | Task | Owner | Deadline |
|----------|------|-------|----------|
| 1 | Wait for experiments to finish | Automatic | Tonight ~2 AM |
| 2 | Generate figures + analysis | Anyone | Thu morning |
| 3 | GPU server access | Srujan | Thu morning |
| 4 | UTKFace feature extraction | Srujan | Thu afternoon |
| 5 | Update report | Srujan | Fri morning |
| 6 | Madam meeting | Srujan | Fri 3 PM |

---

## FILES TO SHOW MADAM

| File | What it shows |
|------|--------------|
| `src/corruption/adversarial.py` | FairnessTargetedPGD code |
| `figures/fig8_fairness_pgd_comparison.png` | Attack comparison |
| `figures/fig9_fairness_pgd_curves.png` | DP vs alpha curves |
| `docs/ADVERSARIAL_FAIRNESS_REPORT.md` | Full progress |
| `results/fairness_pgd_results.json` | Raw data proof |
