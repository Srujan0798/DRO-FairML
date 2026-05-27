# DRO-FAIR - Fairness Research Project

## ONE COMMAND (Does everything):
```bash
cd /data/srujan.sai/DRO-FairML && git pull origin main && export PYTHONPATH=/data/srujan.sai/DRO-FairML && python3 scripts/test_fairness_pgd.py
```

## PUSH CODE (For review):
```bash
cd /data/srujan.sai/DRO-FairML && git add . && git commit -m "NAME: did X" && git push origin main
```

## WRITE REPORT:
```bash
echo "DONE: DP results etc" >> /data/srujan.sai/DRO-FairML/REPORTS/daily.txt
```

## AUTO RUNNER (All in one):
```bash
bash /data/srujan.sai/DRO-FairML/scripts/auto_runner.sh
```

---

## Project Overview
- **Goal**: Test Fairness-Targeted PGD attacks on ML models (DRO-FAIR framework)
- **Datasets**: Adult, Credit, LSAC (tabular), UTKFace (image)
- **Attack**: Gradient-based label flipping to maximize unfairness (DP violation)

## Key Files
- `src/corruption/adversarial.py` - FairnessTargetedPGD class
- `scripts/test_fairness_pgd.py` - Main experiment script
- `scripts/auto_runner.sh` - Runs everything automatically

## Login
- JupyterHub: http://flair2.iitgn.ac.in:8000/hub/login
- User: srujan.sai | Pass: ss#081
- Server: flair2.iitgn.ac.in (10.0.62.234)