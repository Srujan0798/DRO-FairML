# DRO-FairML — Full audit for boss review (stop point)

**Date:** 2026-08-04 (late evening)  
**Author:** agent session (Grok) after stop-all request  
**Purpose:** Single snapshot so leadership can re-task. **No experiments running** (10m loop cancelled; experiment workers stopped).

---

## 0. Executive verdict

| Bucket | Verdict |
|--------|---------|
| **Ship package (Aug 10 core)** | **READY** — science locked, paper/report rebuilt, claims audited |
| **Advisor “extra” ablations** | **PARTIAL** — A3 λ-grid done; others mid-run (safe to resume later) |
| **flair2 NVIDIA** | **UNLOCKED** — 2× L40S + torch CUDA; optional for speed/extra runs |
| **What we stopped** | Recurring 10m loop + local ablation workers |
| **Do not retrain** | `results/canonical_tau1.json` core 540 story; UTKFace 90 REAL |

**Bottom line for boss:** Core DRO-FAIR story under Fairness-Targeted PGD is complete and honest. Remaining work is optional ablations + optional GPU-scale runs, not a blocker for sharing paper/report with locked claims.

---

## 1. What was stopped (this session)

1. **Scheduler loop** `019fcdc67f53` (every 10m “continue improving”) — **cancelled**.
2. Local **Wave-1 ablation parents/workers** (`run_a*`, orchestrators, etc.) — **stopped**.
3. No further auto-commits or auto-launches until new tasks are assigned.

---

## 2. Locked science (do not re-open without greenlight)

### 2.1 Tabular canonical grid
| Item | Value |
|------|--------|
| File | `results/canonical_tau1.json` |
| Protocol | τ=1, K_inner=10, epochs=60, PGD steps=20, n_seeds=6, radii_mode=uniform |
| Grid | 3 datasets × 5 α × {dp, if, combined} × 6 seeds × 2 methods |
| Core claim rows | **540** (dp/if/combined = 180 each) |
| Note | File may contain **extra extension rows** (e.g. total ~560 if n=10 pilot started). **Claims and Wilcoxon use locked seeds 0–5 / 540 protocol.** Do not rewrite 540. |

### 2.2 Headline claims (verified)
1. Fixed **τ=1** makes DRO beat Naive on **Adult & Credit, α≤0.2**, DP + Combined (Wilcoxon p&lt;0.05).
2. Adult/DP **α=0.1 is 5/6**, not “6/6 every α”.
3. **IF-attack third: MIXED** (not a clean three-attack mirror). IF *metric* wins on Adult/Credit for α∈{0.1–0.4}; DP-under-IF can lose (e.g. Adult α=0.3).
4. **LSAC/DP: degenerate** (0/6; accuracy pinned ~0.90 majority).
5. **α≥0.3:** both methods below constant-predictor accuracy on **Adult/Credit only** (not LSAC).
6. **UTKFace:** REAL ResNet18 features, **90/90**, clean-test **mixed** (significant DRO DP mainly at high α) — not Adult copy-paste.

### 2.3 UTKFace
| Item | Value |
|------|--------|
| Features | `data/raw/utkface_features.npz` (N=23705, REAL) |
| Results | `results/utkface_canonical.json` — **90/90**, provenance REAL |
| Summary | `results/utkface_summary.md` |
| Run hardware | Mac MPS (not flair2) |

### 2.4 Share package
| Deliverable | Path |
|-------------|------|
| Paper | `paper/main.pdf` |
| Report | `report/report.pdf` |
| Meeting handout (GChat-safe) | `docs/MEETING_HANDOUT_2026-08-04.md` |
| Live board | `STATUS.md` |
| Verification | `docs/VERIFICATION_REPORT.md` |
| Advisor map | `docs/ADVISOR_CONCERNS_CHECKLIST.md` |

---

## 3. What we finished this push (beyond baseline 540)

| Item | Status | Evidence |
|------|--------|----------|
| Full IF third + real IF Wilcoxon | Done | `results/if_wilcoxon_summary.txt`, paper/report mixed narrative |
| Claim hygiene (5/6, LSAC pinned, no false three-attack sweep) | Done | STATUS, VERIFICATION, paper abstract/results |
| Paper figures/tables wired from 540 | Done | `paper/main.pdf` larger package with figs |
| UTKFace REAL 90 pilot + honest mixed write-up | Done | utkface_canonical + summary + paper section |
| **A3 λ/lr grid (Kuldeep Q1)** | **Complete 72/72** | `results/lambda_grid.json` + summary; **no α=0.3 acc rescue** above ~0.752 |
| flair2 NVIDIA unlock | **Done** | `docs/FLAIR2_GPU_READY.md`; torch 2.6.0+cu124; `cuda True count 2`; Adult DRO smoke ~33s on L40S |
| Unit tests / validate | Green at last checks | ~90 tests; validate PASS on 540 DP gate |

---

## 4. Ablation progress at stop (partial — resume-safe)

All write **separate JSON files** (never intended to overwrite canonical 540).

| Code | Ablation | Target | At stop | File |
|------|----------|--------|---------|------|
| A3 | λ_init × lr_λ (Adult DP) | 72 | **72/72 ✅** | `results/lambda_grid.json` |
| A4 | Random vs adversarial | 144 | **43/144** | `results/random_vs_adversarial.json` |
| A5 | Empirical radii | 180 | **69/180** | `results/empirical_radii.json` |
| A2 | τ ∈ {10,100} | 360 | **76/360** | `results/tau_ablation.json` |
| A1 | kNN attack_k ∈ {5,15} | 360 | **48/360** | `results/knn_ablation.json` |
| N5 | K_inner ∈ {5,20} | 180 | **24/180** | `results/kinner_ablation.json` |
| (extra) | Other pilots (high-α, radii, COMPAS loaders, etc.) | varies | partial | see `results/*` |

**Partial-data rule:** Do **not** put unfinished multipliers (e.g. “12–40×”) into abstract until A4 has full n=6 cells. Early partial A4 already suggests observed multipliers may be **≪ 12×** — needs full grid before a corrected number.

Summaries (analysis-only, re-run when complete): `results/*_summary.md`.

---

## 5. Compute inventory

| Resource | State |
|----------|--------|
| **Mac M4 Pro** | Local work; ablations were CPU; MPS available; **no NVIDIA** |
| **flair2** | SSH OK; **2× L40S**; driver 570; **venv_gpu** with CUDA torch; **no outbound PyPI** (offline wheelhouse only) |
| Project on server | `/data/srujan.sai/DRO-FairML-run` |
| Wheelhouse | `/data/srujan.sai/wheelhouse` (~3.7G) |

---

## 6. Honest gaps / risks (for boss tasking)

### Science / claims
1. **A4 random-vs-adv incomplete** — historical “12–40×” may need revision once full 144 finishes.
2. **Canonical file size creep** — if total rows &gt; 540, treat extras as extension only; SSOT claims remain n=6 / 540.
3. **UTKFace** is feature-space pilot, not pixel-space attack (paper states this).
4. **LSAC/DP** remains a negative/degenerate result (documented).

### Engineering / process
1. **Multi-agent collision history** — dual writers, `pkill` wars, SIGTERM of long jobs. Fix direction: one parent per results file + file locks (partially landed).
2. **Do not** restart 10m loops that kill experiments.
3. flair2 installs **must** stay offline (`--no-index --find-links wheelhouse`).

### Advisor asks still open (from MASTER_PROTOCOL)
- Finish A1/A2/A4/A5/N5 grids (or descope with written reasons).
- Optional: empirical-radii full table in paper only after A5 complete.
- Optional: flair2 UTKFace re-run for “server” label (Mac already has REAL 90).
- Optional: pixel-level / larger GPU experiments (new design — not required for current claim set).

---

## 7. Suggested task menu for boss reassignment

Pick any subset; each is independent of locked 540.

| Priority | Task | Owner type | Depends on |
|----------|------|------------|------------|
| P0 | Review/share `paper/main.pdf` + `report/report.pdf` with locked claims | Human | Nothing |
| P1 | Finish **A4** to 144 → honest random-vs-adv multiplier | Compute (Mac or flair2) | Sole writer |
| P1 | Finish **A2 τ** ablation → clean τ=100 artifact table | Compute | Sole writer |
| P1 | Finish **A5** empirical radii → appendix table | Compute | Sole writer |
| P2 | Finish A1 kNN + N5 K_inner | Compute | Sole writer |
| P2 | Fold completed A3 λ findings already in appendix; verify PDF | Writing | A3 done |
| P3 | Move remaining ablations to **flair2 CUDA** for speed | Infra + compute | GPU ready |
| P3 | Optional flair2 UTKFace re-run (server provenance) | Compute | GPU ready |
| Later | Pixel-level PGD / end-to-end image attack | Research design | Explicit greenlight |

---

## 8. What agents were doing before stop (audit of activity)

1. **Maintaining ship package** — claim audit, paper/report rebuilds, tests/validate.
2. **Running Wave-1 ablations** on Mac CPU (parallel parents at times — risky).
3. **10m improve loop** — snapshots, summaries, prose polish; cancelled for boss review.
4. **Unlocking flair2 NVIDIA** — offline wheel download + install + smoke.
5. **Not** rewriting the core 540 claim set.

---

## 9. Commands for humans

```bash
# Verify science counts
python3 -c "import json; print(len(json.load(open('results/canonical_tau1.json'))))"
python3 -c "import json; print(len(json.load(open('results/utkface_canonical.json'))))"

# Gate
make test && make validate && make paper && make report

# flair2 GPU check
ssh flair2 'cd /data/srujan.sai/DRO-FairML-run && source venv_gpu/bin/activate && python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"'
```

---

## 10. Sign-off

| Role | State |
|------|--------|
| Experiments | **Stopped** |
| Auto-loop | **Stopped** |
| Core deliverables | **Ready for review** |
| Waiting on | Boss re-task list |

**Repo tip at audit:** see `git log -1` on machine after pull; recent highlights include A3 complete, flair2 GPU unlock, checkpoint commits.
