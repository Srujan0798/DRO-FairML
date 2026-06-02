# DRO-FAIR Week 2 Orchestration Plan
**Deadline:** Tuesday May 26, 2026 (8 days)
**Author:** Srujan Sai (orchestrator: Claude)
**Goal:** Complete both tasks the professor assigned, push to GitHub, prepare a 5-minute progress demo for next meeting.

---

## 🎯 Professor's Tasks (verbatim)

> 1) Implement PGD for fairness metrics (Both DP and IF, only DP, only IF) and see the performance of DRO on Adult etc
> 2) Set up an experiment for the UTKFace dataset in the server and repeat the similar experiment

---

## 🔀 High-Level Architecture

```
              ┌────────────── Day 1 (research + design) ──────────────┐
              │                                                       │
   AGENT A ──→├─ Task 1: Fairness-targeted PGD on Adult (tabular)    │
              │  - Implement 3 new attack modes                       │
              │  - Run on Adult, Credit, LSAC                         │
              │  - Compare DRO vs Naive under each attack             │
              │                                                       │
   AGENT B ──→├─ Task 2: UTKFace pipeline (image)                     │
              │  - GPU server setup                                   │
              │  - Image preprocessing                                │
              │  - CNN classifier (replaces MLP)                      │
              │  - Adapt corruption pipeline for images               │
              │                                                       │
   ORCH ─────→├─ Daily sync, integration, report writing              │
              └───────────────────────────────────────────────────────┘
```

---

# 📅 Day-by-Day Plan

## Day 1 (May 19, Mon) — Research + Design

### Agent A (Fairness PGD)
**Reading (1h):**
- `src/corruption/adversarial.py` (current implementation, lines 50-110)
- Madry et al. 2018 — "Towards Deep Learning Models Resistant to Adversarial Attacks" — Section 2
- Solans, Biggio, Castillo 2021 — "Poisoning Attacks on Algorithmic Fairness"
- Your own report `submission/report.pdf` pages 1, 3, 7-8

**Deliverable (end of day):**
- Design doc: `docs/FAIRNESS_PGD_DESIGN.md` describing the 3 attack modes mathematically
- API sketch for `FairnessTargetedPGD` class

### Agent B (UTKFace)
**Reading (1h):**
- UTKFace dataset paper / website: https://susanqq.github.io/UTKFace/
- `src/data/datasets.py` (current data loaders)
- `src/models/classifier.py` (current MLP)
- PyTorch CNN tutorials (any standard ResNet-style)

**Deliverable (end of day):**
- Confirm GPU server access (check email reply from sysadmin)
- Design doc: `docs/UTKFACE_PIPELINE.md` describing image pipeline + CNN choice
- Identify minimum changes needed (data loader, model, possibly corruption)

---

## Day 2 (May 20, Tue) — Implementation Start

### Agent A
**Build:**
1. `src/corruption/fairness_pgd.py` — new file
2. Class `FairnessTargetedPGD(attack_mode, epsilon, steps, step_size)` where `attack_mode ∈ {'dp', 'if', 'joint', 'classification'}`
3. Method `attack(model, X, y, a, knn_graph) -> X_adv` that:
   - Forward through model
   - Compute the target loss (`g_DP`, `g_IF`, `λ_DP·g_DP + λ_IF·g_IF`, or classification)
   - Backprop w.r.t. **X** (not θ)
   - PGD step: `X ← clip(X + step_size · sign(∇_X loss), X_orig ± ε)`
   - Repeat `steps` times

**Tests:** `tests/test_fairness_pgd.py` — verify each mode produces nonzero perturbation, respects ε-ball, increases the target metric.

### Agent B
**Build:**
1. GPU server: SSH in, set up venv with PyTorch + CUDA
2. Download UTKFace to server: ~24K images, gender as protected attribute, age bucket as target
3. `src/data/utkface.py` — image loader (PyTorch `Dataset`)
4. `src/models/cnn_classifier.py` — simple CNN (3 conv blocks + 2 FC, ~1M params) OR pretrained ResNet18 with new head

**Tests:** `tests/test_utkface_loader.py` — verify dataset loads, shapes are correct.

---

## Day 3 (May 21, Wed) — Wire Up + Smoke Tests

### Agent A
1. `experiments/run_fairness_pgd.py` — new experiment driver
   - Loop over: dataset ∈ {adult, credit, lsac} × attack ∈ {classification, dp, if, joint} × method ∈ {naive, dro} × seed ∈ 0-9 = 720 runs
   - Reduce to **3 seeds for first smoke run** (= 72 runs, ~2 hours on CPU)
2. Run smoke test: 1 dataset × 1 attack × 1 method × 1 seed → confirm it doesn't crash
3. Verify the attack actually moves the metric: assert `g_DP(X_adv) > g_DP(X_clean)` for DP-mode

### Agent B
1. Adapt `src/training/naive_fair.py` and `dro_fair.py` to accept image tensors
   - The MLP forward pass is replaced by CNN
   - DP/IF losses unchanged (still operate on h_tilde)
2. Adapt corruption pipeline (`adversarial.py`) for images:
   - PGD on pixels with ε small (e.g., 4/255) for image-space attacks
   - Label flips: same logic
   - Attribute flips: same logic
3. Run smoke test on tiny subset (100 images, 5 epochs) — must complete without OOM

---

## Day 4 (May 22, Thu) — Full Experiments Begin

### Agent A — RUNNING
Launch full experiment in background:
```bash
python3 experiments/run_fairness_pgd.py --datasets adult credit lsac \
  --attacks classification dp if joint \
  --methods naive dro \
  --n_seeds 10
```
Expected runtime: ~4-6 hours on CPU. Use `tmux` or `nohup`.

### Agent B
1. Launch UTKFace baseline run on GPU server:
   - Method: Naive-FAIR only (no DRO yet)
   - α ∈ {0.0, 0.1, 0.2, 0.3}, 3 seeds
   - Target metric: accuracy, DP (gender), IF (k-NN on CNN features)
2. Monitor GPU memory + training time per epoch
3. If smoke run shows ~5 min/epoch with batch=128 on V100, full 60-epoch run = 5h per config; total ~60h. **May need to reduce to 30 epochs or fewer seeds.**

---

## Day 5 (May 23, Fri) — Analysis + Full UTKFace

### Agent A — ANALYSIS
1. Aggregate `results/fairness_pgd_results.json`
2. Generate **Figure A1: Attack-vs-Defense matrix** (4 attacks × 2 methods, heatmap of DP/IF degradation)
3. Generate **Table A1: Mean ± SE under each attack** (Wilcoxon p-values)
4. Write `docs/FAIRNESS_PGD_RESULTS.md` (2 pages, structured like a mini-paper)

### Agent B
1. Launch full DRO-FAIR runs on UTKFace
2. Estimated time: 24-36h. Keep monitoring; abort early if collapsing.

---

## Day 6 (May 24, Sat) — Integration

### Both agents merge work:
1. Agent A finishes writeup
2. Agent B finishes UTKFace runs, generates Figure B1 + Table B1

### Orchestrator (you):
1. Pull both branches, resolve conflicts
2. Update `report/report.tex` with a new **Section 13: Week 2 Extensions**
3. Re-render PDF

---

## Day 7 (May 25, Sun) — Dry Run + Polish

1. Final results check: do the numbers tell a coherent story?
2. Update `CHEAT_SHEET.md` with week 2 numbers
3. Practice the 5-minute progress demo
4. Push everything to GitHub, tag as `v1.1`

---

## Day 8 (May 26, Tue) — Meeting Day

**Before 4 PM:**
- Run `pytest tests/ -q` → all green
- Run `python3 experiments/run_fairness_pgd.py --quick-sanity` → confirms results reproduce
- Open the 5-minute demo deck

**At 4 PM:**
- 1 min: recap
- 2 min: Fairness-PGD results
- 2 min: UTKFace results
- Answer questions

---

# 🤖 Agent A Prompt Template — Fairness PGD

Use this as the prompt when spawning Agent A:

```
You are AGENT A. Your job: implement fairness-targeted PGD attacks for the
DRO-FAIR project at /Users/srujansai/Desktop/DRO-FairML.

DELIVERABLE:
- src/corruption/fairness_pgd.py — FairnessTargetedPGD class
- tests/test_fairness_pgd.py — unit tests, all passing
- experiments/run_fairness_pgd.py — experiment driver
- docs/FAIRNESS_PGD_DESIGN.md — math + API doc
- docs/FAIRNESS_PGD_RESULTS.md — results writeup after experiments

WHAT THE ATTACK DOES:
Current adversarial.py does PGD on classification loss: δ = ε·sign(∇_X L_cls).
You will add 3 new modes that target FAIRNESS METRICS instead:
  - 'dp':    δ = ε·sign(∇_X g_DP)
  - 'if':    δ = ε·sign(∇_X g_IF)
  - 'joint': δ = ε·sign(∇_X (λ_DP·g_DP + λ_IF·g_IF))
where g_DP is the demographic parity gap and g_IF is the k-NN fairness loss
already defined in src/training/dro_fair.py (lines 100-140).

KEY CONSTRAINTS:
- Reuse the existing g_DP and g_IF loss functions (don't reimplement)
- Respect ε-ball in feature space (ε=0.1, 10 PGD steps)
- Run on Adult, Credit, LSAC, all α ∈ {0.1, 0.2, 0.3}, 10 seeds
- Compare Naive-FAIR vs DRO-FAIR under each attack
- Use Wilcoxon p<0.05 for win counts (matches report convention)

CRITICAL: After implementing, run experiments and produce figures + table.
Read docs/FAIRNESS_PGD_DESIGN.md before coding to confirm the math is right.
```

---

# 🤖 Agent B Prompt Template — UTKFace

```
You are AGENT B. Your job: set up the UTKFace experiment pipeline on the GPU
server for the DRO-FAIR project at /Users/srujansai/Desktop/DRO-FairML.

DELIVERABLE:
- src/data/utkface.py — UTKFace PyTorch Dataset class
- src/models/cnn_classifier.py — CNN model (ResNet18 with new head OR custom 3-block CNN)
- src/training/dro_fair.py — adapt to accept image inputs (no major changes; the
  MLP→CNN swap is in the model layer)
- src/corruption/image_pgd.py — image-space PGD (ε=4/255, attacks pixels)
- experiments/run_utkface.py — experiment driver
- docs/UTKFACE_PIPELINE.md — design doc
- docs/UTKFACE_RESULTS.md — results writeup

DATASET DETAILS:
- UTKFace: ~24K face images, 200x200 RGB, labels: age (0-116), gender (0/1), race (0-4)
- Protected attribute: gender (binary, as before)
- Target: age bucket (binary: under-40 vs 40+) OR race (multi-class — simpler to start with gender as target)
- Source: https://susanqq.github.io/UTKFace/

CRITICAL CONSTRAINTS:
- You must run on GPU (CPU will take too long). Confirm sysadmin grants access.
- Batch size: start at 128, drop to 64 if OOM
- Use mixed precision (torch.cuda.amp) to speed up
- α ∈ {0.0, 0.1, 0.2, 0.3} (skip 0.4 to save time)
- 3-5 seeds initially (not 10) due to compute cost
- Both methods: Naive-FAIR and DRO-FAIR

CRITICAL: The MLP→CNN swap is the only major code change to the training pipeline.
The DRO algorithm itself (θ→λ→p loop) is unchanged. Just the f_θ becomes a CNN.

Coordinate with Agent A: confirm the corruption pipeline interface so PGD attacks
work consistently across tabular and image data.
```

---

# 📊 Success Criteria

By Tuesday May 26 EOD, you should have:

| Deliverable | Status |
|---|---|
| `src/corruption/fairness_pgd.py` | exists, tested |
| `src/data/utkface.py` | exists, tested |
| `experiments/run_fairness_pgd.py` results | 720 experiments complete |
| `experiments/run_utkface.py` results | ≥ 24 experiments complete |
| Figure: Attack-vs-defense matrix (Adult) | committed |
| Figure: UTKFace fairness curves | committed |
| Updated report PDF with Section 13 | committed |
| GitHub tag `v1.1` | pushed |
| 5-minute progress slides | prepared |

---

# 🚨 Risk Register

| Risk | Mitigation |
|---|---|
| GPU access delayed | Start UTKFace prep locally (CPU smoke tests with subset of 100 images) |
| UTKFace download too large/slow | Use the cropped+aligned 23K version, not the in-the-wild version |
| OOM on GPU | Drop batch size, use gradient accumulation, use mixed precision |
| Fairness PGD doesn't degrade DP | Verify gradient direction; may need larger ε or more steps |
| Adult α=0.3 still collapses | Document as expected — same failure mode as week 1 |
| Time overrun | Cut UTKFace seeds to 3, document as "preliminary" for the meeting |

---

# 🎤 Meeting Prep (Day 8)

**5-minute structure:**

1. **Slide 1: Week 2 Recap (30 sec)**
   - Tasks completed: PGD-fairness on Adult/Credit/LSAC, UTKFace baseline
   - Honest list of what's done vs in-progress

2. **Slide 2: Fairness-PGD Results (90 sec)**
   - Attack-vs-defense matrix
   - Key finding: DRO defends against DP-PGD on Credit/LSAC; Adult still collapses

3. **Slide 3: UTKFace Setup (90 sec)**
   - Pipeline architecture
   - Baseline numbers (just Naive-FAIR if DRO not done)

4. **Slide 4: Surprises / Issues (60 sec)**
   - Any unexpected findings — show you're a researcher, not a code monkey

5. **Slide 5: Next Week (30 sec)**
   - What to do next based on this week's findings

---

# 📞 Daily Standup (text yourself or me)

Each day at 9 PM, write 3 lines:
1. What I did today
2. What I'm doing tomorrow
3. What's blocking me

If anything is blocking → escalate immediately, don't wait.

---

**Status:** Day 0 (May 18) — planning complete. Start Day 1 tomorrow morning.
