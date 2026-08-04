# Weekly Progress Report — Tuesday, June 2, 2026
**Student:** Srujan Sai · **Course:** AI & ML · **Project:** DRO-FAIR

---

## Where We Started

**Original project (v1.0, submitted previously):** Implementation and empirical
evaluation of DRO-FAIR (Algorithm 1, ICML submission). My contribution was
replacing the paper's random-noise corruption model with **multi-modal
adversarial corruption** — PGD on features, coordinated label flips,
minority-targeted attribute flips. Evaluated on Adult, Credit, LSAC across
150 experiments. Tagged `v1.0` on GitHub.

---

## What Madam Asked Last Meeting

> "She asked me to: (1) implement PGD for fairness metrics — both DP-and-IF,
> only-DP, only-IF — and see DRO performance on Adult etc.; (2) set up an
> experiment for the UTKFace dataset on the server and repeat the similar
> experiment. Madam also said the original adversarial study was on small
> datasets so I should go larger."

---

## What I Did This Week

### Task 1 — PGD attacks targeting fairness metrics

**Implementation:** New `FairnessTargetedPGD` class in
`src/corruption/adversarial.py`. Three attack modes:

| Mode | Gradient | Picks samples that, when flipped, …  |
|---|---|---|
| `dp` | ∂(DP_gap)/∂y | maximise demographic-parity gap |
| `if` | ∂(IF_violation)/∂y | maximise k-NN individual-fairness violation |
| `combined` | weighted sum | maximise both jointly |

Each mode flips the top ⌊αn⌋ samples by absolute gradient. 70% of flips
target the minority group when `coordinated=True`. Reference: Solans et al.,
*Poisoning Attacks on Algorithmic Fairness*, ECML 2021.

**Experiments:** 3 datasets × 3 attacks × 2 methods × 3 α values × 5 seeds
= **270 runs.** Results in `results/fairness_pgd_results.json`,
significance tests in `results/fairness_pgd_wilcoxon.csv`.

**Headline results** (Wilcoxon paired test, one-sided H₁: Naive DP > DRO DP):

| Dataset | α | Attack | DP reduction (DRO vs Naive) | p-value |
|---|---|---|---|---|
| Credit | 0.2 | IF | **+64.5%** | 0.031 ✓ |
| Credit | 0.3 | IF | **+97.5%** | 0.031 ✓ |
| LSAC | 0.3 | IF | **+96.2%** | 0.031 ✓ |
| Adult | 0.1–0.3 | DP | DRO loses (feedback loop) | n.s. |
| All | 0.1 | any | weak signal | n.s. |

DRO wins under IF-targeted attacks on Credit/LSAC at moderate-to-high
corruption. Adult continues to fail under DP attacks (same λ_DP-runaway
feedback loop documented in v1.0).

**Figures:**
- `figures/fig8_attack_defense_matrix.pdf` — 3×3 attack-vs-dataset heatmap
- `figures/fig9_fairness_pgd_curves.pdf` — DP curves vs α per attack

#### On the α=0.1 "no significant difference"

A natural follow-up question: *if DRO doesn't win at α=0.1, is the attack
affecting the radius? Or is the attack just too weak?*

**The TV radii depend only on α (the corruption budget), not on attack design.**
ρ_DP,j = α/((1−α)π_j + α) and ρ_IF = 2α − α² are functions of α only.
Attack design (gradient direction, PGD steps, ε) controls how damaging each
corrupted sample is, but the radius — and thus DRO's defensive envelope — is
fixed by α.

At α=0.1, both Naive and DRO perform similarly because the 10% corruption
budget is too small for Naive to actually fail. DRO is calibrated for the
worst-case-within-α=0.1, but if realized corruption is not worst-case, DRO
slightly **over-prepares** — the classical "robustness tax" of distributionally
robust optimization. Example, Credit under IF attack:

| α | Naive DP | DRO DP | DP change | Note |
|---|---|---|---|---|
| 0.1 | 0.0133 | 0.0131 | +1.8% | tie (n.s.) — robustness tax visible |
| 0.2 | 0.0237 | 0.0084 | **+64.5%** | DRO wins (p=0.031) |
| 0.3 | 0.0823 | 0.0021 | **+97.5%** | DRO dominates (p=0.031) |

This is the **textbook DRO pattern**: little advantage at low corruption (cost
of being prepared for a threat that didn't materialise), large advantage at
high corruption (protection actually pays off).

---

### Task 2 — UTKFace experiment on GPU server

**Setup:**
- 24,000 UTKFace face images
- ResNet18 (ImageNet-pretrained) for 512-dim feature extraction
- Gender as binary protected attribute
- GPU server: flair2.iitgn.ac.in (access granted Friday May 29)
- 5 seeds × 3 α values × 2 methods = **15 baseline runs**

**Results** (evaluated on corrupted test set, mean of 5 seeds):

| α | Naive DP | DRO DP | Naive Acc | DRO Acc | Verdict |
|---|---|---|---|---|---|
| 0.0 | 0.029 | 0.023 | 0.862 | 0.863 | DRO better (no corruption) |
| 0.1 | 0.116 | 0.141 | 0.859 | 0.860 | **Naive better — DRO 22% worse** |
| 0.2 | 0.080 | 0.092 | 0.852 | 0.849 | **Naive better — DRO 15% worse** |

**Figure:** `figures/fig10_utkface_curves.pdf` — DP/IF curves vs α.

---

## The Surprising Finding: DRO Inverts on Image Features

On tabular Credit/LSAC, DRO **reduces** fairness violation under corruption.
On UTKFace, DRO **increases** it. The behavior inverts.

Trend is consistent across all 5 seeds at α=0.1 and α=0.2, though p > 0.05
with n=5.

### Three candidate hypotheses

**H1: ResNet18 features lack demographic signal**
Pretrained on ImageNet → features are roughly gender-agnostic. DRO's
worst-case reweighting has no demographic axis to anchor on, so it
over-corrects on noise.

**H2: Feature-space attacks don't simulate real corruption**
I attack cached 512-dim ResNet outputs, not raw pixels. A realistic
attacker would perturb pixels and propagate through the CNN. DRO is
defending against an attack pattern that doesn't reflect deployment.

**H3: Inner maximization amplifies noise on continuous embeddings**
On tabular features the inner-max over weights p finds clear high-DP
samples. On continuous 512-dim embeddings, the inner-max may find
spurious directions, causing λ_DP to over-grow.

I do not have a confident answer yet. That's where I want guidance.

---

## Proposed Next Week

1. **λ_DP trajectory diagnostic on UTKFace** — does it runaway like Adult? If yes → supports H3.
2. **Run UTKFace with λ_max = 0.5** instead of 1.5 — does DRO stop inverting? Quick test of H3.
3. **Pixel-space PGD vs feature-space attack** — directly tests H2.
4. **Random-init backbone or train-from-scratch CNN** — directly tests H1.
5. **Extend UTKFace to α ∈ {0.3, 0.4}** — fill the corruption sweep.
6. **Run FairnessTargetedPGD on UTKFace** — the new attacks haven't been tried on images yet.

---

## Honest Limitations

- UTKFace n=5 seeds. Inversion trend is consistent but not statistically
  significant (p > 0.05). Need more seeds.
- UTKFace only α ∈ {0, 0.1, 0.2}. Need α ∈ {0.3, 0.4} to compare with
  tabular sweep.
- UTKFace uses cached features, not end-to-end pixel pipeline.
- The new FairnessTargetedPGD attacks were run only on tabular data, not
  on UTKFace.

---

## Artifacts

| Item | Location |
|---|---|
| Original v1.0 report | `submission/report.pdf` |
| Source code | `src/` |
| Task 1 experiment runner | `experiments/run_fairness_pgd.py` |
| Task 1 analysis | `experiments/analyze_fairness_pgd.py` |
| Task 2 experiment runner | `experiments/run_utkface.py` |
| Task 2 analysis | `experiments/analyze_utkface.py` |
| Task 1 results | `results/fairness_pgd_*.json/.csv` |
| Task 2 results | `results/utkface_results.json` |
| Figures | `figures/fig8–fig11` (see above) |
| Theory verification | `experiments/verify_theory.py` (8/8 pass) |
| Tests | `tests/` (40 passing) |
| GitHub | https://github.com/Srujan0798/DRO-FairML · tag `v1.0` |

---

## Status One-Liner

Both tasks shipped on the assigned datasets. A new finding — DRO behavior
inverts between tabular and image features — needs investigation. Ready
for guidance on next-week direction.
