# HANDOFF — Grok

**Your lane: flair2 GPU + the metrics/attack math.** You do not touch the Mac CPU
ablation queue (GLM owns that). You do not touch `paper/*.tex`, `report/*.tex`, or
`STATUS.md` — integration is a separate later pass.

---

## PHASE 0 — CORRECTNESS AUDIT (do this FIRST, blocks everything below)

Direct instruction from the project owner: stop adding new experiments, go back to
first principles, verify the math is actually correct — not just that experiments ran
and produced plausible numbers. **Do not resume the GPU task queue at the bottom of
this file until every box in "Phase 0 done" is checked.** GLM has a parallel Phase 0 on
the training/radii math (`docs/HANDOFF_GLM.md`) — yours is the metrics/attack side.

### Finding 2 — α=0 does not make DRO coincide with Naive (documented, verify disclosure)

`src/training/dro_fair.py` line 345: `for _ in range(self.K_inner if self.alpha > 0
else 0)` correctly stops the inner p-ascent at α=0 — but DRO still optimizes a **tilted
risk** (β·logsumexp) while Naive optimizes plain BCE mean, and DRO's dual-ascent
λ-update still runs. So at α=0 (zero corruption) the two methods are **not the same
training objective** — not expected to produce identical numbers, and every "DRO wins
6/6 at α=0.0" cell means "DRO's tilted objective happens to beat plain BCE absent any
attack," not "DRO survives zero attack."

**Check:** does any doc, the paper, or the report present α=0 results as if DRO is
"surviving" or "being robust to" zero corruption, without stating the objectives differ?
If so, fix the text. This is a disclosure check, not a code fix — the behavior is
expected, the risk is that prose overclaims what it means.

### Finding 3 — IF metric uses cosine distance, not the paper's original distance (verify disclosure)

`src/evaluation/metrics.py::compute_if_violation` docstring: the original Euclidean IF
metric was degenerate (feature distances ~0.8–2.8 dwarfed prediction differences
~0.02, so the relu saturated to ~0 everywhere) — fixed by switching to cosine distance,
which "changes the meaning of IF from an absolute feature-space threshold to an
angular-similarity threshold." **A legitimate fix for a real bug — but every IF number
in every table now measures something different from the paper's original IF formula.**

**Check:** does the paper/report explicitly disclose this metric change wherever IF
numbers appear, or are they presented as if measuring the original paper metric without
the caveat? If undisclosed, that's a scientific-honesty gap, not a code bug — flag it
for the integration pass to fix in the text.

### Q6 (Kuldeep, Jun 9) — IF attack/eval k-NN graph mismatch, re-verify current state

Kuldeep's original concern: the IF attack computed k-NN *within* protected groups while
training/eval computed it over *all* samples — an attack/eval mismatch. `metrics.py`
line ~96 now documents eval as intentionally global (not within-group). **Check
`src/corruption/adversarial.py`'s IF attack — does it use the SAME global k-NN graph as
eval now, or does it still build neighbors within-group?** If eval was fixed but the
attack wasn't, the mismatch Kuldeep flagged is still live even though it looks resolved
from the eval side alone. Report definitively either way with the exact line numbers.

### Q7 (Kuldeep, Jun 9) — IF attack decreasing DP: is this mathematically expected?

Confirmed empirically in the N4 analysis (Adult α=0.3 shows a DP-loss coupling under the
IF attack). **Write the one-paragraph mathematical argument for why an IF-targeted
attack should plausibly decrease DP** — e.g., if the attack smooths predictions near
decision-boundary samples to increase local similarity violations, does that plausibly
compress group-level prediction-rate gaps too? If you can't construct a convincing
argument, say so explicitly rather than asserting the coupling is "expected" — an
unexplained empirical correlation is a different, weaker claim than a derived one.

### Phase 0 done when

- [x] Finding 2 disclosure checked across paper/report/docs, fixed if overclaiming
- [x] Finding 3 disclosure checked across every place IF appears, fixed if undisclosed
- [x] Q6 attack/eval k-NN graph mismatch resolved or confirmed already fixed, with line numbers
- [x] Q7 mathematical argument written, or explicitly marked unexplained
- [x] Findings appended to this file (below this line), don't rewrite the sections above

**Findings go here:**

### Phase 0 results (2026-08-04, Grok lane)

#### Finding 2 — α=0 objectives (DISCLOSURE OK)
- **Code:** `dro_fair.py` skips p-ascent when `alpha > 0` is false; tilted risk + λ dual still run.
- **Paper:** `paper/sections/results.tex` lines 25–30 already state α=0 is **not** an attack cell and DRO vs Naive optimise **different** objectives (tilted / dual vs fixed-Lagrange BCE); excluded from attack-robustness headline.
- **Verdict:** No overclaim of “robustness to zero corruption.” No prose fix required. Integration may mirror the same sentence in the report if any residual α=0 “win” language remains.

#### Finding 3 — cosine IF vs original Euclidean (DISCLOSE FOR INTEGRATION)
- **Eval default:** `metrics.compute_if_violation` / `compute_metrics_torch` use `metric='cosine'` (docstring lines 70–74, 144–147).
- **Attack:** IF path uses `metric='cosine'` in `_precompute_if_neighbors` / greedy attack (`adversarial.py` 293–315, 424).
- **Paper status:** Mentions “cosine IF” / “cosine fix” in results/conclusion but does **not** always spell out: “this is **not** the original paper Euclidean IF; absolute feature-space threshold was abandoned because it was degenerate.”
- **Verdict:** **Flag for integration pass** — strengthen one explicit sentence wherever IF numbers appear (experimental setup + first IF table). Not a code bug. Numbers in tables measure angular IF, not the original Euclidean formula.

#### Q6 — IF attack k-NN within-group vs global (FIXED)
| Site | Graph | Metric | Lines |
|------|-------|--------|-------|
| Eval | **Global** (all samples) | cosine | `metrics.py` 101–104 |
| Attack precompute | **Global** (`'global'` key, `np.arange(n)`) | cosine | `adversarial.py` 293–315, 424 |
| Training (was gap) | **Global** | was **Euclidean** (sklearn default) | `dro_fair.py` / `naive_fair.py` `_build_knn_graph` |

- **Kuldeep mismatch (within-group attack vs global eval):** **already fixed** on attack+eval — both global cosine.
- **Residual found & fixed this session:** training IF graphs used default Euclidean distances while attack/eval used cosine. Updated both trainers to `metric='cosine'` so train = attack = eval.

#### Q7 — Why IF-targeted attack can lower DP (mathematical sketch)
DP measures a **group-mean** gap `|E[y|A=0] − E[y|A=1]|` (or soft rates). The IF attack maximises
`Σ_{(i,j)∈N_k} max(0, |y_i−y_j| − d(x_i,x_j) − γ)` over label flips, i.e. it creates
**local prediction disagreements among feature-neighbors**.

Many near-boundary pairs are **cross-group** in feature space (features are not perfectly
group-separable). Forcing such neighbors toward **similar** labels to create/remove IF
violations (depending on the marginal-gain sign) tends to **pull group-conditional rates
toward each other**: the attack is not maximising DP, so it has no incentive to keep
minority/majority means far apart. Empirically (Adult IF-attack α=0.3) IF worsens while
DP falls for DRO relative to Naive (or DRO loses on DP) — consistent with **coupling**, not
with a DP-maximising adversary.

This is a **plausible mechanism**, not a theorem that IF↑ always implies DP↓. We treat
DP-under-IF as a **separate** reported outcome (mixed), not as a corollary of IF robustness.

#### Phase 0 residual for integration (not Grok GPU)
- Paper/report: one explicit “cosine ≠ original Euclidean IF” sentence (Finding 3).
- Optional: re-run IF third after trainer cosine fix if advisors demand re-lock (canonical 540 was trained under old Euclidean train graph) — **do not silently retrain without greenlight**.

---

## PHASE 1 — flair2 GPU work (resume only after Phase 0 is checked off)

### flair2 is already unlocked — do not reinstall

```
torch 2.6.0+cu124
cuda_available: True
n_gpu: 2
gpu 0: NVIDIA L40S 48GB
gpu 1: NVIDIA L40S 48GB
```
`venv_gpu` exists at `/data/srujan.sai/DRO-FairML-run/venv_gpu`. Use it directly:
`./venv_gpu/bin/python ...`. **Do not download wheels again, do not recreate venv_gpu**
— that already happened twice tonight (two agents independently redid it) for nothing.
Verify it still works before starting anything new:
```bash
ssh flair2 '/data/srujan.sai/DRO-FairML-run/venv_gpu/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"'
```
Should print `True 2`. If it doesn't, stop and report — don't reinstall blind.

### Locked truth — read once, don't re-derive

- `results/utkface_canonical.json`: **90/90 REAL rows, complete, on Mac MPS** (dp/if/
  combined = 30/30/30, all 6 seeds, all 5 alphas). Done. Your GPU work is either
  (a) reproducing it on CUDA for a reproducibility row, or (b) genuinely new experiments
  a laptop can't do. **Don't re-run the same grid pointlessly** — U1 below reproduces it
  deliberately for a specific reason (see below).
- `results/canonical_tau1.json`: 540-row locked tabular grid. Not your lane, read-only.
- Central finding: fixed τ=1 makes DRO beat Naive on Adult/Credit DP+Combined at α≤0.2.
  LSAC/DP degenerate. IF MIXED. See `docs/LSAC_DEGENERACY.md`, `docs/VERIFICATION_REPORT.md`.

### Code sync — flair2 has no internet/git past the SSL firewall

Every session, before running anything:
```bash
rsync -az --exclude='.git' --exclude='data/raw' --exclude='figures' \
  --exclude='logs' --exclude='results' --exclude='venv_gpu' --exclude='paper' \
  --exclude='report' /Users/srujansai/Desktop/DRO-FairML/ \
  flair2:/data/srujan.sai/DRO-FairML-run/
```

### The 4 tasks, in priority order

Use `CUDA_VISIBLE_DEVICES=0` for one job, `=1` for another, so two can run at once
without colliding. flair2 is a **shared cluster** (other users have Jupyter kernels
running there) — check `nvidia-smi` before assuming both GPUs are free.

**U1 — Reproducibility (do this first, cheap, an advisor ask).** Run the identical
UTKFace canonical protocol (τ=1, K=10, n=6, epochs=60, pgd_steps=20, attacks dp/if/
combined, α 0.0–0.4) on CUDA instead of Mac MPS:
```bash
CUDA_VISIBLE_DEVICES=0 ./venv_gpu/bin/python experiments/run_utkface_server.py \
  --attacks dp if combined --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
  --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
  --output results/utkface_flair2.json
```
90 configs. Write `results/utkface_reproducibility_summary.md`: compare CPU/MPS vs CUDA
per (attack, alpha) — same seeds should be near-identical; large gaps are a bug to
report, not hide. Makes Manisha's May-19 "set up UTKFace in the server" ask literally true.

**U2 — Multi-group UTKFace (5-race, not the binary collapse).** UTKFace race is
currently binarized to White/non-White. `src/evaluation/metrics.py` already computes
max-min DP for >2 groups — wire the raw 5-category race label through instead. 60
configs (5 α × 6 seeds × 2 methods, dp) → `results/utkface_multigroup.json` +
`results/utkface_multigroup_summary.md`: does DRO's advantage hold across all 5 groups,
or concentrate/vanish for the smallest (Indian/Other)? Likely the most novel result
available for the paper.

**U3 — Pixel-space PGD (the genuine GPU-only stretch).** Everything so far attacks
cached ResNet18 features. This attacks raw pixels:
```bash
git show 3c371a8:src/corruption/image_pgd.py > src/corruption/image_pgd.py
```
(fixed version; check `git log --oneline --all -- src/corruption/image_pgd.py` for the
matching CNN scaffolding from commit a31d43f if needed). Scope α∈{0.1,0.2}, 6 seeds, dp,
2 methods, raw UTKFace JPEGs (already on flair2) = 24 configs →
`results/utkface_pixel_pgd.json` + `results/pixel_pgd_summary.md` comparing
feature-space vs pixel-space attack strength. Timebox ~half a day; report partial
honestly rather than block everything else.

**U4 — CelebA second modality (stretch, only if U1-U3 done with time to spare).** Cut
without guilt if short on time — U1-U3 alone are a strong result.

### Rules

- Full provenance on every row (tau, k_inner, epochs, pgd_steps, seed, n_seeds_planned,
  device='cuda', data_provenance='REAL' — never let synthetic UTKFace features get
  reported as real, the loader already rejects them, keep it that way).
- Never write `results/canonical_tau1.json` or the original `results/utkface_canonical.json`
  — always a new file.
- Rsync results back to the Mac and commit from there (flair2 has no git/PyPI access,
  it is not the source-of-truth repo, don't push from it).
- Do not edit `paper/*.tex` / `report/*.tex` — leave summaries ready for the integration pass.

### When done

Append status to `STATUS.md`'s "what's left" section — don't rewrite the whole file,
and don't attempt final paper integration yourself.
