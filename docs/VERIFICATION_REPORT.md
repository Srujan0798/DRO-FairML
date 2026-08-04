# Independent Verification Report (Agent L)

> **CURRENT (post-540):** Canonical grid is **540/540 complete** (dp=180, combined=180, if=180).  
> IF story is **MIXED** — not a clean three-attack sweep. See **§ Post-540 re-audit** at the bottom  
> and `results/if_wilcoxon_summary.txt` + `docs/MEETING_2026-08-04.md`.  
> Adult/DP α=0.1 is **5/6**. LSAC/DP is **degenerate**.  
> Sections **§0–§9** below are a **historical mid-sweep snapshot** (424 rows / IF 64) kept for audit trail — **do not cite as live counts**.

**Verifier:** Agent L (adversarial / independent)  
**Historical snapshot (UTC):** 2026-08-04T08:10:15Z (mid IF sweep)  
**Canonical file:** `results/canonical_tau1.json`  
**Method:** Recompute means, seed win-counts, and one-sided Wilcoxon signed-rank (`scipy.stats.wilcoxon`, `alternative='greater'` on `naive − dro` for DP) from raw rows. Claims assumed false until proven.

---

## 0. Row counts at verification time *(historical — superseded by 540/540)*

| Attack | Rows | Expected (full grid) | Status (at snapshot) |
|--------|------|----------------------|----------------------|
| `dp` | **180** | 180 (3×5×6×2) | **COMPLETE** |
| `combined` | **180** | 180 | **COMPLETE** |
| `if` | **64** | 180 | ~~PARTIAL~~ → **now 180/180 COMPLETE** |
| **Total** | **424** | **540** | ~~Growing~~ → **540 COMPLETE** |

**IF-attack completeness matrix at snapshot** (`n_naive` / `n_dro`, target 6/6 each cell) — historical:

| Dataset | α=0.0 | α=0.1 | α=0.2 | α=0.3 | α=0.4 |
|---------|-------|-------|-------|-------|-------|
| adult | 6/6 | 6/6 | 6/6 | 6/3 | 6/0 |
| credit | 6/6 | 1/0 | 0/0 | 0/0 | 0/0 |
| lsac | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

**IF readiness (live):** complete 180/180; ship claims must be **honest mixed** (Adult/Credit α≤0.2 OK on DP under IF; LSAC/IF and Adult IF α≥0.3 not a DP sweep).

---

## 1. Provenance uniformity (all 424 rows)

| Field | Observed unique values | Required | Match? |
|-------|------------------------|----------|--------|
| `tau` | `{1.0}` | 1.0 | **YES** |
| `k_inner` | `{10}` | 10 | **YES** |
| `epochs` | `{60}` | 60 | **YES** |
| `pgd_steps` | `{20}` | 20 | **YES** |
| `lambda_init` | `{0.0}` | 0.0 | **YES** |
| `radii_mode` | `{uniform}` | uniform | **YES** |
| `coordinated` | `{False}` | False | **YES** |
| seeds | `{0,1,2,3,4,5}` | 0–5 | **YES** |

### IF non-degeneracy (attack == `if` only)

| Check | Value |
|-------|-------|
| IF-attack rows | 64 |
| `max \|if_clean\|` | **0.0978** |
| `min \|if_clean\|` | **0.0178** |
| Rows with `\|if_clean\| > 1e-6` | **64 / 64** |
| Non-IF attacks `max \|if_clean\|` | **4.66e-10** (still ~machine zero under DP/Combined attacks) |

**Conclusion:** New IF-attack rows are **non-degenerate**. STATUS.md / KULDEEP_CORRECTION.md statements that “IF cells currently read 0.0000” and “IF never generated” are **stale** relative to the live file (they were true of DP/Combined IF *columns* and pre-sweep state).

---

## 2. Recomputed DP cell table (source of truth)

One-sided Wilcoxon: \(H_1\): Naive DP > DRO DP. Exact \(p\) for 6/6 same-sign wins is **0.015625** (docs often round to **0.016**).

### 2.1 Adult / DP

| α | Naive DP | DRO DP | Wins | p | Naive Acc | DRO Acc |
|---|----------|--------|------|---|-----------|---------|
| 0.0 | 0.1491 | 0.1426 | **6/6** | 0.0156 | 0.8135 | 0.8147 |
| 0.1 | 0.2026 | 0.1999 | **5/6** | 0.0312 | 0.8163 | 0.8177 |
| 0.2 | 0.2452 | 0.2334 | **6/6** | 0.0156 | 0.7557 | 0.7586 |
| 0.3 | 0.2848 | 0.2614 | **6/6** | 0.0156 | 0.6669 | 0.6755 |
| 0.4 | 0.3140 | 0.2855 | **6/6** | 0.0156 | 0.5512 | 0.5607 |

Loser seed at α=0.1: **seed 2** (Naive 0.192526 vs DRO 0.192608).

### 2.2 Adult / Combined

| α | Naive DP | DRO DP | Wins | p |
|---|----------|--------|------|---|
| 0.0 | 0.1491 | 0.1426 | 6/6 | 0.0156 |
| 0.1 | 0.1509 | 0.1432 | 6/6 | 0.0156 |
| 0.2 | 0.1963 | 0.1784 | 6/6 | 0.0156 |
| 0.3 | 0.2176 | 0.1922 | 6/6 | 0.0156 |
| 0.4 | 0.2110 | 0.1815 | 6/6 | 0.0156 |

### 2.3 Credit / DP

All α: **6/6**, p=0.0156. Means match auto tables (e.g. α=0.2 Naive 0.0198 / DRO 0.0178).

### 2.4 Credit / Combined

| α | Wins | p | Notes |
|---|------|---|-------|
| 0.0 | 6/6 | 0.0156 | |
| 0.1 | **5/6** | 0.0312 | seed 3 loses |
| 0.2–0.4 | 6/6 | 0.0156 | |

### 2.5 LSAC / DP (degenerate loss)

| α | Naive DP | DRO DP | Wins | p | Acc (Naive/DRO) |
|---|----------|--------|------|---|-----------------|
| 0.0 | 0.1447 | 0.1829 | 0/6 | 1.000 | 0.9015 / 0.9023 |
| 0.1 | 0.2201 | 0.2539 | 0/6 | 1.000 | 0.9035 / 0.9046 |
| 0.2 | 0.1827 | 0.2230 | 0/6 | 1.000 | 0.9022 / 0.9033 |
| 0.3 | 0.1827 | 0.2220 | 0/6 | 1.000 | 0.9022 / 0.9032 |
| 0.4 | 0.1827 | 0.2211 | 0/6 | 1.000 | 0.9022 / 0.9029 |

Per-seed Naive DP is **bit-identical across α∈{0.2,0.3,0.4}** (metric frozen); mean 0.182709… ≈ **0.1827**. Accuracy pinned near majority baseline **0.9016**.

### 2.6 LSAC / Combined

| α | Naive DP | DRO DP | Wins | p |
|---|----------|--------|------|---|
| 0.0 | 0.1447 | 0.1829 | 0/6 | 1.000 |
| 0.1 | 0.1437 | 0.1361 | 6/6 | 0.0156 |
| 0.2 | 0.1321 | 0.1248 | 5/6 | 0.0312 |
| 0.3 | 0.1874 | 0.1602 | 6/6 | 0.0156 |
| 0.4 | 0.2151 | 0.1830 | 6/6 | 0.0156 |

### 2.7 IF-attack (PARTIAL — DP metric only; IF metric claims PENDING)

| Cell | n | DP wins | p (DP) | mean if_clean (N/D) | Status |
|------|---|---------|--------|---------------------|--------|
| adult/if α=0.0 | 6/6 | 6/6 | 0.0156 | 0.0942 / 0.0933 | provisional |
| adult/if α=0.1 | 6/6 | 6/6 | 0.0156 | 0.0407 / 0.0328 | provisional |
| adult/if α=0.2 | 6/6 | 6/6 | 0.0156 | 0.0276 / 0.0222 | provisional |
| adult/if α=0.3 | 6/3 | incomplete | — | 0.0334 / 0.0248 | **PENDING** |
| adult/if α=0.4 | 6/0 | incomplete | — | 0.0326 / — | **PENDING** |
| credit/if α=0.0 | 6/6 | 6/6 | 0.0156 | 0.0236 / 0.0234 | provisional |
| credit/if α≥0.1 | incomplete | — | — | — | **PENDING** |
| lsac/if * | 0 | — | — | — | **PENDING** |

---

## 3. Claim-by-claim verification table

Legend: **MATCH** = recomputed agrees · **MISMATCH** = disagrees with data · **STALE** = was true of older state · **PENDING** = IF incomplete · **ROUNDED** = exact p=0.015625 printed as 0.016 · **OK-SCOPE** = narrative correct with noted scope caveat.

| # | Claim | Source file | Recomputed | Match? |
|---|-------|-------------|------------|--------|
| 1 | Canonical DP+Combined = 360 rows | STATUS.md §2, README | 180+180=360 | **MATCH** |
| 2 | tau=1, k_inner=10, epochs=60, seeds 0–5 | STATUS.md §2 | all rows | **MATCH** |
| 3 | Adult/DP: wins every α **(6/6)**, p≤0.031 | STATUS.md §3 | α=0.1 is **5/6** (p=0.0312); others 6/6 | **MISMATCH** (win count) |
| 4 | Adult/Combined: 6/6 every α, p=0.016 | STATUS.md §3 | 6/6 all α, p=0.015625 | **MATCH** (ROUNDED) |
| 5 | Credit DP+Combined: wins essentially every cell, p<0.05 | STATUS.md §3 | 10/10 cells p<0.05; Combined α=0.1 is 5/6 | **MATCH** (with 5/6 nuance) |
| 6 | LSAC/Combined genuine win, p=0.016 at α=0.1/0.3/0.4 | STATUS.md §3 | p=0.0156 at those α; α=0.2 p=0.031 | **MATCH** (ROUNDED) |
| 7 | LSAC/DP: 0/6 every α; Naive DP frozen 0.1827 for α≥0.2; acc ~0.90 | STATUS.md, KULDEEP_CORRECTION | confirmed; per-seed identity across α≥0.2 | **MATCH** |
| 8 | Defensible regime α≤0.2; α≥0.3 below constant predictor (Adult 0.752, Credit 0.779, **LSAC 0.902**) | STATUS.md §3 | Adult/Credit yes below; **LSAC DP stays ≥0.9016 (pinned, not below)** | **MISMATCH** (LSAC “below”) |
| 9 | IF-attack third never generated; IF cells 0.0000 | STATUS.md §3,§6,§8 | 64 IF rows live; non-deg if_clean; DP/Combined IF column still ~0 | **STALE** |
| 10 | PoC if_clean=0.0333 | STATUS.md §3 | not re-run here; separate file | not checked against canonical |
| 11 | Adult/DP α=0.0: N 0.1491 / D 0.1426 (p=0.016) | KULDEEP_CORRECTION §solid | exact; p=0.0156 | **MATCH** (ROUNDED) |
| 12 | Adult/DP α=0.1: N 0.2026 / D 0.1999 (p=0.031) | KULDEEP_CORRECTION | exact | **MATCH** |
| 13 | Adult/DP α=0.2: N 0.2452 / D 0.2334 (p=0.016) | KULDEEP_CORRECTION | exact | **MATCH** |
| 14 | **“All six α are 6/6 wins”** (Adult/DP) | KULDEEP_CORRECTION §solid | α=0.1 is **5/6** | **MISMATCH** |
| 15 | Adult/Combined 6/6 every α, p=0.016 | KULDEEP_CORRECTION | yes | **MATCH** (ROUNDED) |
| 16 | Credit: DRO beats Naive every cell, all p<0.05 (DP+Combined) | KULDEEP_CORRECTION | yes | **MATCH** |
| 17 | LSAC/DP table (exact 4-decimal means, 0/6, p=1.0) | KULDEEP_CORRECTION §2 | all 5 rows match | **MATCH** |
| 18 | LSAC/Combined p=0.0156 at 0.1/0.3/0.4 and p=0.031 at 0.2 | KULDEEP_CORRECTION §2 | exact | **MATCH** |
| 19 | max \|IF\| across 360 DP+Combined rows = 4.66e-10 | KULDEEP_CORRECTION §1 | max non-if = 4.66279e-10 | **MATCH** |
| 20 | Adult baseline 0.7521; Naive acc α=0.3≈**0.676**, α=0.4≈**0.608** | KULDEEP_CORRECTION §3 | Naive α=0.3=**0.6669**, α=0.4=**0.5512**; 0.676 is **DRO** α=0.3; 0.608 is **wrong** | **MISMATCH** |
| 21 | Credit baseline 0.7788; Naive α=0.3≈0.757, α=0.4≈0.744 | KULDEEP_CORRECTION §3 | 0.7527 / 0.7513 | **MISMATCH** (approx off; direction OK) |
| 22 | Baselines 0.7521 / 0.7788 / 0.9016 in loaders | KULDEEP_CORRECTION | `_CONSTANT_PREDICTOR_FALLBACK` matches | **MATCH** |
| 23 | README: α≤0.2 DRO lower DP on Adult+Credit **all three attacks** p<0.05 n=6 | README.md | DP+Combined: yes; IF Credit incomplete | **PENDING** / overclaim until IF done |
| 24 | README: 360 rows DP+Combined; IF pending | README Project Flow | yes | **MATCH** (counts outdated only for IF progress) |
| 25 | paper/results.tex: Adult+Credit DP+Combined α≤0.2 p<0.05 n=6 | paper/sections/results.tex | yes | **MATCH** |
| 26 | paper/results.tex: α≥0.3 below constant predictor **on every dataset** | results.tex L10–12 | LSAC DP not below | **MISMATCH** |
| 27 | paper τ-comparison τ=1 values (0.2068/0.2046 … wins 2/3,3/3) | results.tex tab:tau-comparison | equals **seeds 0–2 only**, not n=6 means (0.2026/0.1999, 5/6) | **STALE / n=3** (honest if labeled seeds 0–2; misleading vs “6-seed grid”) |
| 28 | Stepped τ=100 Adult α=0.2 Naive 0.327 / DRO 0.503 | results.tex, report.tex | not in canonical_tau1 (historical τ-schedule) | **out of scope** of this JSON (not contradicted by it) |
| 29 | LSAC Combined p=0.0156 at α=0.1,0.3,0.4 | results.tex | yes | **MATCH** |
| 30 | key_findings.tex: “all three attacks” at α≤0.2 | paper/auto_generated/key_findings.tex | IF incomplete | **PENDING** / overclaim |
| 31 | auto tabular_results.tex all DP means (3 d.p.) | paper/auto_generated/tabular_results.tex | 0/30 mismatches | **MATCH** |
| 32 | auto wilcoxon.tex ΔDP% + p | paper/auto_generated/wilcoxon.tex | 0 mismatches (0.016 = round 0.015625) | **MATCH** |
| 33 | report auto_generated_main_results.tex means±SE (DP attack) | report/sections/… | SE and means match | **MATCH** |
| 34 | report auto_generated_pgd.tex full DP+Combined | report/sections/… | matches §2 | **MATCH** |
| 35 | report auto_generated_wilcoxon (DP attack only) | report/sections/… | matches Adult/Credit/Lsac DP cells | **MATCH** |
| 36 | IF columns printed as 0.0000 under DP/Combined tables | auto tables | correct for those attacks | **MATCH** (not IF-attack rows) |
| 37 | make validate: DP wins 6/9 at p<0.05 | STATUS.md §8 | validate PASS 6/9 (uses all_results.json DP-like grid α∈{0.1,0.2,0.3}) | **MATCH** (tool PASS) |
| 38 | FINAL_COMPLETION_PLAN “Adult/DP wins every α” | docs/FINAL_COMPLETION_PLAN.md | 5/6 at α=0.1 | **MISMATCH** |
| 39 | KULDEEP_DISCUSSION Adult DP table 5/6 at α=0.1 | KULDEEP_DISCUSSION.md | correct | **MATCH** (better than STATUS) |

---

## 4. Hardcoded / stale number hunt

Scope: active `.md`, `.tex`, figure-generator `.py` (not `docs/_archive/`, not `results/stale_archived/` as primary claims).

| Constant / pattern | Where | Verdict |
|--------------------|-------|---------|
| **0.752 / 0.7521** | loaders fallback; STATUS/KULDEEP baselines; figure labels in `generate_all_deliverables.py` (string “0.752” even when `CONSTANT_PREDICTOR_ACC` is dynamic); report.tex “acc≈0.752” | Adult majority rate — **correct for Adult**. Caption strings still hardcode “0.752” while code uses `constant_predictor_acc('adult')`. |
| **0.0195 / 0.0177** | KULDEEP_CORRECTION (as withdrawn Jun-30 mislabel); MASTER_DISPATCH history | Documented as **false IF claim** (was DP). Not presented as current result. OK if kept as correction narrative. |
| **tau=100** | results.tex comparison, report.tex, plot_tau1_headline.py, tests | Historical artifact contrast — **not** canonical config. Canonical has only tau=1. |
| **n=3 / 2/3 / 3/3** | `paper/sections/results.tex` τ-table; `docs/TAU1_ABLATION_SUMMARY.md`; report.tex τ=1 row 0.2480/0.2371 | **n=3 (seeds 0–2)** numbers, not n=6. Conflicts with “6-seed grid” framing unless explicitly labeled. |
| **0.0000 IF** | STATUS, auto tables for DP/Combined, validate IF column | True for **DP/Combined** IF metric column; **false** as “no IF experiment” once IF-attack rows exist. |
| **0.327 / 0.503** | results.tex, FINDING_DRO_FAILS_ON_ADULT, etc. | τ=100 historical — not from canonical_tau1. |
| **0.2068 / 0.2046 / 0.2480 / 0.2371** | paper results.tex, TAU1_ABLATION_SUMMARY | seeds 0–2 subset of canonical Adult DP. |
| **n_seeds 3** | README run example CLI | Example only, not result claim. |
| **tau=100 in tests** | `tests/test_end_to_end.py` | Intentional regression test of old temperature path — OK. |
| Figure generators | `CONSTANT_PREDICTOR_ACC = constant_predictor_acc('adult')` in plot_high_alpha_tau, generate_final_figures, plot_lambda_heatmap_highalpha, generate_all_deliverables | Code fixed to helper; **plot title strings** still say “0.752”. Adult-only plots → acceptable if documented. |

---

## 5. Tests and validate

### pytest

```text
$ python3 -m pytest tests/ -q
62 passed, 1 warning in 4.76s
```

- Warning: unknown mark `pytest.mark.slow` in `tests/conftest.py`  
- **Status: PASS (62/62)**

### make validate

```text
$ make validate
Total: 90 experiments
WARNING: Expected 150, got 90
...
DP WINS (Wilcoxon p<0.05):  6/9  (need >= 6/9)
IF WINS (Wilcoxon p<0.05):  0/9
RESULT: PASS
```

**Caveats (tool quality, not data falsification):**
1. Reads **`results/all_results.json` (90 rows)**, not `canonical_tau1.json` (424).
2. Header says “n=10 paired seeds” but data are not 10-seed.
3. IF path still sees 0.0000 (all_results structure / DP-attack-derived merge).
4. Warning Expected 150 is stale vs 540-row full grid.

Validate **PASSes** its own criterion; it is **not** a full canonical audit.

---

## 6. Unresolved mismatches (must be empty for Aug 10 ship)

| ID | Severity | Issue | Fix required |
|----|----------|-------|--------------|
| M1 | **HIGH** | STATUS.md / KULDEEP_CORRECTION / FINAL_COMPLETION_PLAN claim Adult/DP **6/6 at every α**; data: **α=0.1 is 5/6** | Change prose to “5/6 at α=0.1 (p=0.031), 6/6 otherwise” (as KULDEEP_DISCUSSION already has) |
| M2 | **HIGH** | KULDEEP_CORRECTION Adult Naive acc α=0.3≈0.676, α=0.4≈0.608 | Replace with **0.6669 / 0.5512** (Naive); do not quote DRO as Naive |
| M3 | **MED** | Credit high-α Naive acc ≈0.757 / 0.744 claimed | Use **0.7527 / 0.7513** |
| M4 | **MED** | STATUS/paper: α≥0.3 “below constant predictor on **every** dataset” including LSAC | Scope to Adult+Credit; LSAC is **pinned at** majority, not below |
| M5 | **MED** | paper `results.tex` τ=1 column is **n=3** while surrounding text says 6-seed canonical | Either recompute table at n=6 or label “seeds 0–2 pilot / ablation” |
| M6 | **MED** | README + key_findings “**all three attacks**” while IF grid incomplete | Restrict to DP+Combined until IF=180; or mark IF PENDING |
| M7 | **MED** | STATUS still says IF never generated / IF=0.0000 | Update to live IF row count + non-degeneracy; keep “no IF *claim* until complete” |
| M8 | **LOW** | make validate not reading canonical; Expected 150 | Point validate at canonical_tau1 + correct expected count |
| M9 | **LOW** | Figure caption strings hardcode “0.752” | Cosmetic; values computed from helper for Adult |

**Ship gate (Aug 10):** M1–M7 must be closed in prose/tables. M8–M9 tooling polish. IF section remains **PENDING** until 180 rows — do not block ship of DP+Combined claims if scoped correctly, but “all three attacks” must not ship until IF complete.

**Unresolved mismatch count for ship-critical prose: 7 (M1–M7).** Not empty.

---

## 7. Verified solid core (safe to lead with)

After adversarial check, these hold on complete DP+Combined (360 rows):

1. **Adult DP:** significant DRO win every α (p≤0.031); wins 6/6 except **α=0.1 (5/6)**.
2. **Adult Combined:** 6/6, p=0.0156 every α.
3. **Credit DP+Combined:** every cell p<0.05 (one cell 5/6).
4. **LSAC DP:** total loss 0/6, degenerate (frozen Naive DP, acc pinned).
5. **LSAC Combined:** significant at α∈{0.1,0.2,0.3,0.4} (not α=0).
6. **Defensible accuracy regime α≤0.2** on Adult and Credit vs constant predictor; α≥0.3 both methods below baseline on Adult/Credit.
7. **Auto-generated LaTeX tables** from canonical (means, SE, Wilcoxon Δ%, p) **match recomputation** for DP+Combined.
8. **Provenance** uniform (τ=1, K=10, epochs=60) on all current rows.
9. **pytest 62 passed**; **make validate PASS** (with caveats in §5).

---

## 8. IF section (template for later completion)

_When IF reaches 180 rows, re-run the §2.7 matrix and fill:_

- [ ] All 30 cells (3 ds × 5 α × 2 methods) have 6 seeds  
- [ ] Wilcoxon on **if_clean** (and dp_clean) per cell  
- [ ] Regenerate auto tables including IF-attack rows  
- [ ] Withdraw any remaining “IF=0.0000” STATUS language  
- [ ] Re-verify README “all three attacks” claim  

**At verification time:** 64/180 · non-degenerate · Adult α≤0.2 provisional DP wins only · **no ship claim on IF metric**.

---

## 9. Summary counts

| Category | Count |
|----------|-------|
| Claims traced | **39** (primary table) + full cell recomputes |
| **MATCH** (incl. ROUNDED) | **27** |
| **MISMATCH** | **6** (M1–M5 + related STATUS LSAC) |
| **STALE / PENDING** | **6** |
| Ship-critical unresolved | **7 (M1–M7)** |
| pytest | **62 passed** |
| make validate | **PASS** (canonical-incomplete tooling) |
| IF readiness | **64/180 — PENDING** |

**Bottom line:** Auto-generated DP+Combined tables are honest. Several human-written STATUS / KULDEEP_CORRECTION / paper sentences still overstate win counts (6/6 vs 5/6), mis-quote high-α accuracy, mis-scope LSAC below-baseline, and lag the live IF sweep. **Unresolved mismatches are not empty — block Aug 10 ship until M1–M7 prose is fixed; IF remains PENDING separately.**

---

## Corrections applied 2026-08-04

**Agent:** correction agent (post–Agent L).  
**Basis:** recompute from `results/canonical_tau1.json` (424 rows at apply time: DP 180, Combined 180, IF 64). Scientific JSON rows **not** modified.

| ID | Action |
|----|--------|
| **M1** | STATUS.md, `docs/KULDEEP_CORRECTION.md`, `docs/FINAL_COMPLETION_PLAN.md`: Adult/DP is no longer “6/6 every α”; now “wins every α at p≤0.031 (**α=0.1 is 5/6**; others **6/6**)”. README + `paper/sections/results.tex` / `key_findings.tex` aligned. MEETING already correct. |
| **M2** | KULDEEP_CORRECTION Adult high-α **Naive** acc → **0.6669 / 0.5512** (was ≈0.676 / 0.608; 0.676 was DRO α=0.3). |
| **M3** | KULDEEP_CORRECTION Credit high-α Naive acc → **0.7527 / 0.7513** (was ≈0.757 / 0.744). |
| **M4** | STATUS, FINAL_COMPLETION_PLAN, paper `results.tex`, KULDEEP_CORRECTION: α≥0.3 “below constant predictor” scoped to **Adult+Credit**; LSAC DP **pinned at** majority (~0.902), not below. |
| **M5** | `paper/sections/results.tex` tab:tau-comparison **τ=1** rows regenerated to n=6 canonical means (0.2026/0.1999 5/6; 0.2452/0.2334 6/6; 0.2848/0.2614 6/6; 0.3140/0.2855 6/6). Caption: τ=1 = 6 seeds; τ=10/100 remain historical n=3 pilot. |
| **M6** | README + `paper/auto_generated/key_findings.tex` + paper results/conclusion + FINAL_COMPLETION_PLAN: drop ship claim of “**all three attacks**”; lead with **DP + Combined**; IF only when 180 complete. |
| **M7** | STATUS §3 IF: live **64/180**, non-degenerate max\|if\|≈**0.098**; forbid “IF never generated / IF=0.0000” for IF-attack third. KULDEEP_CORRECTION §1 updated (local sweep in progress, no full IF claim). Paper IF language: incomplete local re-run, not “cluster never generated”. |

**Left unresolved (not closed here):**
- **M8** — `make validate` still reads `all_results.json` / Expected 150 (tooling).
- **M9** — figure caption strings hardcode “0.752” (cosmetic).
- **IF scientific ship** — still **PENDING** until 180/180; no full IF story in prose claims.
- Secondary docs outside the ship-critical set (`docs/FAIRNESS_PGD_RESULTS.md`, `docs/FINDING_DRO_FAILS_ON_ADULT.md`, `docs/TAU1_ABLATION_SUMMARY.md`, `report/report.tex` n=3 mini-table if still present, `KULDEEP_DISCUSSION.md` “all three attacks” history) were not fully scrubbed.
- `paper/auto_generated/key_findings.tex` is hand-patched; regenerate path may overwrite — keep generator honest if re-run before IF=180.

**Ship-critical prose M1–M7:** closed in the files listed above. Re-audit after IF hits 540.

---

## Post-540 re-audit (Agent L / completion loop) — 2026-08-04 ~14:17 IST

**Canonical:** `results/canonical_tau1.json` — **540 rows**, unique keys **540**, attacks dp=180 / combined=180 / if=180.  
**H artifacts:** `results/if_wilcoxon_summary.txt`, `results/canonical_wilcoxon.csv` (re-run with `PYTHONPATH=.`), `paper/main.pdf`, `report/report.pdf`.  
**Provenance (all 540):** tau=1.0, k_inner=10, epochs=60, pgd_steps=20, lambda_init=0.0, radii_mode=uniform, coordinated=False.  
**IF non-deg:** max |if_clean| on attack=if = **0.2389**; 180/180 ≫ 1e-6.

### Adult/DP α=0.1 (honesty lock)

Wins **5/6**, p=0.0312; loser **seed 2** (Naive 0.192526 vs DRO 0.192608). **MATCH** STATUS / MEETING.

### IF-attack DP Wilcoxon (recomputed; matches summary)

| Dataset | α≤0.2 DP story | Notes |
|---------|----------------|-------|
| adult | **6/6 every α**, p=0.0156 | solid |
| credit | α=0.0 6/6; **α=0.1 4/6 n.s.**; α=0.2 5/6 | mixed at 0.1 |
| lsac | **0/6** α≤0.3 | DP loss under IF attack |

### IF-attack IF-metric Wilcoxon (recomputed)

| Dataset | Read |
|---------|------|
| adult | α=0.0 4/6 n.s.; α∈{0.1–0.4} **6/6** p=0.0156 |
| credit | α=0.0 4/6 n.s.; α≥0.1 **6/6** |
| lsac | α≤0.2 loss/n.s. on IF; α≥0.3 **6/6** IF win while DP still fails |

### Ship judgment (honest)

| Claim class | Gate |
|-------------|------|
| Adult+Credit, α≤0.2, DP+Combined | **PASS** (n=6, p&lt;0.05; Adult DP α=0.1 = 5/6) |
| Adult IF-attack α≤0.2 on DP (+ IF metric at 0.1/0.2) | **PASS** |
| “All three attacks, all datasets” | **FAIL** — LSAC/IF DP loss; Adult IF α≥0.3 DP fails |
| LSAC/DP | **FAIL** (degenerate; document, do not sell) |
| UTKFace | **NOT READY** — REAL-only partial grid (`results/utkface_canonical.json`); **no paper claim** |

**M6 residual:** prose must stay “DP+Combined + selective IF”, not blanket three-attack.  
**M8 residual:** largely closed — `make validate` uses `load_canonical_tau1()` (540 rows).  

**Loop note (2026-08-04 ~09:10 UTC):** UTKFace REAL rows advanced (~23/90, attack=`dp` only). Still no paper claim.
