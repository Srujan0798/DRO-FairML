# IF Attack Partial Analysis (Agent I meeting brief)

**Source:** `results/canonical_tau1.json` (re-read fresh at analysis time)  
**Snapshot:** 2026-08-04 ~13:39 local · **64 IF rows** of 420 total rows in file  
**Scope:** `attack == "if"` only; paired naive vs dro by `(dataset, alpha, seed)`  
**Honesty policy:** report losses and incomplete cells plainly. No spin.

Sweep is **still running**. Counts will change; treat this as a mid-sweep brief.

---

## 0. Executive headline (by dataset)

| Dataset | Status | IF-metric (lower better) | DP-metric under IF attack | Notes |
|---------|--------|--------------------------|---------------------------|--------|
| **adult** | **Mixed → mostly win** | Strong wins at α=0.1, 0.2 (6–0); weak/mixed at α=0.0 (4–2, n.s.); α=0.3 partial IF 3–0 but **DP metric 0–3 LOSS** | Complete cells: DRO wins DP 6–0 at α=0.0/0.1/0.2 | α=0.4 DRO missing; α=0.3 incomplete |
| **credit** | **Mixed (thin)** | α=0.0 only complete: IF 4–2, **not significant** | α=0.0: DRO DP 6–0, p=0.016 | α≥0.1 barely started; no LSAC |
| **lsac** | **Missing** | — | — | 0 IF rows |

**One-liner for the room:** Under IF attack so far, DRO reliably reduces **IF violation** on Adult at moderate α (0.1–0.2) with Wilcoxon p&lt;0.05; at α=0 the IF edge is soft (not significant). Credit α=0 looks like a small DP win and a **non-significant** IF edge. Adult α=0.3 (3 seeds only) shows an early **red flag: DRO worse on DP** while still better on IF — do not claim a clean sweep until seeds finish.

---

## 1. Coverage inventory (IF rows only)

### 1.1 Seed counts by cell

| dataset | α | naive seeds | dro seeds | paired n | enough for Wilcoxon (n≥6)? |
|---------|---|-------------|-----------|----------|----------------------------|
| adult | 0.0 | 6 (0–5) | 6 (0–5) | **6** | **yes** |
| adult | 0.1 | 6 | 6 | **6** | **yes** |
| adult | 0.2 | 6 | 6 | **6** | **yes** |
| adult | 0.3 | 6 | **3** (0–2) | **3** | **incomplete** |
| adult | 0.4 | 6 | **0** | **0** | no DRO yet |
| credit | 0.0 | 6 | 6 | **6** | **yes** |
| credit | 0.1 | **1** (seed 0) | **0** | **0** | no |
| credit | 0.2–0.4 | 0 | 0 | 0 | missing |
| lsac | 0.0–0.4 | 0 | 0 | 0 | **all missing** |

**IF rows analyzed: 64.**  
Expected full IF grid (if matching DP: 3 datasets × 5 α × 2 methods × 6 seeds = **180** rows) → **~36% done**.

### 1.2 Still missing (relative to DP-complete grid)

- **lsac:** entire IF arm (all α, both methods)  
- **credit:** α∈{0.1,0.2,0.3,0.4} almost entirely (only naive α=0.1 seed 0 landed)  
- **adult:** dro α=0.3 seeds 3–5; dro α=0.4 all seeds  

---

## 2. IF non-degeneracy check

| attack | n | max \|if_clean\| | min \|if_clean\| | mean \|if_clean\| | # near-zero (&lt;1e-12) |
|--------|---|------------------|------------------|-------------------|------------------------|
| **if** | 64 | **0.0978** | **0.0178** | **0.0415** (approx) | **0** |
| dp (reference) | 180 | ~4.7e-10 | ~0 | ~2.4e-11 | essentially all |

**Conclusion:** IF attack is **non-degenerate**. Under DP attack, `if_clean` is numerical noise (~1e-10); under IF attack it is O(10⁻²)–O(10⁻¹). The IF story **cannot** be read off DP-attack rows.

---

## 3. Paired means (IF attack): naive vs DRO

Means ± sample std over available seeds (not only paired, but for complete cells they coincide).  
Lower is better for `dp_clean` and `if_clean`; higher is better for `acc_clean`.

### 3.1 Adult

| α | method | n | mean dp_clean | mean if_clean | mean acc_clean |
|---|--------|---|---------------|---------------|----------------|
| 0.0 | naive | 6 | 0.1491 ± 0.0055 | 0.0942 ± 0.0036 | 0.8135 ± 0.0031 |
| 0.0 | dro | 6 | **0.1426 ± 0.0048** | **0.0933 ± 0.0023** | 0.8147 ± 0.0038 |
| 0.1 | naive | 6 | 0.0819 ± 0.0040 | 0.0407 ± 0.0023 | 0.8126 ± 0.0046 |
| 0.1 | dro | 6 | **0.0770 ± 0.0065** | **0.0328 ± 0.0017** | 0.8137 ± 0.0049 |
| 0.2 | naive | 6 | 0.0474 ± 0.0063 | 0.0276 ± 0.0038 | 0.7809 ± 0.0077 |
| 0.2 | dro | 6 | **0.0455 ± 0.0073** | **0.0222 ± 0.0033** | 0.7837 ± 0.0078 |
| 0.3 | naive | 6 | 0.0227 ± 0.0082 | 0.0334 ± 0.0026 | 0.7158 ± 0.0120 |
| 0.3 | dro | 3 | 0.0257 ± 0.0112 ⚠️ | **0.0248 ± 0.0030** | 0.7369 ± 0.0152 |
| 0.4 | naive | 6 | 0.0042 ± 0.0011 | 0.0326 ± 0.0040 | 0.6475 ± 0.0094 |
| 0.4 | dro | 0 | — | — | — |

⚠️ Adult α=0.3: mean DP is **higher** for DRO on the 3 seeds present (see losses below).

### 3.2 Credit

| α | method | n | mean dp_clean | mean if_clean | mean acc_clean |
|---|--------|---|---------------|---------------|----------------|
| 0.0 | naive | 6 | 0.0127 ± 0.0039 | 0.0236 ± 0.0051 | 0.8054 ± 0.0044 |
| 0.0 | dro | 6 | **0.0119 ± 0.0038** | **0.0234 ± 0.0050** | 0.8068 ± 0.0039 |
| 0.1 | naive | 1 | 0.0018 | 0.0200 | 0.8022 |
| 0.1 | dro | 0 | — | — | — |

### 3.3 LSAC

No IF rows yet.

---

## 4. Win counts & Wilcoxon (paired seeds only)

**Win definition:** DRO better = **lower** metric (for DP and IF separately).  
**Wilcoxon:** one-sided `alternative="less"` on paired diffs `(dro − naive)` when n≥6; two-sided also reported.  
With n=6 and no ties, one-sided min p = 0.015625 (all ranks same direction).

### 4.1 Summary table

| dataset | α | n_paired | DP W–L–T | mean Δdp (dro−naive) | p_DP (less) | IF W–L–T | mean Δif (dro−naive) | p_IF (less) | Verdict (honest) |
|---------|---|----------|----------|----------------------|-------------|----------|----------------------|-------------|------------------|
| adult | 0.0 | 6 | **6–0–0** | −6.42e-3 | **0.016** | **4–2–0** | −9.59e-4 | 0.156 (n.s.) | DP win; **IF mixed / not sig** |
| adult | 0.1 | 6 | **6–0–0** | −4.97e-3 | **0.016** | **6–0–0** | −7.89e-3 | **0.016** | **Clear IF + DP win** |
| adult | 0.2 | 6 | **6–0–0** | −1.88e-3 | **0.016** | **6–0–0** | −5.33e-3 | **0.016** | **Clear IF + DP win** |
| adult | 0.3 | 3 | **0–3–0** | +2.45e-3 | incomplete | **3–0–0** | −7.61e-3 | incomplete | **IF win so far; DP LOSS so far** |
| adult | 0.4 | 0 | — | — | — | — | — | — | DRO missing |
| credit | 0.0 | 6 | **6–0–0** | −8.06e-4 | **0.016** | **4–2–0** | −1.93e-4 | 0.156 (n.s.) | Small DP win; **IF not sig** |
| credit | 0.1 | 0 | — | — | — | — | — | — | incomplete |

Two-sided p when one-sided = 0.015625: **p₂ = 0.03125** (still &lt;0.05).  
When one-sided = 0.156: **p₂ = 0.3125**.

### 4.2 Per-seed losses (flag explicitly)

**Adult α=0.0 — IF losses (DRO higher IF):**
- seed 2: naive if=0.09362 → dro if=0.09409 (LOSS)
- seed 4: naive if=0.08769 → dro if=0.08983 (LOSS)

**Adult α=0.3 — DP losses on all 3 paired seeds (incomplete but consistent direction):**
- seed 0: dp 0.03516 → 0.03829 (LOSS)
- seed 1: dp 0.01476 → 0.01691 (LOSS)
- seed 2: dp 0.01999 → 0.02205 (LOSS)  
  Meanwhile IF improved on all three. **Tradeoff, not free lunch — flag for meeting.**

**Credit α=0.0 — IF losses:**
- seed 2: if 0.01838 → 0.01870 (LOSS)
- seed 5: if 0.02387 → 0.02400 (LOSS)

No DP losses on any complete cell (n=6) under IF attack except the incomplete adult α=0.3 pattern above.

---

## 5. IF-attack story vs DP-attack story (same cells)

DP attack has full 6×6 for all datasets/α. `if_clean` under DP attack is ~0 (not informative). Comparison uses **dp_clean** (and acc) on the DP-attack arm.

### 5.1 Side-by-side (mean dp_clean)

| dataset | α | IF-attack naive dp | IF-attack dro dp | DP-attack naive dp | DP-attack dro dp | DP-attack DP W–L | DP-attack p_less |
|---------|---|--------------------|------------------|--------------------|------------------|------------------|------------------|
| adult | 0.0 | 0.1491 | 0.1426 | 0.1491 | 0.1426 | 6–0 | 0.016 |
| adult | 0.1 | 0.0819 | 0.0770 | 0.2026 | 0.1999 | 5–1 | 0.031 |
| adult | 0.2 | 0.0474 | 0.0455 | 0.2452 | 0.2334 | 6–0 | 0.016 |
| adult | 0.3 | 0.0227 / **0.0257** (n=3 dro) | | 0.2848 | 0.2614 | 6–0 | 0.016 |
| adult | 0.4 | 0.0042 / missing dro | | 0.3140 | 0.2855 | 6–0 | 0.016 |
| credit | 0.0 | 0.0127 | 0.0119 | 0.0127 | 0.0119 | 6–0 | 0.016 |

### 5.2 Interpretation (honest)

1. **α=0 is attack-invariant for these metrics** (adult & credit): IF-attack and DP-attack rows match on dp/acc — consistent with zero (or ineffective) attack budget at α=0, so both arms train the same.

2. **α&gt;0 diverges sharply by attack type.** Under **DP attack**, clean DP stays large (~0.20–0.31 adult) and DRO reduces it. Under **IF attack**, clean DP is much smaller (models are not DP-adversarially trained), while **IF** becomes the stressed metric.

3. **You cannot transfer the DP-attack win narrative to IF without looking at `if_clean`.** On Adult α=0.1–0.2 the IF narrative **does** support DRO (6–0, p=0.016 on IF). On Adult α=0.0 and Credit α=0.0, DP wins but **IF does not clear significance**.

4. **Adult α=0.3 tension:** DP-attack arm says DRO helps DP a lot (−0.023 mean). IF-attack partial says DRO **hurts** DP (+0.0024 on 3 seeds) while helping IF. If this holds at n=6, the meeting line is: *DRO trades DP for IF under IF attack at high α* — not “DRO wins everything.”

5. Accuracy: DRO does not tank accuracy under IF attack on completed cells; often slight gains (e.g. adult α=0.1–0.2). Not the failure mode here.

---

## 6. Statistical caveats (for Agent I)

- n=6 Wilcoxon cannot go below p=0.015625 one-sided; “highly significant” is overselling — say **p=0.016, all seeds same direction**.
- Incomplete cells (adult α=0.3, credit α≥0.1, all lsac) must stay **out of abstract claims**.
- Multiple α × metrics × datasets: no multiplicity correction applied; treat as exploratory mid-sweep.
- Win counts without Wilcoxon (n&lt;6) are directional only.

---

## 7. What to say in the meeting (script-level)

**Safe claims now:**
- IF attack is live and non-degenerate (max \|if_clean\| ≈ 0.098).
- Adult α=0.1 and 0.2: DRO beats naive on **both** IF and DP under IF attack, 6/6 seeds, Wilcoxon p≈0.016.
- Credit α=0.0: DRO better on DP (6/6, p≈0.016); IF only 4/6, not significant.
- Sweep ~1/3 done on IF grid; LSAC not started on IF.

**Do not claim yet:**
- “DRO always wins on IF” — false at adult α=0 and credit α=0 (mixed / n.s.).
- “DRO never worsens DP under IF attack” — **contradicted by adult α=0.3 seeds 0–2**.
- Any LSAC IF result.
- Full α grid for credit.

**Watch items when sweep finishes:**
1. Adult α=0.3: does DP loss hold at n=6?  
2. Adult α=0.4: does IF stay favorable as DP collapses near 0?  
3. Credit α&gt;0 and full LSAC: any dataset-specific failure (cf. historical Adult tau issues).

---

## 8. Raw paired detail (complete cells)

### Adult α=0.0 (IF attack)

| seed | naive dp | dro dp | naive if | dro if | naive acc | dro acc |
|------|----------|--------|----------|--------|-----------|---------|
| 0 | 0.15162 | 0.14175 | 0.09781 | 0.09580 | 0.8176 | 0.8182 |
| 1 | 0.15786 | 0.15106 | 0.09641 | 0.09334 | 0.8143 | 0.8165 |
| 2 | 0.14666 | 0.14430 | 0.09362 | **0.09409** | 0.8124 | 0.8151 |
| 3 | 0.15096 | 0.14070 | 0.09576 | 0.09503 | 0.8143 | 0.8168 |
| 4 | 0.14406 | 0.14147 | 0.08769 | **0.08983** | 0.8142 | 0.8140 |
| 5 | 0.14317 | 0.13653 | 0.09396 | 0.09142 | 0.8082 | 0.8075 |

### Adult α=0.1 (IF attack) — clean sweep

| seed | naive dp | dro dp | naive if | dro if | naive acc | dro acc |
|------|----------|--------|----------|--------|-----------|---------|
| 0 | 0.08672 | 0.08613 | 0.04294 | 0.03515 | 0.8208 | 0.8223 |
| 1 | 0.08459 | 0.07834 | 0.03986 | 0.03253 | 0.8129 | 0.8137 |
| 2 | 0.07790 | 0.07449 | 0.04152 | 0.03389 | 0.8116 | 0.8138 |
| 3 | 0.07838 | 0.06828 | 0.04322 | 0.03344 | 0.8106 | 0.8116 |
| 4 | 0.08535 | 0.08184 | 0.03719 | 0.03060 | 0.8125 | 0.8136 |
| 5 | 0.07876 | 0.07280 | 0.03929 | 0.03107 | 0.8070 | 0.8074 |

### Adult α=0.2 (IF attack) — clean sweep

| seed | naive dp | dro dp | naive if | dro if | naive acc | dro acc |
|------|----------|--------|----------|--------|-----------|---------|
| 0 | 0.05762 | 0.05695 | 0.03332 | 0.02660 | 0.7879 | 0.7941 |
| 1 | 0.04365 | 0.04312 | 0.02488 | 0.01928 | 0.7881 | 0.7903 |
| 2 | 0.04696 | 0.04665 | 0.03075 | 0.02494 | 0.7758 | 0.7816 |
| 3 | 0.04162 | 0.03531 | 0.02342 | 0.01776 | 0.7851 | 0.7820 |
| 4 | 0.05205 | 0.04914 | 0.02733 | 0.02311 | 0.7802 | 0.7828 |
| 5 | 0.04264 | 0.04207 | 0.02568 | 0.02175 | 0.7686 | 0.7716 |

### Adult α=0.3 (IF attack) — **partial, n=3**

| seed | naive dp | dro dp | naive if | dro if | naive acc | dro acc |
|------|----------|--------|----------|--------|-----------|---------|
| 0 | 0.03516 | **0.03829** | 0.03519 | 0.02763 | 0.7362 | 0.7530 |
| 1 | 0.01476 | **0.01691** | 0.02871 | 0.02161 | 0.7190 | 0.7350 |
| 2 | 0.01999 | **0.02205** | 0.03321 | 0.02503 | 0.7034 | 0.7228 |

### Credit α=0.0 (IF attack)

| seed | naive dp | dro dp | naive if | dro if | naive acc | dro acc |
|------|----------|--------|----------|--------|-----------|---------|
| 0 | 0.00995 | 0.00917 | 0.02007 | 0.01992 | 0.8038 | 0.8048 |
| 1 | 0.01427 | 0.01322 | 0.02473 | 0.02419 | 0.8083 | 0.8093 |
| 2 | 0.01104 | 0.01026 | 0.01838 | **0.01870** | 0.7987 | 0.8013 |
| 3 | 0.00750 | 0.00692 | 0.02169 | 0.02094 | 0.8045 | 0.8062 |
| 4 | 0.01681 | 0.01629 | 0.03282 | 0.03266 | 0.8052 | 0.8065 |
| 5 | 0.01692 | 0.01580 | 0.02387 | **0.02400** | 0.8117 | 0.8127 |

---

## 9. File / reproduction notes

- Analysis scripted ad hoc against `results/canonical_tau1.json`; filter `attack == "if"`.
- Metrics: `dp_clean`, `if_clean`, `acc_clean` as logged by the sweep.
- Wilcoxon: `scipy.stats.wilcoxon`, one-sided less on `(dro − naive)`.
- Re-run this brief when IF row count hits ~180 or after adult α=0.3/0.4 and credit close.

**Bottom line for Agent I:** Mid-sweep IF evidence **supports DRO on Adult at α=0.1–0.2 (IF+DP)**. Elsewhere either incomplete, mixed, or (α=0.3 partial) a **DP regression under IF attack**. Do not overclaim.
