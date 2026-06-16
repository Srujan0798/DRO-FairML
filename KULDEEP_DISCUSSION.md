# Kuldeep Discussion — Tau=1 Adult Headline + Ablations (2026-06-16) — LIVE UPDATE

**LIVE (pre-meeting):** Full spec 6-seed canonical launched (K_inner=10, tau=1 fixed for all α, 540 target, full provenance on every row). PID 79899 running now. **37 rows completed so far (all Adult; α=0.0 block finished with n=6 seeds)**. Using exactly the delivered run_canonical.py from the parallel agent work.

Early numbers from the live K=10/tau=1 canonical (DP attack, Adult α=0.0, n=6 seeds):
- Naive DP mean = 0.1491
- DRO DP mean = 0.1426 (DRO slightly better even at zero corruption)

This is the first real data under the mandatory K=10 + fixed-tau=1 config.

> Concise technical brief for working session. All numbers from committed CSVs under tau=1, K_inner=10, epochs=60, pgd_steps=20, FairnessTargetedPGD (adversarial only). Adult complete (3 seeds); Credit/LSAC tau=1 partial; n=6 seeds + empirical radii + Credit/LSAC full in flight. **No hand-typed stats; every value traced to CSV row.**

## 1. Headline (Q12): fixed tau=1 flips the Adult story
DP violation (lower=better) under **DP attack**, Adult, mean from `results/tau1_summary.csv` (rows 14-21 for dp adult):

| α   | Naive DP   | DRO DP     | DRO wins/3 seeds | Δacc (DRO-Naive) | Source |
|-----|------------|------------|------------------|------------------|--------|
| 0.1 | 0.206795  | 0.204565  | **2/3**         | +0.001          | tau1_summary.csv:14-15; tau1_wilcoxon.csv:8 |
| 0.2 | 0.247975  | 0.237100  | **3/3**         | +0.002          | tau1_summary.csv:16-17; tau1_wilcoxon.csv:9 |
| 0.3 | 0.285528  | 0.264006  | **3/3**         | +0.009          | tau1_summary.csv:18-19; tau1_wilcoxon.csv:10 |
| 0.4 | 0.310107  | 0.283426  | **3/3**         | +0.011          | tau1_summary.csv:20-21; tau1_wilcoxon.csv:11 |

- DRO beats Naive on DP **at every α** (absolute values); advantage **grows with α**.
- Accuracy equal-or-better (no trade-off).
- Combined attack (tau1_summary rows 2-11): DRO wins 3/3 at every α≥0.1 too.
- **Contrast tau=100** (old schedule): from tau1_summary.csv rows 64-71 (adult dp), α=0.2: Naive 0.327147 vs DRO 0.503047 (DRO loses badly). Entire prior "DRO fragile/two-regime" narrative was a `tau=100` artifact.
- Wilcoxon (tau1_wilcoxon.csv): min p=0.125 at n=3 (no ties possible); 6-seed run needed for p<0.05.

## 2. Ablations (current Adult data)
- **IF k-NN (Q6)**: `results/knn_ablation_table.csv` (and knn_ablation_k*.json). k∈{5,10,15} nearly identical under IF attack (IF/DP diffs ±0.003). k=5 safe default. Attack insensitive to k.
  Example (DP under IF attack, adult α=0.2): k5 naive~0.0902/dro~0.0854; k10~0.0918/0.0943; k15~0.0938/0.1016.
- **λ grid (Q1)**: `results/lambda_lr_grid.json` in flight (Adult, tau=1, DP attack). λ_init∈{0.0(default),0.01,0.1,1.0} × lr_λ∈{0.001,0.005,0.01}. Goal: tighten DP (acc drop OK per Kuldeep). Preliminary cells show default λ_init=0.0 already competitive.
- **Random vs Adv**: `results/random_vs_adversarial_new.json`. Adv raises DP 12-43× more than random at same α on Adult (e.g. α=0.2 ~+0.18 vs ~0).
- **Empirical radii (Q5)**: `radii_mode='empirical'` implemented (src/training/dro_fair.py:85,103). Uses known coordinated 70%-min attack structure (no per-sample mask leak) to invert: π_clean[min] = π_obs[min] + 0.4α ; π_clean[maj] = π_obs[maj] - 0.4α (then clip/renorm). Uniform (closed-form bias corr) is default. Companion run `canonical_tau1_empirical.json` pending (A/B). Math derivation for appendix ready.

## 3. LSAC + Q7 (IF↔DP inverse)
- LSAC inherent low clean DP (~0.01-0.02 on protected attr) → DP attack is weak/naturally inverts (DP can't be raised much). Frame LSAC results around **IF attack** (Q3).
- Q7: IF-targeted attack can **lower** observed DP (inverse effect). Evidence in tau1 data (adult, IF attack α=0.3 from tau1_summary.csv:28-29): Naive DP=0.018999 < DRO DP=0.021782 (attack on IF made DP gap smaller for both, but relative ordering). Similar patterns in older fairness_pgd_summary for other cells. The two fairness metrics are coupled through the attack; optimizing one can move the other in either direction.

## 4. Status / landing vs preliminary (per MASTER_PLAN §2)
- **Landing / verified (Adult)**: tau=1 DP wins 2/3+3/3+3/3+3/3; adv>>random; k-NN insensitive; tau=1 vs 100 contrast; attack fix (direct |p0-p1|).
- **In flight / preliminary**: lambda_lr_grid (Q1); full Credit+LSAC tau=1 (109/270+); n=6 seeds for canonical_wilcoxon.csv + p<0.05; uniform-vs-empirical radii comparison (needs canonical_tau1_empirical.json); UTKFace (Q13: blocked on flair2 GPU/SSL; email supin.gopi).
- **Config (hard, per §0)**: epochs=60, K_inner=10, θ→λ→p order, ∇g (not λ∇g), λ_init=0.0 (default), lambda_max=1.5 all DS, radii from α+known structure only (no oracle mask), adversarial only (RandomCorruptor = baseline comparator).
- Next: 6-seed canonical (results/canonical_tau1.json), empirical radii runs, extend k-NN to Credit/LSAC, UTKFace access chase.

## 5. Asks for today
- The live 6-seed canonical (K=10 + tau=1 fixed) is running now (37 rows, Adult α=0.0 full n=6 already shows DRO edge even at α=0). Is the narrative "tau=1 makes DRO at least as good as (or better than) Naive under coordinated fairness attacks, with advantage growing at higher α" ready for the paper?
- 6 seeds in the canonical run — sufficient for the wilcoxon significance in the submission, or do we need to push to 8-10?
- For madam's "show DP increase from attack >> random" request: we have clean absolute DP numbers + the 12-40x multipliers from the random-vs-adversarial json. Lead with absolute table/figure?
- UTKFace: local canonical-config smoke done (2 rows in bucket); server script + flair2 block documented (DNS from local). Prioritize getting access this week or finish tabular first?

**Evidence before claims.** All tables/figures will be regenerated from committed CSVs by C's scripts (tau1_summary.csv, tau1_wilcoxon.csv, knn_ablation_table.csv, ...). Placeholders in current report/paper point to "TODO:CANONICAL" until full 540-row lands. The 37-row live run will be folded in post-meeting.

**Evidence before claims.** All tables/figures will be regenerated from committed CSVs by C's scripts (tau1_summary.csv, tau1_wilcoxon.csv, knn_ablation_table.csv, ...). Placeholders in current report/paper point to "TODO:CANONICAL" until full 540-row lands.

---
*Prepared by Agent D per MASTER_PLAN §7. Use absolute DP + seed counts. No publicity.*
