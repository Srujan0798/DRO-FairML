# Speaking notes (for you only — do not send)

**Share / screen-share:** `docs/MEETING_HANDOUT_2026-08-04.md` + 2 PDFs  
**Do not share:** internal STATUS, agent plans, VERIFICATION_REPORT, ADVISOR_CONCERNS, bash blocks

---

## Open (10 sec)
“Update on DRO-Fair under Fairness-Targeted PGD. Main message: **fixed τ = 1** makes DRO robust on Adult and Credit at **α ≤ 0.2**. Earlier ‘fragile DRO’ was a **τ = 100** temperature artifact.”

## Protocol (10 sec)
“Full grid done: **540 runs**, 6 seeds, K_inner = 10, three attacks.”

## Results (30 sec)
“Adult and Credit: DRO lower DP than Naive under DP and Combined attacks.  
Honest nuance: Adult DP at α = 0.1 is **5 of 6 seeds**, still p = 0.03.  
LSAC Combined: DRO wins. **LSAC DP: model collapses** — we flag it, not hide it.”

## IF (20 sec)
“First real IF-attack numbers. **Mixed** — helps Adult/Credit at moderate α; **not** a clean win everywhere. Adult α = 0.3 under IF attack: IF better but DP worse.”

## Scope (15 sec)
“We only claim **α ≤ 0.2** for accuracy — above that both methods fall below the constant predictor on Adult and Credit.”

## Figures (while sharing)
1. `figures/fig_tau1_headline.pdf`  
2. `figures/fig_final_wilcoxon_table.pdf`

## If asked about UTKFace
“Full real-feature grid is done — 90 configs on ResNet18 features from 23k faces. Clean-test DP is mixed: significant DRO wins mainly at high α, not the Adult low-α story. We treat it as a real image-feature pilot, not a tabular copy.”

## If asked “verify claims”
“Every number is from the locked 540-row file; we rechecked win counts and Wilcoxon ourselves — including the 5/6 cell.”

## Close
“Package for Aug 10: paper and report with τ=1 story, honest IF mixed, LSAC degeneracy, and UTKFace as mixed pilot.”
