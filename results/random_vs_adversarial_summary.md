# Random vs adversarial (Wave-1 A4) — live summary

Source: `results/random_vs_adversarial.json` — **2** rows so far (target 144).
Protocol: τ=1, K_inner=10, pgd_steps=20, n_seeds=6, attack=dp.
Ratio = mean(DP_adversarial) / mean(DP_random) on clean test (higher ⇒ adversarial raises DP more than random).

| dataset | α | method | n_adv | n_rnd | DP_adv | DP_rnd | ratio | acc_adv | acc_rnd |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| adult | 0.1 | naive | 1 | 1 | 0.2197 | 0.1197 | 1.83 | 0.822 | 0.821 |

Complete (dataset,α,method) cells with n≥6 both arms: **0/12**.

Do **not** put incomplete ratios in the paper abstract. Prefer full 144-row file.
