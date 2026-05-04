# fairml-adversarial-noise
the project is to send Adversarrial noise to model and test it instaed of sending normal random noise to check its response through 2nd approach .
Project Title
Implement DRO-FAIR (the 2nd approach from the ICML submission PDF) for joint Demographic Parity (DP) + Individual Fairness (IF) under α-corruption, but replace the paper’s random noise with adversarial noise.
Core Goal
Build and evaluate a binary classifier h : X → {0,1} that satisfies both group fairness (DP) and individual fairness (IF) on the clean test distribution, even when a fraction α ∈ {0.0, 0.1, 0.2, 0.3, 0.4} of the training data is adversarially corrupted. The model must achieve this with provable robustness guarantees and only moderate accuracy cost (1–4 %).
Exact Requirements (2nd Approach Only)
Data Model (PDF Section 4)
Clean distribution P over Z = X × A × Y.
Corrupted distribution P_pert = (1−α)P + αQ, where α < 1/2.
Adversarial noise (not random): Use the exact guidance from https://jonathan-hui.medium.com/adversarial-attacks-b58318bb497b (PGD/FGSM-style attacks on features, coordinated label flips, and protected-attribute flips) to generate the corrupted samples within the α budget. Apply corruption only to train/val sets; test set remains clean.
Fairness Definitions (PDF Section 3 + fairmlbook.org)
DP violation: Δ_DP(h; P) = |p₀ − p₁| where p_a = P(h(X)=1 | A=a).
IF violation: L_{d,γ}(h; P) (metric fairness with task-specific distance d and slack γ).
Enforce both jointly via Lagrangian.
Main Method: DRO-FAIR (PDF Section 6 – the 2nd approach)
Use corruption-calibrated TV uncertainty sets with exact radii: ρ_DP,j = α / ((1−α)π_j + α) for each group j, ρ_IF = 2α − α².
Optimize the min-max Lagrangian (Eq. 19): min_θ max_λ [R(h_θ; S_c) + λ_DP · max_{˜p∈U_DP} g_DP(h_θ, ˜p) + λ_IF · max_{˜p∈U_IF} g_IF(h_θ, ˜p)]
Follow Algorithm 1 (Appendix G) exactly: outer θ update + K=10 inner projected gradient steps on reweighting vectors ˜p (ℓ₁-ball + simplex projection via Dykstra) + dual ascent on λ.
Use k-NN (k=5) approximation for the IF term, τ=100, β=5, and all hyperparameters from Section 7.1.
Baseline
Naive-FAIR (1st approach, PDF Section 5) for direct comparison.
Experiments (PDF Section 7)
Datasets: Adult, Credit, LSAC (exact preprocessing from Appendix G.5).
Metrics: Accuracy ↑, DP violation ↓, IF violation ↓ (mean ± SE over 10 seeds).
Table 1 style results for α = {0.0–0.4}.
Ablations: radius choice, DP-only vs joint, random vs adversarial noise impact.
Report runtime overhead (~12× expected).
Deliverables
Fully working PyTorch code for adversarial corruption + DRO-FAIR (2nd approach).
Reproduce Table 1 (and ablations) showing DRO-FAIR’s superiority under adversarial noise.
Theoretical guarantees hold exactly as proven in Theorems 6.1 and Remark 6.2.
Success Criteria
DRO-FAIR must reduce DP violations by up to 83 % and IF violations by 3–5× versus Naive-FAIR at α ≤ 0.3, with 1–4 % accuracy trade-off, while surviving the stronger adversarial corruption.

