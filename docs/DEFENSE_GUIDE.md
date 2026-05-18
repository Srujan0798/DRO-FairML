# DRO-FAIR Oral Defense Guide

## Questions You WILL Be Asked

### Algorithm 1 (Most Important)
**Q: "Walk me through Algorithm 1 step by step. Why θ before λ before p?"**

**A:**
1. Forward pass: compute `h̃ = σ(τ·f_θ(x))` with current θ
2. Compute losses: tilted risk `L_tilt`, DP violation `g_DP`, IF violation `g_IF`
3. Update θ (outer minimization): gradient descent on `L_tilt + λ_DP·g_DP + λ_IF·g_IF`
4. Update λ (dual ascent): `λ ← clamp(λ + lr_λ·g, 0, λ_max)`
5. Update p (inner maximization): K=10 steps of gradient ASCENT on `g(p)` alone (NOT λ·g)

Why this order: θ must adapt to the current p before p re-optimizes for the NEW θ. If p updates first, it optimizes for the old θ, creating a stale gradient.

**Q: "Why is the inner gradient ∇g and not λ∇g?"**

**A:** The inner loop maximizes `g(θ, p)` over p. The Lagrangian is `L_tilt + λ·g`. For fixed λ, maximizing `g` is equivalent to maximizing `λ·g` when λ>0 (same argmax). But using `λ·g` causes numerical instability when λ grows. Using `∇g` alone avoids this.

### Theory
**Q: "Derive the DP radius ρ_j = α / ((1−α)π_j + α)."**

**A:** Start from TV distance bound. Under α-corruption, group j's conditional distribution is perturbed. The worst-case TV distance between clean and corrupted group-j distribution is bounded by α / ((1−α)π_j + α). This comes from counting how many samples in group j can be corrupted given the overall α budget.

**Q: "Derive bias correction π_clean = (π_obs − α)/(1 − 2α)."**

**A:** Observed proportion: π_obs = (1−α)·π_clean + α·(1−π_clean). Solve for π_clean: π_obs = π_clean − α·π_clean + α − α·π_clean = π_clean(1−2α) + α. Therefore π_clean = (π_obs − α)/(1−2α).

**Q: "Why is the L1-ball radius 2ρ and not ρ?"**

**A:** TV distance bounds total variation: TV(P,Q) = ½ Σ |P_i − Q_i|. The L1 ball constraint is Σ |p_i − q_i| ≤ r. To convert TV ≤ ρ to L1: r = 2ρ.

### Threat Model
**Q: "What is your adversarial threat model?"**

**A:** White-box attacker controls α-fraction of training samples. Can modify features (PGD), labels (coordinated to maximize DP gap), and protected attributes (70% targeted at minority). Attacker knows the model architecture and training data. Test data remains clean.

**Q: "Theorem 6.1 guarantees fairness under random TV-ball corruption. What holds under your adversarial model?"**

**A:** The TV-ball containment still holds in theory (Theorems 4.2 and 4.3). Empirically: holds on Credit and LSAC (6 significant wins each). **Fails on Adult α=0.1–0.3** because coordinated attacks exploit Adult's already-large baseline DP (~0.17), triggering λ_DP runaway and model collapse. This is documented as an honest limitation.

### Results
**Q: "You claim 5/9 IF wins with Wilcoxon. Why not 7/9?"**

**A:** Mean-based gives 7/9 (DRO mean < Naive mean in 7 cells). But Adult α=0.2 and Credit α=0.2 have p>0.05 despite mean differences. The report explicitly claims "Wilcoxon p<0.05" in figure captions, so 5/9 is the only internally consistent answer.

**Q: "Why does Adult fail at high α?"**

**A:** Adult has baseline DP ~0.17 (8× larger than Credit/LSAC). Coordinated label flips amplify this disparity. DRO's conservative TV radii cause λ_DP to grow until the model collapses to near-uniform predictions (accuracy 25–40% on 6/10 seeds at α=0.3). This is NOT a code bug — it's a fundamental challenge when adversarial corruption exploits inherently large group disparities.

### Hyperparameters
**Q: "Why lambda_max=1.5 and not the paper's 2.0?"**

**A:** With λ_max=2.0, λ_DP runaway on Adult caused model collapse even at α=0.2. Reducing to 1.5 caps the penalty and prevents collapse while preserving wins on Credit/LSAC. Documented as stability fix in report Table 3.

## What To Do If Caught Off-Guard

1. **Admit uncertainty honestly.** "That's a good question — I need to think through the derivation more carefully."
2. **Point to the code.** "The implementation is in `src/training/dro_fair.py` line X. Let me walk you through it."
3. **Cite the report.** "This is documented in Section Y of the report."
4. **Never defend a wrong number.** If a professor finds a discrepancy, say: "You're right — let me check the data and fix it."
