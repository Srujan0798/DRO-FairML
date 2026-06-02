# Weekly Research Summary — May 29, 2026
**Adversarial Fairness Attacks on DRO-FAIR**

---

## ✅ Both Tasks Completed

### Task 1: PGD Attacks on Tabular Data (270 experiments)
**What:** Gradient-based attacks targeting DP/IF fairness metrics on Adult, Credit, LSAC.

**Result:** DRO-FAIR significantly reduces DP violation under IF-targeted attacks.

| Dataset | Attack | α | DRO Improvement | p-value |
|---------|--------|---|-----------------|---------|
| Credit | IF | 0.2 | **64.5%** DP reduction | 0.031* |
| Credit | IF | 0.3 | **97.5%** DP reduction | 0.031* |
| LSAC | IF | 0.3 | **96.2%** DP reduction | 0.031* |

*Statistically significant (Wilcoxon signed-rank, n=5 seeds).*

**Insight:** Under DP-targeted attacks on Adult, the adversary is strong enough to break DRO's advantage — a new finding not seen in prior random-noise work.

---

### Task 2: UTKFace Image Data (9 experiments, GPU server)
**What:** ResNet18 features (512-dim) from 23,705 images. Naive-FAIR vs DRO-FAIR.

**Result:** DRO helps on clean data but **hurts under corruption** — opposite of tabular data!

| α | Condition | Naive DP | DRO DP | Winner |
|---|-----------|----------|--------|--------|
| 0.0 | Clean | 0.036 | **0.027** | **DRO** (+25%) |
| 0.1 | Corrupted | **0.094** | 0.130 | **Naive** (+39%) |
| 0.2 | Corrupted | **0.103** | 0.110 | **Naive** (+7%) |

**🔍 New Finding:** On image data, DRO-FAIR **increases** DP violation under label corruption. On tabular data, DRO reduces it. DRO's robustness is **modality-dependent**.

**Hypothesis:** ResNet18 features are fairness-agnostic (no demographic info). DRO overfits to corrupted fairness signal. Tabular features naturally encode demographics, so DRO finds robust patterns.

---

## 📊 Figures Generated

- `fig8_fairness_pgd_comparison.png` — Tabular results
- `fig9_fairness_pgd_curves.png` — Tabular curves
- `fig_utkface_dp_comparison.png` — UTKFace clean vs corrupted
- `fig_utkface_tradeoff.png` — UTKFace accuracy-fairness tradeoff

---

## 🎯 Key Message for This Week

> Adversarial fairness attacks reveal that DRO-FAIR's robustness is **not universal**:
> - **Metric-dependent:** DRO wins against IF attacks, loses against DP attacks
> - **Modality-dependent:** DRO wins on tabular data, loses on image features
> 
> This nuance is novel — prior work only tested random noise, not gradient-based adversarial attacks.

---

## 📋 Next Steps

1. Run more UTKFace seeds (n=5) to confirm the DRO-worse finding
2. Test CelebA / FairFace for larger-scale validation
3. Try deeper networks (ResNet50, ViT)
4. Draft paper for NeurIPS / ICLR

---

*Server: flair2.iitgn.ac.in (2× NVIDIA L40S 48GB) | Code: github.com/Srujan0798/DRO-FairML @ cd8a9e8*
