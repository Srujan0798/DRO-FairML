# DRO-FAIR Algorithm Details

## Algorithm 1: Robust Fair Classification via DRO

### Inputs
- Corrupted dataset $S_c = \{(x_i^c, a_i^c, y_i^c)\}_{i=1}^n$
- Corruption-calibrated radii $\rho_{DP,j}$, $\rho_{IF}$
- Learning rates $\eta_\theta$, $\eta_\lambda$, $\eta_p$
- Epochs $T$, inner steps $K$
- Hyperparameters: temperature $\tau$, tilting $\beta$, neighbors $k$

### Initialization
- Model parameters: $\theta \leftarrow \theta_0$
- Lagrange multipliers: $\lambda_{DP}, \lambda_{IF} \leftarrow \lambda_0$
- Importance weights: $\tilde{p}_j \leftarrow \hat{p}_{n_j}$ (uniform per group), $\tilde{p}_{IF} \leftarrow \hat{p}_n$ (uniform global)
- Precompute $k$-NN graph $N(i)$ for all $i$

### Main Loop (for $t = 1$ to $T$)

#### 1. Forward Pass
Compute soft predictions: $\tilde{h}_i = \sigma(\tau \cdot f_\theta(x_i^c))$

#### 2. Compute Losses

**Classification Loss (Tilted ERM):**
$$L_{tilt} = \beta \log\left(\frac{1}{|M|}\sum_{i \in M} \exp\left(\frac{\ell(f_\theta(x_i^c), y_i^c)}{\beta}\right)\right)$$

**Demographic Parity Violation:**
$$g_{DP} = \left|\sum_{i: a_i=1} \tilde{p}_{1,i} \tilde{h}_i - \sum_{i: a_i=0} \tilde{p}_{0,i} \tilde{h}_i\right|$$

**Individual Fairness Violation:**
$$g_{IF} = \frac{1}{n-1}\sum_{i=1}^n \sum_{j \in N(i)} \frac{\tilde{p}_{IF,i} + \tilde{p}_{IF,j}}{2} \cdot \left(|\tilde{h}_i - \tilde{h}_j| - d_{ij} - \gamma\right)_+$$

#### 3. Outer Minimization (Update $\theta$)
$$L = L_{tilt} + \lambda_{DP} \cdot g_{DP} + \lambda_{IF} \cdot g_{IF}$$
$$\theta \leftarrow \theta - \eta_\theta \nabla_\theta L$$
(with gradient clipping)

#### 4. Dual Ascent (Update $\lambda$)
$$\lambda_{DP} \leftarrow \max(0, \lambda_{DP} + \eta_\lambda \cdot g_{DP})$$
$$\lambda_{IF} \leftarrow \max(0, \lambda_{IF} + \eta_\lambda \cdot g_{IF})$$
(clamped to $[0, \lambda_{max}]$)

#### 5. Inner Maximization (Update $\tilde{p}$, repeat $K$ times)
For each group $j \in \{0, 1\}$:
$$\tilde{p}_j \leftarrow \tilde{p}_j + \eta_p \nabla_{\tilde{p}_j} g_{DP}$$
$$\tilde{p}_j \leftarrow \mathcal{P}_{\Delta_{n_j} \cap \mathcal{B}_1(\hat{p}_{n_j}, 2\rho_{DP,j})}(\tilde{p}_j)$$

For global IF weights:
$$\tilde{p}_{IF} \leftarrow \tilde{p}_{IF} + \eta_p \nabla_{\tilde{p}_{IF}} g_{IF}$$
$$\tilde{p}_{IF} \leftarrow \mathcal{P}_{\Delta_n \cap \mathcal{B}_1(\hat{p}_n, 2\rho_{IF})}(\tilde{p}_{IF})$$

### Projection Operator $\mathcal{P}$

The projection onto the intersection of probability simplex and $\ell_1$-ball uses **Dykstra's alternating projection algorithm**:

1. Initialize $x^{(0)} = v$ (vector to project), $p^{(0)} = q^{(0)} = 0$
2. For $m = 1, 2, \ldots$ until convergence:
   - $y^{(m)} = \mathcal{P}_{\Delta}(x^{(m-1)} + p^{(m-1)})$
   - $p^{(m)} = x^{(m-1)} + p^{(m-1)} - y^{(m)}$
   - $x^{(m)} = \mathcal{P}_{\mathcal{B}_1}(y^{(m)} + q^{(m-1)})$
   - $q^{(m)} = y^{(m)} + q^{(m-1)} - x^{(m)}$

Where:
- $\mathcal{P}_{\Delta}$: projection onto probability simplex (sorting-based algorithm)
- $\mathcal{P}_{\mathcal{B}_1}$: projection onto $\ell_1$-ball (soft-thresholding)

## Corruption-Calibrated Radii

Derived from total variation distance bounds (Theorems 4.2 and 4.3):

**Per-group DP radius:**
$$\rho_{DP,j} = \frac{\alpha}{(1-\alpha)\pi_j + \alpha}$$

Where $\pi_j = P(A=j)$ is the group proportion. Minority groups (small $\pi_j$) get larger radii, reflecting their greater vulnerability.

**IF radius:**
$$\rho_{IF} = 2\alpha - \alpha^2$$

## Naive-FAIR Baseline

Special case with $\rho_{DP,j} = \rho_{IF} = 0$ and $K = 0$ (no inner maximization). Uses fixed uniform weights throughout training.

## Adversarial Corruption Protocol

Replaces random noise from the paper with targeted attacks:

1. **Feature Perturbation**: FGSM-style attack moving features towards the opposite class, scaled by column standard deviation
2. **Coordinated Label Flips**: Labels flipped to maximize group rate disparity
3. **Coordinated Attribute Flips**: 70% of corruption focused on minority group

## Computational Complexity

- **Naive-FAIR**: $O(T \cdot n \cdot d)$ per epoch
- **DRO-FAIR**: $O(T \cdot K \cdot n \cdot d)$ per epoch (≈12× overhead due to inner loop)
- **k-NN graph**: $O(n \log n)$ preprocessing
- **Dykstra projection**: $O(m \log m)$ per group of size $m$
