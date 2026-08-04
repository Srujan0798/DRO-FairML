# UTKFace REAL summary

- rows: **90/90**
- attacks: {'dp': 30, 'if': 30, 'combined': 30}
- provenance: {'REAL': 90}
- all REAL: **True**
- protocol: τ=1, k_inner=10, epochs=60, pgd_steps=20, n_seeds=6
- task: gender prediction; protected = race White/non-White; N=23705 features

| attack | α | n | DP naive | DP dro | wins DRO (↓DP) | p | acc N | acc D | IF n | IF d |
|--------|---:|--:|---------:|-------:|---------------:|---:|------:|------:|------:|------:|
| dp | 0.0 | 6 | 0.0211 | 0.0202 | 3/6 | 0.2812 | 0.861 | 0.862 | 0.0706 | 0.0529 |
| dp | 0.1 | 6 | 0.0477 | 0.0505 | 1/6 | 0.9688 | 0.849 | 0.851 | 0.0627 | 0.0473 |
| dp | 0.2 | 6 | 0.1568 | 0.1586 | 2/6 | 0.8438 | 0.793 | 0.792 | 0.0677 | 0.0538 |
| dp | 0.3 | 6 | 0.1824 | 0.1803 | 4/6 | 0.2812 | 0.795 | 0.793 | 0.0921 | 0.0711 |
| dp | 0.4 | 6 | 0.2215 | 0.2132 | 6/6 | 0.0156 | 0.783 | 0.785 | 0.1079 | 0.0808 |
| if | 0.0 | 6 | 0.0211 | 0.0202 | 3/6 | 0.2812 | 0.861 | 0.862 | 0.0706 | 0.0529 |
| if | 0.1 | 6 | 0.0199 | 0.0200 | 3/6 | 0.5781 | 0.852 | 0.854 | 0.0717 | 0.0549 |
| if | 0.2 | 6 | 0.0184 | 0.0191 | 2/6 | 0.7188 | 0.852 | 0.853 | 0.0679 | 0.0507 |
| if | 0.3 | 6 | 0.0162 | 0.0165 | 3/6 | 0.6562 | 0.847 | 0.848 | 0.0535 | 0.0372 |
| if | 0.4 | 6 | 0.0160 | 0.0156 | 4/6 | 0.4219 | 0.841 | 0.841 | 0.0478 | 0.0320 |
| combined | 0.0 | 6 | 0.0211 | 0.0202 | 3/6 | 0.2812 | 0.861 | 0.862 | 0.0706 | 0.0529 |
| combined | 0.1 | 6 | 0.0314 | 0.0329 | 2/6 | 0.9531 | 0.850 | 0.852 | 0.0681 | 0.0520 |
| combined | 0.2 | 6 | 0.0494 | 0.0512 | 1/6 | 0.8438 | 0.849 | 0.849 | 0.0672 | 0.0504 |
| combined | 0.3 | 6 | 0.0869 | 0.0840 | 3/6 | 0.2188 | 0.837 | 0.838 | 0.0634 | 0.0450 |
| combined | 0.4 | 6 | 0.1443 | 0.1356 | 6/6 | 0.0156 | 0.812 | 0.816 | 0.0717 | 0.0503 |

**COMPLETE 90/90 REAL.**

### Honest read (do not overclaim)
- This is an **image-feature** experiment (ResNet18 embeddings), not a pixel-space attack.
- Win pattern does **not** mirror Adult/Credit low-α: significant clean DP wins for DRO appear mainly at **α = 0.4** (DP and Combined, 6/6, p=0.0156). Low/mid α cells are mixed or non-significant (many favor Naive on DP).
- IF attack: clean DP never reaches p&lt;0.05 for DRO; IF values often lower for DRO (coupling, not a DP sweep).
- Use Wilcoxon only with n=6; report losses as well as wins.
- Paper/report framing: **real image-feature pilot, mixed clean-test** — not an Adult/Credit α≤0.2 copy.
