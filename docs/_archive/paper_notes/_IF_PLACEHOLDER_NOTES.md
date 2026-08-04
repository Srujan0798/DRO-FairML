# IF results placeholder notes (honest template)

**Status:** PREP only — **no numbers invented here.**  
**Fill source:** `results/if_wilcoxon_summary.txt` + regenerated  
`paper/auto_generated/*.tex` / `report/sections/auto_generated_*.tex` after the 540-row grid.

Partial grid already showed non-degenerate IF (IFmax ~0.1, per-config ~0.025–0.033).  
Expect **mixed** outcomes (esp. possible soft/negative cells on Adult/IF at higher α and/or LSAC/IF). Report them plainly.

---

## Template paragraph (paper Results — replace pending block)

```latex
\paragraph{IF-targeted attack (canonical $\tau{=}1$, $n{=}6$).}
Under the IF-targeted PGD attack, DRO-FAIR vs.\ Naive-FAIR on
\textbf{[metric: IF violation / DP under IF attack --- pick primary]}:
% FILL from if_wilcoxon_summary.txt --- examples of honest phrasing:
% - Adult: DRO lower IF at $\alpha\le0.2$ in $k/6$ seeds ($p{=}\ldots$); or mixed/ns.
% - Credit: ...
% - LSAC: ... (if degenerate or loss, say so; do not bury).
% Secondary: DP measured under IF attack (coupling / Q7).
Full cells in Table~\ref{tab:...} (auto-generated from
\texttt{results/canonical\_tau1.json}).  Earlier IF metrics were
identically $\sim10^{-10}$ due to a threshold bug; those figures remain
withdrawn.  The numbers below use the fixed cosine-based IF metric.
```

---

## Template sentence (Abstract / Conclusion — one line each)

```text
Under the IF-targeted attack, [DRO significantly reduces IF vs Naive on
{Adult,Credit} at α≤0.2 (p<…, n=6) | results are mixed: … | DRO does not
improve IF on …].  We report all cells, including non-significant and
negative ones.
```

Pick **exactly one** branch from the real summary; do not blend “wins everywhere” with a footnote of losses.

---

## Win-table discipline

- Primary win definition for IF attack: **lower IF violation** (not DP), unless the paper explicitly studies cross-metric coupling.
- Scope claims to **α ≤ 0.2** unless accuracy stays above the constant-predictor baseline.
- LSAC: if accuracy remains pinned ~0.90, flag **diagnostic/degenerate** the same way as LSAC/DP (`docs/LSAC_DEGENERACY.md`), even if IF “wins.”
- Never quote Jun-30 “IF = 0.0195” — that was **mislabelled DP** (`docs/KULDEEP_CORRECTION.md` §1).

---

## Checklist when pasting numbers

- [ ] Every p-value and win count matches `if_wilcoxon_summary.txt`
- [ ] Auto tables show non-zero IF where expected (no blanket `0.0000 ± 0.0000`)
- [ ] Paper and report use the **same** IF headline sentence
- [ ] “Pending / cluster re-run” deleted from IF sections
- [ ] UTKFace still not claimed unless real image run exists
