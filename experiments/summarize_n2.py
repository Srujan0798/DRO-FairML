#!/usr/bin/env python3
"""Agent N2 — high-α rescue summary (analysis-only).

Reads results/high_alpha_tau.json (the 120-config file produced by
experiments/run_n2_high_alpha.py — Kuldeep's Jun-16 3-step protocol):

  STEP 1 — per-α tau: tau ∈ {2,5,20} on Adult/dp at α∈{0.3,0.4}, 6 seeds,
            2 methods. Question: does any τ lift α=0.3 DRO accuracy above
            0.7521 (Adult constant-predictor baseline)?
  STEP 2 — lr_lambda=0.01 (canonical 5e-3) at the same α. Question: does
            the higher dual lr help at high α?
  STEP 3 — epochs=200 + dump_history. History JSONs land at
            results/history_adult_dp_{alpha}_{seed}_{method}.json (per
            run_fairness_pgd.py dump_history code). Questions:
              (i)  does val_loss plateau before epoch 60 (underfitting) or
                   keep decreasing (would benefit from more epochs)?
              (ii) does val_acc at epoch 200 exceed 0.7521?

Outputs:
    results/high_alpha_summary.md
    results/high_alpha_convergence.png   (train/val loss vs epoch per τ per α)

Run:
    PYTHONPATH=. python3 experiments/summarize_n2.py
"""
from __future__ import annotations

import json
import os
import glob
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "high_alpha_tau.json")
OUT_MD = os.path.join(RESULTS_DIR, "high_alpha_summary.md")
OUT_FIG = os.path.join(RESULTS_DIR, "high_alpha_convergence.png")

ADULT_CONSTANT_PREDICTOR = 0.7521
ALPHAS = [0.3, 0.4]
TAUS_STEP1 = [2.0, 5.0, 20.0]
SEEDS = list(range(6))
METHODS = ["naive", "dro"]
CANONICAL_LR_LAMBDA = 5e-3
LR_LAMBDA_STEP2 = 0.01
CANONICAL_EPOCHS = 60
STEP3_EPOCHS = 200
PLATEAU_EPOCH = 60  # the underfitting question: plateau before this?


def _load_rows():
    if not os.path.exists(RESULTS_FILE):
        return [], f"MISSING: {RESULTS_FILE}"
    with open(RESULTS_FILE) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{RESULTS_FILE} is not a JSON list")
    return rows, f"{RESULTS_FILE} ({len(rows)} rows)"


def _per_cell(rows, step, fields):
    """Group rows of a given step by `fields` -> list of row dicts."""
    g = defaultdict(list)
    for r in rows:
        if r.get("ablation_step") != step:
            continue
        k = tuple(float(r.get(f, np.nan)) if f in ("alpha", "tau", "lr_lambda")
                  else r.get(f) for f in fields)
        g[k].append(r)
    return g


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _sem(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.std(xs) / np.sqrt(len(xs))) if xs else float("nan")


def _step1_table(rows):
    """Per (α, τ) DRO acc/dp; also naive baseline acc at the same α.

    STEP 1 cells use tau ∈ {2,5,20} (NOT 1.0). For the canonical τ=1.0
    comparator we load it from the locked canonical file READ-ONLY. If that
    file is unavailable we fall back to reporting STEP 1 cells only.
    """
    g = _per_cell(rows, "step1_tau", ["alpha", "tau", "method"])
    # also pull naive rows (same α) for completeness
    out = {}
    for (alpha, tau, method), rs in g.items():
        out[(alpha, tau, method)] = {
            "n": len(rs),
            "acc_mean": _mean([r["acc_clean"] for r in rs]),
            "acc_sem": _sem([r["acc_clean"] for r in rs]),
            "dp_mean": _mean([r["dp_clean"] for r in rs]),
            "dp_sem": _sem([r["dp_clean"] for r in rs]),
        }
    return out


def _step2_table(rows):
    """Per (α, method): lr_lambda=0.01 vs canonical 5e-3 (from STEP 1 τ=1?
    No — STEP 1 varies τ not lr_lambda. Canonical lr is implicit at τ=1.0
    in STEP 3 / canonical file). For a same-α comparator we use the STEP 3
    τ=1.0 epochs=60 arm... but STEP 3 uses epochs=200, so not directly
    comparable. We instead report STEP 2 cells on their own and against
    the canonical Adult/dp α=0.3,0.4 rows (read-only).
    """
    g = _per_cell(rows, "step2_lr_lambda", ["alpha", "method"])
    out = {}
    for (alpha, method), rs in g.items():
        out[(alpha, method)] = {
            "n": len(rs),
            "acc_mean": _mean([r["acc_clean"] for r in rs]),
            "acc_sem": _sem([r["acc_clean"] for r in rs]),
            "dp_mean": _mean([r["dp_clean"] for r in rs]),
            "dp_sem": _sem([r["dp_clean"] for r in rs]),
        }
    return out


def _canonical_adult_reference():
    """Read-only pull of canonical Adult/dp rows at α∈{0.3,0.4} from the
    locked canonical_tau1.json (DRO and naive). Returns
    {(alpha, method): {acc_mean, dp_mean, n}}.
    """
    path = os.path.join(RESULTS_DIR, "canonical_tau1.json")
    if not os.path.exists(path):
        return {}
    out = {}
    try:
        with open(path) as f:
            rows = json.load(f)
    except Exception:
        return {}
    for r in rows:
        if r.get("dataset") != "adult" or r.get("attack") != "dp":
            continue
        if r.get("method") not in METHODS:
            continue
        a = float(r.get("alpha", -1))
        if a not in ALPHAS:
            continue
        out.setdefault((a, r["method"]), []).append(r)
    return {k: {
        "acc_mean": _mean([x["acc_clean"] for x in v]),
        "dp_mean": _mean([x["dp_clean"] for x in v]),
        "n": len(v),
    } for k, v in out.items()}


def _load_history(alpha, seed, method):
    """Read results/history_adult_dp_{alpha}_{seed}_{method}.json.
    Returns the history dict or None."""
    pat = os.path.join(RESULTS_DIR,
                       f"history_adult_dp_{alpha}_{seed}_{method}.json")
    if not os.path.exists(pat):
        return None
    try:
        with open(pat) as f:
            return json.load(f)
    except Exception:
        return None


def _list_history_files():
    """Return list of (alpha, seed, method) tuples for which a history JSON
    exists. Glob is robust to either {alpha} being a float like 0.3 or 0.4.
    """
    out = []
    for p in glob.glob(os.path.join(RESULTS_DIR, "history_adult_dp_*.json")):
        base = os.path.basename(p)
        # history_adult_dp_{alpha}_{seed}_{method}.json
        parts = base[len("history_adult_dp_"):-len(".json")].split("_")
        if len(parts) < 3:
            continue
        method = parts[-1]
        seed = parts[-2]
        alpha = "_".join(parts[:-2])  # robust to "0.3" form
        try:
            out.append((float(alpha), int(seed), method))
        except ValueError:
            continue
    return out


def _val_loss_plateau(history, plateau_epoch=PLATEAU_EPOCH):
    """Decide whether val_loss plateaus before `plateau_epoch` or keeps
    decreasing across the full run. Returns a dict with a verdict string
    and the supporting numbers.

    Heuristic: split the val_loss curve into [0, plateau_epoch] and
    [plateau_epoch, end]. Compute the median slope in each window via a
    simple least-squares fit. If the early slope is materially more
    negative than the late slope AND the late slope is ~flat (|slope| below
    a small floor), -> 'plateau_before_60' (underfitting is NOT the issue;
    it has converged). If the late slope is still meaningfully negative,
    -> 'still_decreasing' (would benefit from more epochs).
    """
    vl = history.get("val_loss", [])
    vl = [float(x) for x in vl if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if len(vl) < 5:
        return {"verdict": "insufficient_data", "n": len(vl)}
    n = len(vl)

    def _slope(seg):
        if len(seg) < 2:
            return 0.0
        x = np.arange(len(seg))
        y = np.array(seg)
        if np.allclose(y.std(), 0.0):
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    early = vl[:plateau_epoch] if len(vl) >= plateau_epoch else vl
    late_start = min(plateau_epoch, n - 1)
    late = vl[late_start:] if late_start < n else []
    slope_early = _slope(early)
    slope_late = _slope(late) if late else 0.0
    floor = 1e-5
    if slope_late < -floor:
        verdict = "still_decreasing"
    elif abs(slope_late) <= floor and slope_early < -floor:
        verdict = "plateau_before_60"
    elif slope_early < -floor and slope_late >= 0:
        verdict = "plateau_before_60"
    else:
        verdict = "plateau_before_60" if abs(slope_late) <= abs(slope_early) else "still_decreasing"
    return {
        "verdict": verdict,
        "n": n,
        "slope_early": slope_early,
        "slope_late": slope_late,
        "val_loss_first": vl[0] if vl else None,
        "val_loss_at_60": vl[plateau_epoch - 1] if len(vl) >= plateau_epoch else None,
        "val_loss_last": vl[-1] if vl else None,
    }


def _val_acc_at_epoch(history, epoch):
    """History records val_acc EVERY epoch (DroFairTrainer.fit appends each
    epoch; NaiveFairTrainer only every 5 epochs). Best-effort index."""
    va = history.get("val_acc", [])
    if not va:
        return None
    if len(va) >= epoch + 1:
        return float(va[epoch])
    return float(va[-1])


def _best_val_acc(history):
    va = history.get("val_acc", [])
    va = [float(x) for x in va if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return max(va) if va else None


def _step3_summary(rows):
    """Aggregate STEP 3 cells from the JSON (test-set acc/dp), and read the
    per-(α,seed,method) history JSONs for convergence diagnostics."""
    g = _per_cell(rows, "step3_convergence", ["alpha", "method"])
    cells = {}
    for (alpha, method), rs in g.items():
        cells[(alpha, method)] = {
            "n": len(rs),
            "acc_mean": _mean([r["acc_clean"] for r in rs]),
            "acc_sem": _sem([r["acc_clean"] for r in rs]),
            "dp_mean": _mean([r["dp_clean"] for r in rs]),
            "dp_sem": _sem([r["dp_clean"] for r in rs]),
            "exceeds_constant": _mean([r["acc_clean"] for r in rs]) > ADULT_CONSTANT_PREDICTOR,
        }

    # Convergence from history JSONs (only DRO dumped; naive history kept
    # in-process only per run_fairness_pgd.py:190 — 'Only DRO history is
    # dumped'). Confirm by listing files.
    history_files = _list_history_files()
    dro_hist = [h for h in history_files if h[2] == "dro"]
    naive_hist = [h for h in history_files if h[2] == "naive"]

    plateau_verdicts = defaultdict(list)  # (alpha, method) -> list of verdict dicts
    val_acc_at_200 = defaultdict(list)    # (alpha, method) -> list of floats
    best_val_acc = defaultdict(list)
    for (alpha, seed, method) in history_files:
        h = _load_history(alpha, seed, method)
        if h is None:
            continue
        plateau_verdicts[(alpha, method)].append(_val_loss_plateau(h))
        va200 = _val_acc_at_epoch(h, 199)  # 0-indexed epoch 200 = index 199
        if va200 is not None:
            val_acc_at_200[(alpha, method)].append(va200)
        bva = _best_val_acc(h)
        if bva is not None:
            best_val_acc[(alpha, method)].append(bva)

    return cells, dict(plateau_verdicts), dict(val_acc_at_200), dict(best_val_acc), \
        {"dro": len(dro_hist), "naive": len(naive_hist)}


def _plot_convergence(alpha, method_filter=("dro",)):
    """Plot train_loss and val_loss vs epoch for each (α, seed, method) in
    method_filter, saved as a single PNG per α. Returns list of saved paths.
    This produces the literal artifact Kuldeep asked for: train/val
    loss-vs-epoch convergence plots per τ per α.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    paths = []
    history_files = _list_history_files()
    for alpha in ALPHAS:
        fig, ax = plt.subplots(figsize=(7, 5))
        any_data = False
        for (a, seed, method) in sorted(history_files):
            if a != alpha or method not in method_filter:
                continue
            h = _load_history(a, seed, method)
            if h is None:
                continue
            tl = h.get("train_loss", [])
            vl = h.get("val_loss", [])
            ep = np.arange(1, len(tl) + 1)
            if len(tl):
                ax.plot(ep, tl, color="C0", alpha=0.35, linewidth=0.8)
                any_data = True
            if len(vl):
                ax.plot(np.arange(1, len(vl) + 1), vl, color="C1", alpha=0.5,
                        linewidth=0.8)
                any_data = True
        if not any_data:
            plt.close(fig)
            continue
        ax.axvline(PLATEAU_EPOCH, color="grey", linestyle="--", linewidth=1,
                   label="epoch 60 (canonical)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"Convergence (α={alpha}, Adult, dp) — "
                     f"train (blue) / val (orange), all seeds")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, f"high_alpha_convergence_a{alpha}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def _fmt(x, p=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{p}f}"


def write_md(rows, source, step1, step2, ref, step3_data, fig_paths):
    cells3, plateau, va200, best_va, hist_counts = step3_data
    n_expected = 120
    n_step = {s: sum(1 for r in rows if r.get("ablation_step") == s)
              for s in ["step1_tau", "step2_lr_lambda", "step3_convergence"]}

    L = []
    L.append("# Agent N2 — high-α rescue (Kuldeep Jun-16 3-step protocol)")
    L.append("")
    L.append("Analysis-only. No new training. Source: `results/high_alpha_tau.json` "
             f"({len(rows)}/{n_expected} rows). Adult constant-predictor acc = "
             f"**{ADULT_CONSTANT_PREDICTOR:.4f}**.")
    L.append("")
    L.append("Kuldeep, Jun 16 (verbatim): \"Different tau value 1st if not improving "
             "then change learning rates for lamda or something else check loss "
             "convergence plots and choose according to it on validation set\".")
    L.append("")
    L.append("## Coverage")
    L.append("")
    L.append(f"- STEP 1 (τ ∈ {{2,5,20}}): {n_step['step1_tau']}/72 rows")
    L.append(f"- STEP 2 (lr_lambda=0.01): {n_step['step2_lr_lambda']}/24 rows")
    L.append(f"- STEP 3 (epochs=200, dump_history): {n_step['step3_convergence']}/24 rows")
    if sum(n_step.values()) < n_expected:
        L.append("- **INCOMPLETE** — partial-data mode; re-run as rows land (idempotent).")
    L.append("")

    # ---- STEP 1 ----
    L.append("## STEP 1 — per-α τ: does any τ lift α=0.3 DRO acc above 0.7521?")
    L.append("")
    if ref:
        L.append("Canonical τ=1.0 (read-only, from `results/canonical_tau1.json`):")
        L.append("")
        L.append("| α | method | n | acc | dp |")
        L.append("|---|---|---|---|---|")
        for (a, m), v in sorted(ref.items()):
            L.append(f"| {a:.1f} | {m} | {v['n']} | {_fmt(v['acc_mean'])} | {_fmt(v['dp_mean'])} |")
        L.append("")
    L.append("STEP 1 cells (τ ∈ {2,5,20}, Adult, dp, α∈{0.3,0.4}, 6 seeds, "
             "2 methods). ✓ = DRO acc > 0.7521 (Adult constant predictor).")
    L.append("")
    L.append("| α | τ | method | n | acc | dp | acc>0.7521? |")
    L.append("|---|---|---|---|---|---|---|")
    step1_dro_lifts = []
    for (alpha, tau, method), v in sorted(step1.items()):
        if method != "dro":
            continue
        lifts = v["acc_mean"] > ADULT_CONSTANT_PREDICTOR
        if lifts and abs(alpha - 0.3) < 1e-9:
            step1_dro_lifts.append((tau, v))
        L.append(f"| {alpha:.1f} | {tau:.1f} | {method} | {v['n']} "
                 f"| {_fmt(v['acc_mean'])} | {_fmt(v['dp_mean'])} "
                 f"| {'✓' if lifts else '✗'} |")
    for (alpha, tau, method), v in sorted(step1.items()):
        if method != "naive":
            continue
        L.append(f"| {alpha:.1f} | {tau:.1f} | {method} | {v['n']} "
                 f"| {_fmt(v['acc_mean'])} | {_fmt(v['dp_mean'])} | — |")
    L.append("")
    L.append("**Verdict (STEP 1):** ")
    if n_step["step1_tau"] < 72:
        L.append(f"INCOMPLETE ({n_step['step1_tau']}/72 rows) — cannot conclude yet.")
    else:
        if step1_dro_lifts:
            L.append("**Yes** — at α=0.3, the following τ values lift DRO acc above "
                     f"{ADULT_CONSTANT_PREDICTOR:.4f}:")
            for tau, v in step1_dro_lifts:
                L.append(f"- τ={tau:.1f}: acc={v['acc_mean']:.4f} (n={v['n']})")
        else:
            L.append(f"**No.** No τ ∈ {{2,5,20}} lifts α=0.3 DRO accuracy above "
                     f"{ADULT_CONSTANT_PREDICTOR:.4f}. Per-α τ tuning alone does "
                     "not rescue high-α — proceed to STEP 2.")
    L.append("")

    # ---- STEP 2 ----
    L.append("## STEP 2 — does lr_lambda=0.01 help at high α?")
    L.append("")
    L.append("lr_lambda=0.01 (canonical 5e-3), Adult, dp, α∈{0.3,0.4}, 6 seeds, "
             "2 methods. Compare to canonical rows above (τ=1.0, lr=5e-3).")
    L.append("")
    L.append("| α | method | n | acc | dp | Δacc vs canonical | Δdp vs canonical |")
    L.append("|---|---|---|---|---|---|---|")
    for (alpha, method), v in sorted(step2.items()):
        rc = ref.get((alpha, method))
        if rc is None:
            dacc_s, ddp_s = "—", "—"
        else:
            dacc_s = f"{v['acc_mean']-rc['acc_mean']:+.4f}"
            ddp_s = f"{v['dp_mean']-rc['dp_mean']:+.4f}"
        L.append(f"| {alpha:.1f} | {method} | {v['n']} "
                 f"| {_fmt(v['acc_mean'])} | {_fmt(v['dp_mean'])} "
                 f"| {dacc_s} | {ddp_s} |")
    L.append("")
    L.append("**Verdict (STEP 2):** ")
    if n_step["step2_lr_lambda"] < 24:
        L.append(f"INCOMPLETE ({n_step['step2_lr_lambda']}/24 rows) — cannot conclude yet.")
    else:
        # Did DRO acc at α=0.3 with lr=0.01 beat 0.7521 OR beat canonical acc?
        dro03 = step2.get((0.3, "dro"))
        ref03 = ref.get((0.3, "dro"))
        if dro03 is None:
            L.append("No DRO α=0.3 cell found.")
        else:
            acc = dro03["acc_mean"]
            beats_const = acc > ADULT_CONSTANT_PREDICTOR
            if ref03 is not None and acc > ref03["acc_mean"]:
                L.append(f"lr_lambda=0.01 **raises** DRO α=0.3 acc to {acc:.4f} "
                         f"(canonical {ref03['acc_mean']:.4f}); "
                         + (f"and clears 0.7521 — a lift." if beats_const
                            else "but still below 0.7521 (constant predictor)."))
            elif ref03 is not None:
                L.append(f"lr_lambda=0.01 does NOT raise DRO α=0.3 acc "
                         f"({acc:.4f} vs canonical {ref03['acc_mean']:.4f}); "
                         + ("it clears 0.7521, however." if beats_const
                            else "high-α is not rescued by dual-step tuning."))
            else:
                L.append(f"DRO α=0.3 acc = {acc:.4f} "
                         + ("(clears 0.7521)." if beats_const
                            else "(below 0.7521)."))
    L.append("")

    # ---- STEP 3 ----
    L.append("## STEP 3 — convergence diagnostics (epochs=200, dump_history)")
    L.append("")
    L.append(f"History JSONs present: DRO={hist_counts['dro']}, "
             f"naive={hist_counts['naive']} (only DRO history is dumped by "
             "run_fairness_pgd.py dump_history code; naive is in-process only).")
    L.append("")
    L.append("Test-set cells at epochs=200:")
    L.append("")
    L.append("| α | method | n | acc | dp | acc>0.7521? |")
    L.append("|---|---|---|---|---|---|")
    for (alpha, method), v in sorted(cells3.items()):
        L.append(f"| {alpha:.1f} | {method} | {v['n']} "
                 f"| {_fmt(v['acc_mean'])} | {_fmt(v['dp_mean'])} "
                 f"| {'✓' if v['exceeds_constant'] else '✗'} |")
    L.append("")
    L.append(f"**Q3(i): does val_loss plateau before epoch {PLATEAU_EPOCH} "
             "(underfitting) or keep decreasing (would benefit from more epochs)?**")
    L.append("")
    L.append("| α | method | n_hist | verdict | slope_early | slope_late | val_loss@1 | val_loss@60 | val_loss@end |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    plateau_summary = {}
    for (alpha, method), vlist in sorted(plateau.items()):
        n_h = len(vlist)
        if n_h == 0:
            L.append(f"| {alpha:.1f} | {method} | 0 | — | — | — | — | — | — |")
            continue
        verdicts = [v.get("verdict", "?") for v in vlist]
        from collections import Counter
        vc = Counter(verdicts)
        top, cnt = vc.most_common(1)[0]
        slope_e = _mean([v.get("slope_early") for v in vlist])
        slope_l = _mean([v.get("slope_late") for v in vlist])
        vlf = _mean([v.get("val_loss_first") for v in vlist])
        vl60 = _mean([v.get("val_loss_at_60") for v in vlist])
        vle = _mean([v.get("val_loss_last") for v in vlist])
        plateau_summary[(alpha, method)] = (top, cnt, n_h)
        L.append(f"| {alpha:.1f} | {method} | {n_h} | {top} ({cnt}/{n_h}) "
                 f"| {_fmt(slope_e, 6)} | {_fmt(slope_l, 6)} "
                 f"| {_fmt(vlf)} | {_fmt(vl60)} | {_fmt(vle)} |")
    L.append("")
    # Q3(i) verdict
    dro03_p = plateau_summary.get((0.3, "dro"))
    if dro03_p:
        top, cnt, n_h = dro03_p
        if top == "still_decreasing":
            L.append(f"**Verdict Q3(i):** DRO α=0.3 val_loss is **still decreasing** "
                     f"past epoch {PLATEAU_EPOCH} ({cnt}/{n_h} seeds) — 60 fixed "
                     "epochs UNDERFITS at high corruption; the model would benefit "
                     "from more epochs.")
        elif top == "plateau_before_60":
            L.append(f"**Verdict Q3(i):** DRO α=0.3 val_loss **plateaus before "
                     f"epoch {PLATEAU_EPOCH}** ({cnt}/{n_h} seeds) — 60 epochs is "
                     "enough; the high-α limitation is NOT underfitting.")
        else:
            L.append(f"**Verdict Q3(i):** inconclusive ({top}, {cnt}/{n_h}).")
    else:
        L.append(f"**Verdict Q3(i):** INCOMPLETE — no DRO α=0.3 history JSONs yet.")
    L.append("")

    L.append(f"**Q3(ii): does val_acc at epoch 200 exceed 0.7521?**")
    L.append("")
    L.append("| α | method | n | mean val_acc@200 | best val_acc | acc>0.7521? |")
    L.append("|---|---|---|---|---|---|")
    valacc_200_summary = {}
    for (alpha, method), vals in sorted(va200.items()):
        if not vals:
            continue
        m = _mean(vals)
        bv = _mean(best_va.get((alpha, method), []))
        beats = m > ADULT_CONSTANT_PREDICTOR
        valacc_200_summary[(alpha, method)] = (m, beats, len(vals))
        L.append(f"| {alpha:.1f} | {method} | {len(vals)} | {_fmt(m)} | {_fmt(bv)} "
                 f"| {'✓' if beats else '✗'} |")
    L.append("")
    if not valacc_200_summary:
        L.append(f"**Verdict Q3(ii):** INCOMPLETE — no val_acc@200 data yet.")
    else:
        dro03_v = valacc_200_summary.get((0.3, "dro"))
        if dro03_v:
            m, beats, n = dro03_v
            if beats:
                L.append(f"**Verdict Q3(ii):** **Yes** — DRO α=0.3 val_acc@200 = "
                         f"{m:.4f} > {ADULT_CONSTANT_PREDICTOR:.4f} (n={n}). "
                         "Longer validation-monitored training lifts DRO above "
                         "the constant predictor.")
            else:
                L.append(f"**Verdict Q3(ii):** **No** — DRO α=0.3 val_acc@200 = "
                         f"{m:.4f} ≤ {ADULT_CONSTANT_PREDICTOR:.4f} (n={n}). "
                         "Even at epoch 200 the model does not clear the "
                         "constant-predictor baseline.")
        else:
            L.append(f"**Verdict Q3(ii):** no DRO α=0.3 history present yet.")
    L.append("")

    # ---- HEADLINE ----
    L.append("## Headline")
    L.append("")
    # Use STEP 3 test-set acc (cells3) as the headline lift check
    dro03_s3 = cells3.get((0.3, "dro"))
    if n_step["step3_convergence"] < 24:
        L.append(f"INCOMPLETE ({n_step['step3_convergence']}/24 STEP 3 rows) — "
                 "headline will be set once STEP 3 lands.")
    elif dro03_s3 is None:
        L.append("STEP 3 DRO α=0.3 cell missing — INCOMPLETE.")
    else:
        acc = dro03_s3["acc_mean"]
        if acc > ADULT_CONSTANT_PREDICTOR:
            L.append(f"**STEP 3 lifts DRO α=0.3 accuracy above {ADULT_CONSTANT_PREDICTOR:.4f}** "
                     f"(acc={acc:.4f}, n={dro03_s3['n']}). The defensible regime "
                     "**EXTENDS beyond α=0.2** — a headline-level upgrade to the "
                     "paper's main claim.")
        else:
            L.append(f"**STEP 3 does NOT lift DRO α=0.3 accuracy above "
                     f"{ADULT_CONSTANT_PREDICTOR:.4f}** (acc={acc:.4f}, "
                     f"n={dro03_s3['n']}). The α≥0.3 limitation now has "
                     "**convergence evidence** (epochs=200, val-monitored) "
                     "instead of an assertion — Kuldeep's requested check is "
                     "closed with data.")
    L.append("")

    if fig_paths:
        L.append("## Convergence plots (literal artifact Kuldeep requested)")
        L.append("")
        for p in fig_paths:
            rel = os.path.relpath(p, ROOT)
            L.append(f"- `{rel}`")
        L.append("")

    return "\n".join(L)


def main():
    print("AGENT N2: high-α rescue summary (analysis-only)")
    print("=" * 78)
    rows, source = _load_rows()
    print(f"Loaded: {source}")
    n_expected = 120
    if len(rows) < n_expected:
        print(f"  INCOMPLETE: {len(rows)}/{n_expected} rows "
              f"({100.0*len(rows)/n_expected:.1f}%) — partial-data mode.")

    step1 = _step1_table(rows)
    step2 = _step2_table(rows)
    ref = _canonical_adult_reference()
    step3_data = _step3_summary(rows)
    cells3, plateau, va200, best_va, hist_counts = step3_data
    print(f"  STEP 1 cells: {len(step1)} (α, τ, method)")
    print(f"  STEP 2 cells: {len(step2)} (α, method)")
    print(f"  STEP 3 cells: {len(cells3)} (α, method); "
          f"history JSONs: DRO={hist_counts['dro']} naive={hist_counts['naive']}")

    # Convergence plots — literal artifact Kuldeep asked for.
    fig_paths = _plot_convergence(alpha=0.3, method_filter=("dro",))
    if fig_paths:
        for p in fig_paths:
            print(f"  wrote {p}")

    md_text = write_md(rows, source, step1, step2, ref, step3_data, fig_paths)
    with open(OUT_MD, "w") as f:
        f.write(md_text)
    print(f"  saved {OUT_MD}")

    print("\n" + "=" * 78)
    print("AGENT N2 SUMMARY MILESTONE (analysis-only).")
    print(f"  output: {OUT_MD}")
    print("=" * 78)


if __name__ == "__main__":
    main()