# QUICK CARD — May 29, 3:00 PM Meeting
**Print this or keep it open on a second screen**

---

## 🎯 OPENING LINE (say this first)

> *"Madam, both tasks you assigned are complete. Task 1 — PGD attacks on tabular data — 270 experiments done. Task 2 — UTKFace on GPU server — experiments done with real ResNet18 features. We found a surprising result."*

Then **open `figures/final_meeting_figure.png`** on screen.

---

## 📊 THE ONE FIGURE TO SHOW

**`figures/final_meeting_figure.png`**

| Section | Point to | Say |
|---------|----------|-----|
| Top-left green bars | Credit IF 97.5% | "DRO reduces DP violation by 64-97% under IF attacks, statistically significant" |
| Top-right bars | α=0.1 red vs green | "But on UTKFace images, DRO makes fairness WORSE — Naive wins by 39%" |
| Bottom yellow box | Read the text | "This tells us DRO robustness is not universal — it depends on attack metric AND data type" |

---

## 🔢 NUMBERS TO REMEMBER

| Task | Key Number | What it means |
|------|-----------|---------------|
| 1 | **97.5%** | Credit IF α=0.3: DRO's best result |
| 1 | **0.031** | p-value (statistically significant) |
| 2 | **39%** | At α=0.1, Naive is 39% better than DRO on images |
| 2 | **23,705** | UTKFace images used |

---

## ❓ LIKELY QUESTIONS & ANSWERS

**"Did you change the submission?"**
> No. ICML submission frozen. This is for a follow-up paper.

**"Why does DRO fail on images?"**
> ResNet features are fairness-agnostic. DRO overfits to corrupted fairness signal. On tabular data, features naturally carry demographic info.

**"What's next?"**
> (1) More UTKFace seeds running now on GPU, (2) CelebA/FairFace larger datasets, (3) ResNet50 deeper features, (4) Paper draft for NeurIPS/ICLR.

**"Only 9 experiments on UTKFace?"**
> 3 seeds × 3 alphas. 5 seeds are running on the server right now — results will be stronger.

---

## 📁 FILES READY

```
figures/final_meeting_figure.png          ← SHOW THIS FIRST
figures/summary_dashboard_may29.png       ← backup visual
docs/MEETING_CHEAT_SHEET.md               ← full talking points
src/corruption/adversarial.py             ← code if asked
results/utkface_results.json              ← raw data
GitHub: Srujan0798/DRO-FairML @ a833620   ← backup proof
```

---

## ✅ CLOSING LINE

> *"Madam, to summarize: Task 1 complete — DRO defends well against IF attacks on tabular data. Task 2 complete — UTKFace shows DRO hurts under corruption on image features, opposite of tabular. We are running more seeds now and ready for the next phase. No submission code was modified."*

---

*Server: flair2.iitgn.ac.in | GPU: 2× NVIDIA L40S 48GB | GitHub pushed: a833620*
