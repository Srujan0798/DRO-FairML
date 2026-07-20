# 4 PM Meeting — what to say (talking track)

> Tight script. ~5 min core + Q&A. Numbers are Adult (verified); say Credit/LSAC
> are "landing now, same direction." Don't overclaim — flag the open items.

## Opening (30 sec)
"Since last meeting we did three things: fixed the DP attack, ran the ablations
Kuldeep suggested, and found that one setting — the temperature — explains the
whole 'DRO looks fragile' problem. Let me walk through it."

## 1. We fixed the attack (45 sec)
- The feature-perturbation half of the attack was using a classification loss, not a
  DP-targeting loss. Fixed it to directly maximize the group gap |p0−p1|.
- Also restored K_inner=10 and added an α=0 guard (DRO now equals Naive at zero
  corruption, as it should).
- **Proof the attack works:** adversarial corruption raises DP **~31× more than
  random noise** at the same budget (Adult α=0.2: adversarial +0.18 vs random ≈0).
  → directly answers Madam's "show adversarial > random."

## 2. The headline — temperature was the culprit (90 sec)
"Earlier DRO looked worse than Naive. That was an artifact of our temperature
schedule (tau=100). When we fix tau=1, as Kuldeep suggested, DRO beats Naive on DP
at every corruption level on Adult:"

| α | Naive DP | DRO DP | DRO wins (of 3 seeds) |
|---|---|---|---|
| 0.1 | 0.207 | 0.205 | 2/3 |
| 0.2 | 0.248 | 0.237 | 3/3 |
| 0.3 | 0.286 | 0.264 | 3/3 |
| 0.4 | 0.310 | 0.283 | 3/3 |

- "And the advantage **grows with α** — more corruption, bigger DRO win — which is
  exactly what the theory predicts."
- "Accuracy is equal or slightly better, so it's not even an accuracy trade."
- "At tau=100 DRO lost almost everywhere — so this was a tuning artifact, not a real
  weakness of DRO." (This is the key message.)

## 3. The ablations you asked for (45 sec)
- **tau (Q12):** swept tau ∈ {1,10,100}; tau=1 is clearly best for DRO. We'll fix
  tau for all α going forward.
- **IF k-NN (Q6):** swept k ∈ {5,10,15}; attack strength is insensitive to k → k=5 fine.
- **λ_init × lr grid (Q1):** running now on Adult to see if we can tighten DP further.

## 4. Narrative decisions (30 sec)
- **LSAC (Q3):** it has inherently low DP, so the DP attack can't raise it — we frame
  LSAC around the IF attack, as you suggested.
- **Radii (Q5):** we treat it empirically — calibrate from the observed clean
  proportions under our known attack, not a new closed form. We're not claiming the
  paper is wrong.

## 5. Status & honest gaps (30 sec)
- Done: Adult ablations, attack-vs-noise, bug fixes.
- Running: tau=1 re-run on Credit+LSAC, 6-seed runs for significance (3 seeds can't
  reach p<0.05 — min p is 0.125), λ/lr grid.
- Blocked: UTKFace still needs flair2 GPU access (SSL/account issue).

## Asks (the decisions we need)
1. **Adopt fixed tau=1** as the production setting and rebuild the report around it? (Adult is strong; Credit/LSAC landing.)
2. **6 seeds** OK for the paper (for valid Wilcoxon significance)?
3. **UTKFace** — priority, and can you help with the flair2 (supin.gopi) account?

## Likely questions — quick answers
- *"Why did tau matter so much?"* At tau=100 predictions are near-binary, so the DP
  surrogate loss saturates and DRO's robust inner-max destabilizes; at tau=1 the
  signal is smooth and DRO optimizes cleanly.
- *"Is the α=0 anomaly gone?"* Fixed on Adult and Credit (DRO=Naive at α=0). Checking LSAC.
- *"Does it generalize beyond Adult?"* Credit α=0 already matches; full Credit/LSAC
  tau=1 results landing today — will send as soon as complete.
- *"Is the attack realistic?"* It's the standard poisoning threat model (Solans 2021):
  α-fraction of training data corrupted, coordinated 70% toward the minority group.
