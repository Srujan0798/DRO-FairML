# MEETING PREP - FOR FRIEND

## FILES TO HAVE READY (Send to friend)

1. `results/output_*.txt` - Latest experiment results
2. `src/corruption/adversarial.py` - FairnessTargetedPGD code  
3. `docs/FAIRNESS_PGD_DESIGN.md` - Design explanation
4. `src/data/datasets.py` - Dataset info

---

## ONE LINE PROMPT FOR AI (Meeting helper):

```
Srujan has a meeting with Madam. Help friend answer questions. 
Repo: https://github.com/Srujan0798/DRO-FairML. 
Server: /data/srujan.sai/DRO-FairML. 
Run: python3 scripts/test_fairness_pgd.py 
Then answer: What did we do? How does FairnessTargetedPGD work? What results? What's next?
```

---

## WHAT WE DID (2-3 sentences):

We extended our DRO-FAIR course project with a new gradient-based attack called FairnessTargetedPGD. 
This attack computes the exact gradient of Demographic Parity (DP) with respect to each training sample's label, 
then flips the samples that maximally increase unfairness. We tested it on Adult, Credit, and LSAC datasets 
showing significant DP increase (+0.08 on Adult, +0.03 on LSAC).

---

## HOW THE ATTACK WORKS (Simple):

1. Train model normally → measure DP = |P(Y=1|A=0) - P(Y=1|A=1)|
2. Compute gradient: which samples, if flipped, would MOST increase DP
3. Flip top 20% of samples by gradient
4. Retrain → DP goes up (unfairness increases)

Key insight: We don't use rules. We use actual gradient optimization like PGD uses for adversarial examples.

---

## RESULTS:

| Dataset | Clean DP | After Attack DP | Increase |
|---------|----------|-----------------|----------|
| Adult   | ~0.14    | ~0.22           | +0.08    |
| Credit  | ~0.01    | ~0.02           | +0.01    |
| LSAC    | ~0.02    | ~0.05           | +0.03    |

Attack successfully increases unfairness on all datasets.

---

## NEXT STEPS:

1. UTKFace image dataset - extract ResNet18 features, test on images
2. DRO-FAIR defense - train with fairness constraints to resist attack
3. Compare with other attacks (random, heuristic)

---

## TECHNICAL DETAILS (if asked):

- Gradient: d(DP)/d(y_i) computed analytically per sample
- Coordinated: 70% corruption budget targets minority group
- alpha = 0.2 (20% of samples corrupted)
- Model: MLP with [128, 64] hidden dims, Adam optimizer
- Datasets: Adult (29K), Credit (19.5K), LSAC (12K) samples

---

## IF MADAM ASKS SPECIFIC QUESTIONS:

Q: Why gradient-based better than heuristic?
A: Heuristic uses rules (e.g., flip minority 0→1), but gradient tells us EXACTLY which flip causes most unfairness increase. Mathematically optimal.

Q: What is DP?
A: Demographic Parity - difference in positive prediction rates between protected groups. DP=0 means fair, DP=1 means completely unfair.

Q: What's the attack target?
A: Maximize DP - make model's predictions as unequal as possible across gender/race groups.

Q: Defense against this attack?
A: DRO-FAIR training - add fairness constraints to training objective. Future work.

---

## QUICK COMMAND TO RUN EXPERIMENTS:

```bash
cd /data/srujan.sai/DRO-FairML
python3 scripts/test_fairness_pgd.py
```

---

## UNDERSTANDING THE CODE:

FairnessTargetedPGD class (src/corruption/adversarial.py):
- compute_dp_gradient(y, a): For each sample i, computes d(DP)/d(y_i)
- Flips samples with largest positive gradient = most unfairness increase
- coordinated=True: 70% of flips target minority group

MLPClassifier (src/models/classifier.py):
- Simple feedforward network: input → 128 → 64 → 1
- Used for all experiments