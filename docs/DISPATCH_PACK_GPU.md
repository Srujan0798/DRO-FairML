# GPU DISPATCH PACK — flair2 (2× NVIDIA L40S, verified working 2026-08-04 23:xx)

Companion to docs/DISPATCH_PACK.md. GPU gate passed: `torch.cuda.is_available()=True`,
2× L40S visible, real matmul verified on-device. `venv_gpu` is built at
`/data/srujan.sai/DRO-FairML-run/venv_gpu` on flair2, offline (no PyPI needed there).

**Mac status (unaffected by any of this, keep running):** tabular UTKFace-equivalent
work is on CPU via the sequential orchestrator + lock (`scripts/orchestrate_wave1.sh`).
Local UTKFace canonical is ALREADY COMPLETE: 90/90 REAL rows
(`results/utkface_canonical.json`, dp/if/combined = 30/30/30). **Do not repeat that
grid on GPU** — the GPU work below is either (a) a reproducibility check against it, or
(b) genuinely NEW experiments only a GPU-scale machine can do.

## GLOBAL RULES for every GPU agent

```
1. SSH: `ssh flair2` (key auth already works, no password). Working dir on flair2:
   /data/srujan.sai/DRO-FairML-run
2. FIRST STEP, ALWAYS: sync fresh code from Mac before running anything (flair2 has
   no internet/git access — code only arrives via rsync):
     rsync -az --exclude='.git' --exclude='data/raw' --exclude='figures' \
       --exclude='logs' --exclude='results' --exclude='venv_gpu' --exclude='paper' \
       --exclude='report' /Users/srujansai/Desktop/DRO-FairML/ \
       flair2:/data/srujan.sai/DRO-FairML-run/
3. Run everything through venv_gpu: `./venv_gpu/bin/python ...` — never system python3.
4. NEVER write results/canonical_tau1.json or results/utkface_canonical.json (locked,
   same rule as the Mac ablations). All GPU output goes to NEW result files, then
   rsync those files back to the Mac (flair2 is not backed up / not the source of
   truth repo).
5. device='cuda' explicitly in every run — do not rely on 'auto' silently picking CPU.
6. Tag every row with device='cuda' and data_provenance='REAL' (never synthetic).
7. flair2 is a SHARED cluster (other users' Jupyter kernels are running there) — be a
   good citizen: don't max out both GPUs for hours without checking `nvidia-smi` first
   for other users' usage, and prefer one GPU per job (CUDA_VISIBLE_DEVICES=0 or =1) so
   two GPU jobs can run at once without colliding.
8. Full provenance on every row (same fields as the Mac ablations).
9. Deliverable = data file + summary.md, not data alone.
10. Copy results back to the Mac repo when done:
      rsync -az flair2:/data/srujan.sai/DRO-FairML-run/results/<new_file>.json \
        /Users/srujansai/Desktop/DRO-FairML/results/
```

---

## AGENT U1 — Reproducibility: same UTKFace grid, GPU vs Mac MPS

```
Working dir: flair2:/data/srujan.sai/DRO-FairML-run (ssh flair2 first). Read
DISPATCH_PACK_GPU.md GLOBAL RULES.

Sync code (step 2 above). Then run the IDENTICAL canonical UTKFace protocol that
already completed on the Mac (tau=1, k_inner=10, epochs=60, pgd_steps=20, 6 seeds,
attacks dp/if/combined, alphas 0.0-0.4) but on CUDA:

  CUDA_VISIBLE_DEVICES=0 ./venv_gpu/bin/python experiments/run_utkface_server.py \
    --attacks dp if combined --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 \
    --tau 1.0 --k_inner 10 --epochs 60 --pgd_steps 20 --device cuda \
    --output results/utkface_flair2.json

90 configs. Should be materially faster than the Mac's MPS run (real GPU vs Apple
unified memory). Rsync results/utkface_flair2.json back to the Mac.

Deliverable: results/utkface_reproducibility_summary.md — for each (attack, alpha),
compare CPU/MPS (Mac) vs CUDA (flair2) accuracy/DP/IF: same seeds should produce
near-identical numbers (small float/nondeterminism differences expected, large
differences are a bug to report, not to hide). This is an appendix-worthy
reproducibility row AND makes Manisha's original May-19 "in the server" ask literally
true for the first time. Commit + push from the Mac side once files are back.
```

## AGENT U2 — Multi-group UTKFace (5-race, not binary)

```
Working dir: flair2:/data/srujan.sai/DRO-FairML-run. Read GLOBAL RULES + protocol
Part 1B D8.

UTKFace race is currently collapsed to binary White/non-White for the canonical run.
src/evaluation/metrics.py ALREADY computes max-min DP for >2 protected groups — this
just needs the raw 5-category race label wired through instead of the binary collapse
in the UTKFace data loader (check src/data/datasets.py or wherever UTKFace does the
binarization; UTKFace's race field is 0-4: White/Black/Asian/Indian/Other).

Run: CUDA_VISIBLE_DEVICES=1 ./venv_gpu/bin/python experiments/run_utkface_server.py \
  --attacks dp --alphas 0.0 0.1 0.2 0.3 0.4 --n_seeds 6 --tau 1.0 --k_inner 10 \
  --epochs 60 --pgd_steps 20 --device cuda --protected race_multigroup \
  --output results/utkface_multigroup.json
(add a --protected flag to the runner if it doesn't exist yet, defaulting to the
current binary behavior so nothing else breaks)

60 configs (5 alphas x 6 seeds x 2 methods). Deliverable:
results/utkface_multigroup_summary.md — does DRO's fairness advantage hold under
5-group max-min DP, or does it concentrate/vanish for specific minority groups (e.g.
Indian/Other, which are smallest in UTKFace)? This is likely the most novel single
result available to add to the paper — multi-group fairness under adversarial
corruption at this scope is not something a typical course project has. Rsync back,
commit, push.
```

## AGENT U3 — Pixel-space PGD (the "only a GPU can do this" stretch)

```
Working dir: flair2:/data/srujan.sai/DRO-FairML-run. Read GLOBAL RULES.

Everything so far attacks CACHED FEATURES (ResNet18 embeddings). This attacks the RAW
IMAGES through the network — a genuinely different, more expensive threat model, and
the reason a GPU matters here (feature-space attacks run fine on a laptop; pixel-space
does not).

1. Recover the pixel-PGD code from git history (it existed, was archived, and — per
   docs/reference/ARCHIVE_POLICY.md's own cautionary tale — GREP for references before
   assuming it's dead):
     git log --oneline --all -- src/corruption/image_pgd.py
     git show 3c371a8:src/corruption/image_pgd.py > src/corruption/image_pgd.py
   (commit 3c371a8 = "Fix image_pgd device mismatch + add greedy attack superiority
   test" — take that version, it's the fixed one, not an earlier buggy one. Also check
   commit a31d43f "Add UTKFace pipeline: CNN classifier, ImagePGD attack, and
   experiment runner" for the matching CNN classifier / experiment runner scaffolding
   that went with it.)
2. Wire it into src/corruption/__init__.py exports if not already there.
3. Scope: α ∈ {0.1, 0.2}, 6 seeds, dp attack, 2 methods, on raw UTKFace JPEGs
   (data/raw/utkface/UTKFace/ — 23,708 real images, already on flair2) through a
   ResNet18 CNN classifier (not the cached-feature MLP) = 24 configs.
   CUDA_VISIBLE_DEVICES=0 or 1, whichever is free.
   → results/utkface_pixel_pgd.json

Deliverable: results/pixel_pgd_summary.md comparing feature-space attack (existing
utkface_canonical.json, dp, same alphas) vs pixel-space attack (this run) — does DRO's
advantage hold when the adversary can perturb raw pixels instead of just flipping
labels/features? Gives the paper a genuine "feature-space vs pixel-space" section.
If the CNN pipeline needs real debugging time, timebox to ~half a day and report
partial results honestly rather than block everything else. Rsync back, commit, push.
```

## AGENT U4 — CelebA second image modality (stretch, only if U1-U3 land early)

```
Working dir: flair2:/data/srujan.sai/DRO-FairML-run. Only start this if U1, U2, U3 are
done or clearly on track to finish with time to spare — cut without guilt otherwise.

Add a CelebA loader (public dataset, download via torchvision.datasets.CelebA or a
direct URL) following the same pattern as UTKFace: task = gender or attractiveness
classification, protected attribute = Young (binary attribute already in CelebA
annotations). Extract ResNet18 features (same pipeline as UTKFace feature extraction),
then run the canonical protocol: 3 attacks x 5 alphas x 6 seeds x 2 methods = 90 configs
→ results/celeba_canonical.json.

This would give the paper a SECOND independent image dataset — meaningfully
strengthens any image-modality claim beyond "we tried one dataset." Full provenance,
data_provenance='REAL'. Deliverable: results/celeba_summary.md matching the UTKFace
summary format. Rsync back, commit, push.
```

---

## Sequencing

```
NOW:  U1 (reproducibility) and U2 (multi-group) can run in parallel — put U1 on
      CUDA_VISIBLE_DEVICES=0, U2 on CUDA_VISIBLE_DEVICES=1, so both L40S are used
      without one job blocking the other.
NEXT: U3 (pixel-PGD) — needs code recovery + wiring first, run after a GPU frees up.
LAST: U4 (CelebA) — only with time to spare.
```

Every deliverable rsyncs back to the Mac and gets committed there — flair2 itself is
not backed up and is not where `git push` should happen from (no PyPI/git access
without the same SSL-firewall issue that blocked torch).
