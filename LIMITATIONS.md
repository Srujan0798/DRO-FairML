# Limitations

This document lists documented divergences from the paper's Algorithm 1 specification.

## 1. τ Schedule

**Paper:** τ=100 for all α (Table 1 evaluation).
**Implementation:** τ=1 for training (better gradient flow), τ=100 for α≤0.3 / τ=1 for α≥0.4 at evaluation per §G.6.

This affects the sharpness of h̃ = σ(τ·f_θ(x)) during training and evaluation.

## 2. Full-Batch vs Minibatch

**Paper:** Algorithm 1 uses minibatch SGD with batch size B.
**Implementation:** Full-batch training (all data per epoch).

Full-batch ensures fairness gradients flow correctly across the full dataset but loses minibatch noise regularization.

## 3. λ Scalar Update vs Adam

**Paper:** λ updated via scalar projected gradient descent (Algorithm 1, line λ ← Π_C(λ + ηλ·∇λ)).
**Implementation:** λ updated as scalar with constant lr_lambda, no Adam adaptation.

No per-parameter or per-constraint adaptation of the dual learning rate.

## 4. Projection Postcondition on Adversarial Inputs

**Paper:** Dykstra's alternating projection terminates when constraints are satisfied.
**Implementation:** For adversarial/out-of-distribution inputs, the simplex∩L1 intersection may be infeasible. The tail loop of 50 iterations produces a point exactly on the simplex but only ε-close to the L1 ball (admissibility ε ≈ 1e-9).

Training-regime inputs (near uniform distributions) are unaffected.

## 5. τ Warmup

**Paper:** No warmup specified.
**Implementation:** τ=1 for first `tau_warmup_epochs` (default 5), then τ=default (1.0 for our config). This is a no-op in current config since default τ=1.0 and warmup keeps τ=1.0.

## 6. Runtime Overhead

**Paper:** Claims ~12× DRO vs Naive runtime overhead.
**Implementation:** ~1.28× observed overhead.

The discrepancy arises because: (a) our inner maximization is numpy-backed with lightweight projections, not GPU tensor ops; (b) K=10 inner steps are a small fraction of total compute on CPU; (c) the paper's 12× may reflect GPU-specific projection implementations.