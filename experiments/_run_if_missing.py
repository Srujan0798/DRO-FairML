
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.run_fairness_pgd import run_single_experiment

spec_path = sys.argv[1]
with open(spec_path) as f:
    specs = json.load(f)

results_path = "results/canonical_tau1.json"
with open(results_path) as f:
    all_results = json.load(f)
existing = {(r['dataset'],r['alpha'],r['seed'],r['attack'],r['method']) for r in all_results}

for spec in specs:
    ds, a, s, m = spec['dataset'], spec['alpha'], spec['seed'], spec['method']
    if (ds, a, s, 'if', m) in existing:
        print(f"SKIP {ds} a={a} s={s} {m}")
        continue
    t0=time.time()
    try:
        result = run_single_experiment(ds, a, s, 'if', m, device='cpu', verbose=False,
                                       epochs=60, k_inner=10, pgd_steps=20,
                                       tau=1.0, lambda_init=0.0, radii_mode='uniform',
                                       coordinated=False, n_seeds_planned=6)
        elapsed = time.time()-t0
        all_results.append(result)
        with open(results_path,'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"DONE {ds} a={a} s={s} {m} acc={result['acc_clean']:.3f} if={result['if_clean']:.4f} ({elapsed:.0f}s)")
    except Exception as e:
        print(f"FAIL {ds} a={a} s={s} {m}: {e}")
