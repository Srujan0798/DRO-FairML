.PHONY: help install data test monitor validate theory experiments results deliverables review paper report full clean

help:
	@echo "DRO-FAIR Project"
	@echo ""
	@echo "Available targets:"
	@echo "  install       - pip install -r requirements.txt"
	@echo "  data          - Download tabular datasets (Adult/Credit/LSAC) via data/download_data.sh"
	@echo "  test          - Run unit tests (pytest tests/ -v)"
	@echo "  monitor       - List results/*.json count; hint for live canonical watcher"
	@echo "  validate      - Wilcoxon / result checks (experiments/validate_results.py)"
	@echo "  theory        - Theory verification script (experiments/verify_theory.py)"
	@echo "  experiments   - Legacy full suite via run_experiments.py (n_seeds=10; not the canonical grid)"
	@echo "  results       - Generate tables/plots from existing results (main.py --generate-results)"
	@echo "  deliverables  - All deliverables: tables + plots + extras (generate_all_deliverables.py)"
	@echo "  review        - Print paths to archived self-review / verification checklists"
	@echo "  paper         - Build paper/main.pdf with tectonic"
	@echo "  report        - Build report/report.pdf with tectonic"
	@echo "  full          - main.py --full-pipeline (legacy experiments then generate-results)"
	@echo "  clean         - Remove __pycache__, *.pyc, *.pth/*.pt, .pytest_cache"
	@echo ""
	@echo "Canonical tabular grid (preferred over 'make experiments'):"
	@echo "  python3 experiments/run_canonical.py          # tau=1, K=10, 6 seeds, 3 attacks -> results/canonical_tau1.json"
	@echo "  python3 experiments/run_if_parallel.py 10     # resume-safe IF third only (parallel workers)"
	@echo "  Committed DP+Combined rows already live in results/canonical_tau1.json; IF may still be filling."
	@echo "  ./scripts/agent_h_finalize.sh                 # ONLY after total=540 / IF=180 (read-only on JSON)"
	@echo "  docs/LOOP_STATUS.md                           # live completion-loop snapshot"
	@echo ""
	@echo "Reproduce path: make install && make data && make test && make validate && make paper && make report"

install:
	python3 -m pip install -r requirements.txt

data:
	bash data/download_data.sh --verify

test:
	python3 -m pytest tests/ -v

monitor:
	@python3 -c "import json,glob,os; f=sorted(glob.glob('results/*.json')); print('result files:', len(f))" 2>/dev/null || true
	@echo "Live watcher: python3 scripts/canonical_watcher.py"
	@echo "IF progress:  python3 -c \"import json,collections;d=json.load(open('results/canonical_tau1.json'));print(len(d),dict(collections.Counter(r['attack'] for r in d)))\""

validate:
	python3 experiments/validate_results.py

theory:
	python3 experiments/verify_theory.py

experiments:
	python3 experiments/run_experiments.py --n_seeds 10

results:
	python3 main.py --generate-results

deliverables:
	python3 experiments/generate_all_deliverables.py

review:
	@echo "Self-review:    docs/_archive/REVIEW_CHECKLIST.md"
	@echo "Verification:   docs/_archive/VERIFICATION_PROTOCOL.md"
	@echo "Release check:  docs/_archive/RELEASE_CHECKLIST.md"

paper:
	tectonic -X compile paper/main.tex

report:
	tectonic -X compile report/report.tex

full:
	python3 main.py --full-pipeline

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pth" -delete
	find . -type f -name "*.pt" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
