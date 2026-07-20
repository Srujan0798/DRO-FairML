.PHONY: help install test monitor experiments results deliverables review paper report clean

help:
	@echo "DRO-FAIR Project"
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install Python dependencies"
	@echo "  test          - Run unit tests"
	@echo "  monitor       - Check experiment progress"
	@echo "  experiments   - Run full experiment suite (150 exps)"
	@echo "  results       - Generate tables and plots from existing results"
	@echo "  deliverables  - Generate ALL deliverables (tables + plots + ablations + theory)"
	@echo "  review        - Open self-review checklist (docs/_archive/REVIEW_CHECKLIST.md)"
	@echo "  paper         - Build the paper PDF with tectonic (paper/main.tex)"
	@echo "  report        - Build the report PDF with tectonic (report/report.tex)"
	@echo "  full          - Run experiments then generate results"
	@echo "  clean         - Remove Python cache files"

install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest tests/ -v

monitor:
	@python3 -c "import json,glob,os; f=sorted(glob.glob('results/*.json')); print('result files:', len(f))" 2>/dev/null || true
	@echo "Live watcher: python3 scripts/canonical_watcher.py"

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
