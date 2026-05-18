.PHONY: help install test monitor experiments results deliverables review clean

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
	@echo "  review        - Open self-review checklist (docs/REVIEW_CHECKLIST.md)"
	@echo "  full          - Run experiments then generate results"
	@echo "  clean         - Remove Python cache files"

install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest tests/ -v

monitor:
	@echo "Monitor script removed. Use: python3 -c \"import json; d=json.load(open('results/all_results.json')); print(len(d), '/150 experiments')\""

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
	@echo "Self-review: docs/REVIEW_CHECKLIST.md"
	@echo "Verification: docs/VERIFICATION_PROTOCOL.md"
	@echo "Release check: docs/RELEASE_CHECKLIST.md"

full:
	python3 main.py --full-pipeline

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pth" -delete
	find . -type f -name "*.pt" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
