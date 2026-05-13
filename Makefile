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
	@echo "  review        - Run professor review simulator (self-grade)"
	@echo "  full          - Run experiments then generate results"
	@echo "  clean         - Remove Python cache files"

install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest tests/ -v

monitor:
	python3 experiments/monitor_progress.py

experiments:
	python3 experiments/run_experiments.py --n_seeds 10

results:
	python3 main.py --generate-results

deliverables:
	python3 experiments/generate_all_deliverables.py

review:
	python3 experiments/professor_review_simulator.py

full:
	python3 main.py --full-pipeline

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pth" -delete
	find . -type f -name "*.pt" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
