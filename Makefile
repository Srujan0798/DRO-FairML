.PHONY: help install test demo experiments results clean

help:
	@echo "DRO-FAIR Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install Python dependencies"
	@echo "  test         - Run unit tests"
	@echo "  demo         - Run quick demo (single dataset, single seed)"
	@echo "  experiments  - Run full experiment suite"
	@echo "  results      - Generate tables and plots from existing results"
	@echo "  full         - Run experiments then generate results"
	@echo "  clean        - Remove generated files and caches"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v || python test_quick.py

demo:
	python scripts/demo.py --dataset adult --alpha 0.2 --epochs 20

experiments:
	python main.py --run-experiments

results:
	python main.py --generate-results

full:
	python main.py --full-pipeline

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pth" -delete
	find . -type f -name "*.pt" -delete
	rm -rf results/ figures/ data/raw/ .pytest_cache/
