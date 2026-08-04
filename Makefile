.PHONY: help install data test monitor validate wilcoxon tables results deliverables \
        theory review paper report full clean experiments

help:
	@echo "DRO-FairML"
	@echo ""
	@echo "Setup & checks"
	@echo "  install       pip install -r requirements.txt"
	@echo "  data          download + SHA-256 verify tabular datasets"
	@echo "  test          pytest tests/ -v"
	@echo "  monitor       print canonical_tau1.json row counts (expect 540)"
	@echo "  validate      consistency checks (experiments/validate_results.py)"
	@echo "  wilcoxon      paired Wilcoxon from canonical JSON → results/canonical_wilcoxon.*"
	@echo ""
	@echo "Artifacts from committed results (no training)"
	@echo "  tables        regenerate report/paper LaTeX table fragments"
	@echo "  results       tables + plots (main.py --generate-results)"
	@echo "  deliverables  full deliverable pack (generate_all_deliverables.py)"
	@echo "  paper         build paper/main.pdf (tectonic)"
	@echo "  report        build report/report.pdf (tectonic)"
	@echo "  full          regenerate artifacts only: wilcoxon + tables + results + deliverables"
	@echo "                (does NOT retrain; uses results/canonical_tau1.json)"
	@echo ""
	@echo "Other"
	@echo "  theory        theory verification script"
	@echo "  review        paths to archived review checklists"
	@echo "  clean         remove __pycache__, *.pyc, checkpoints, .pytest_cache"
	@echo "  experiments   DEPRECATED legacy train driver (n_seeds=10) — not canonical"
	@echo ""
	@echo "Canonical grid (optional retrain; results already committed at 540 rows):"
	@echo "  python3 experiments/run_canonical.py"
	@echo "  python3 experiments/run_canonical.py --smoke"
	@echo ""
	@echo "Default repro (no retrain):"
	@echo "  make install && make data && make test && make validate && make paper && make report"

install:
	python3 -m pip install -r requirements.txt

data:
	bash data/download_data.sh --verify

test:
	python3 -m pytest tests/ -v

# Print canonical row counts: total + per-attack breakdown.
monitor:
	@python3 -c "import json,collections,os,sys; p='results/canonical_tau1.json'; \
print('=== DRO-FairML monitor ==='); \
assert os.path.isfile(p), 'MISSING '+p; \
d=json.load(open(p)); atk=collections.Counter(r.get('attack') for r in d); \
ds=collections.Counter(r.get('dataset') for r in d); \
print('canonical_tau1.json: total=%d  (expect 540)'%len(d)); \
print('  by attack:', dict(sorted(atk.items()))); \
print('  by dataset:', dict(sorted(ds.items()))); \
ok=(len(d)==540 and all(atk.get(a,0)==180 for a in ('dp','if','combined'))); \
print('status:', 'COMPLETE' if ok else 'INCOMPLETE'); \
print('other results/*.json:', sum(1 for f in os.listdir('results') if f.endswith('.json')))"

validate:
	python3 experiments/validate_results.py

wilcoxon:
	PYTHONPATH=. python3 experiments/compute_canonical_wilcoxon.py

tables:
	PYTHONPATH=. python3 experiments/generate_report_tables.py

results:
	python3 main.py --generate-results

deliverables:
	PYTHONPATH=. python3 experiments/generate_all_deliverables.py

theory:
	python3 experiments/verify_theory.py

review:
	@echo "Live audit:     docs/VERIFICATION_REPORT.md"
	@echo "Meeting brief:  docs/MEETING_2026-08-04.md"
	@echo "Status:         STATUS.md"
	@echo "Doc index:      docs/INDEX.md"

paper:
	tectonic -X compile paper/main.tex

report:
	tectonic -X compile report/report.tex

# Regenerate derived artifacts from committed canonical JSON — does NOT retrain.
full: wilcoxon tables results deliverables
	@echo ""
	@echo "full: regenerated wilcoxon + tables + results + deliverables (no training)."
	@echo "Build PDFs with: make paper && make report"

# DEPRECATED: legacy experiment suite (n_seeds=10 default in main.py).
# Not the canonical τ=1 / k=10 / n=6 grid. Prefer committed results or run_canonical.py.
experiments:
	@echo "WARNING: 'make experiments' is DEPRECATED."
	@echo "  It runs the legacy driver (experiments/run_experiments.py, n_seeds=10),"
	@echo "  which is NOT the canonical 540-row grid (τ=1, K=10, n=6)."
	@echo "  For claims, use results/canonical_tau1.json (already committed)."
	@echo "  Optional retrain: python3 experiments/run_canonical.py"
	@echo "  Refusing to launch full retrain by default. Pass FORCE_LEGACY=1 to override."
	@if [ "$(FORCE_LEGACY)" = "1" ]; then \
		python3 experiments/run_experiments.py --n_seeds 10; \
	else \
		exit 1; \
	fi

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pth" -delete
	find . -type f -name "*.pt" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
