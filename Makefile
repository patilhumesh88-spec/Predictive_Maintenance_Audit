.PHONY: setup data train evaluate test lint app verify reproduce clean

setup:  ## Create venv and install
	python -m venv .venv
	.venv/Scripts/pip install -e ".[all]"

data:  ## Download dataset (checksum-validated)
	python scripts/download_data.py

train:  ## Train model and save artifact
	python -m maint_audit.train --holdout tail --fit-all

evaluate:  ## Generate metrics.json (single source of truth)
	python scripts/generate_metrics.py

test:  ## Run pytest
	python -m pytest tests/ -v

lint:  ## Run ruff linter
	python -m ruff check maint_audit/ tests/ scripts/ dashboard/

app:  ## Launch Streamlit dashboard
	python -m streamlit run dashboard/app.py

verify:  ## Run full verification suite
	python scripts/verify_all.py

reproduce:  ## Full pipeline: data → train → evaluate → test → verify
	$(MAKE) data
	$(MAKE) train
	$(MAKE) evaluate
	$(MAKE) test
	$(MAKE) verify

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
