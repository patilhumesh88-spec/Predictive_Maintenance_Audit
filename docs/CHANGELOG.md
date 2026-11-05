# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.1.0] — 2026-11

### Added
- `rule_twf_elevated` heuristic flag: fires when `tool_wear >= 200 min`, surfaced in API response and dashboard
- **Research Findings tab** in Streamlit dashboard with three interactive charts:
  - LOFMO bar chart — physics rules (100% recall) vs. ML (7–14%) side-by-side
  - Cost-sensitivity charts — recall and cost savings at each C_miss:C_fa ratio
  - Blocked vs. Shuffled CV — average precision drop visualised
- `maint_audit/evaluation.py` — reusable module with `build_lofmo_dataframe()`, `build_cost_dataframe()`, `build_cv_comparison_dataframe()`
- Expanded test suite from 20 to 57 tests:
  - `tests/test_rules.py` — boundary-value coverage for HDF, PWF, OSF and TWF heuristic
  - `tests/test_api.py` — `/predict/batch`, empty batch, TWF field assertions
  - `tests/test_evaluation.py` — full coverage of the evaluation module
  - `tests/test_predict.py` — TWF flag propagation tests
- `requirements-lock.txt` — exact pinned dependency versions for reproducible environments

### Changed
- FastAPI endpoints `predict_single` and `predict_batch` converted to `async def`
- Shared `_row_to_response()` helper extracted to eliminate duplication in `api.py`
- Dashboard Altair charts: column and index names sanitised to remove parentheses, which caused `parse_shorthand` errors in newer Altair versions

---

## [1.0.0] — 2026-10

### Added
- MIT LICENSE
- `pyproject.toml` — installable package (`maint_audit`)
- `scripts/download_data.py` — checksum-validated data retrieval
- `scripts/verify_all.py` — end-to-end quality verification script
- `reports/metrics.json` — single source of truth for all reported numbers
- `reports/bootstrap_ci.json` — 95% confidence intervals (2,000 resamples)
- `reports/calibration.json` — Brier score, ECE, reliability diagram data
- `reports/cost_threshold.json` — cost-sensitive threshold optimisation sweep
- `reports/leave_one_mode_out.json` — leave-one-failure-mode-out experiments
- `reports/residual_analysis.json` — statistical characterisation of rule-unexplained failures
- `model_card.md` — model documentation and artifact integrity disclosure
- `docs/WRITEUP.md` — engineering narrative
- `Makefile` — one-command developer workflow entry points
- `.github/workflows/ci.yml` — lint + test + notebook smoke on Python 3.11–3.13
- `ruff.toml` + `.pre-commit-config.yaml` — code quality tooling
- FastAPI service (`maint_audit/api.py`) with `/health`, `/predict`, `/predict/batch`
- `Dockerfile` — slim non-root image with healthcheck
- Golden-metrics regression test

### Fixed
- README: corrected anomalous-row count and description direction
- Model artifact disclosure: shipped model (all 10,000 rows) separated from holdout metrics (dev-trained model)
- Model artifact metadata: added dataset SHA-256, scikit-learn version, git SHA
- Requirements pinned to exact versions

### Changed
- Package relocated from `src/` layout to `maint_audit/`
- Dashboard reads all metrics from `reports/metrics.json` (no hard-coded values)
