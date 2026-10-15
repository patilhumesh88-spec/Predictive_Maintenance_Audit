# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — Portfolio Upgrade

### Removed
- Stale context documentation that contradicted actual codebase state
- Legacy project checklist and pitch video script artifacts
- All former event and challenge naming references
- Placeholder deployment and upload links
- Unused empty data/processed/ directory

### Fixed
- README: "6 anomalous" corrected to 7; description was backwards (said "flag present but Machine failure=0", actually Machine failure=1 with no flag)
- Dashboard: removed false claim of "no physical warning indicators" and "stochastic recall ceiling" — tool wear is significantly associated with rule-unexplained failures (p<0.0001)
- Shipped model disclosure: model trained on all 10,000 rows now documented; holdout metrics attributed to dev-trained model
- Model artifact: added data hash, sklearn version, git SHA to metadata
- sklearn version pinned to avoid InconsistentVersionWarning on model load
- Requirements pinned to exact versions

### Added
- MIT LICENSE
- `pyproject.toml` — installable package (`maint_audit`)
- `scripts/download_data.py` — checksum-validated data download
- `scripts/verify_all.py` — comprehensive verification script
- `reports/metrics.json` — single source of truth for all numbers
- `reports/bootstrap_ci.json` — 95% confidence intervals for all metrics
- `reports/calibration.json` — Brier score, ECE, reliability diagram data
- `reports/cost_threshold.json` — cost-sensitive threshold sweep
- `reports/leave_one_mode_out.json` — leave-one-failure-mode-out experiments
- `reports/residual_analysis.json` — characterization of rule-unexplained failures
- `model_card.md` — model documentation
- `docs/WRITEUP.md` — 1000-word project narrative
- `Makefile` — one-command entry points
- `.github/workflows/ci.yml` — lint + test + notebook smoke on Python 3.11–3.13
- `ruff.toml` + `.pre-commit-config.yaml`
- FastAPI service (`maint_audit/api.py`) with `/health`, `/predict`, `/predict/batch`
- `Dockerfile` — slim non-root image with healthcheck
- Golden-metrics regression test
- Bootstrap confidence intervals for rules vs ML vs hybrid comparison

### Changed
- Package renamed from `src/` to `maint_audit/`
- README completely rewritten for 60-second recruiter skim
- Dashboard reads metrics from `reports/metrics.json` instead of hard-coded values
- AI disclosure reworded generically (no tool name)
