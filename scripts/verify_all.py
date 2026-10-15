#!/usr/bin/env python3
"""Comprehensive verification script. Run after every milestone.

Usage:  python scripts/verify_all.py
Exit code 0 = all checks pass.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAIL = False


def check(name, passed, detail=""):
    global FAIL
    icon = "[PASS]" if passed else "[FAIL]"
    print(f"  {icon} {name}" + (f" - {detail}" if detail else ""))
    if not passed:
        FAIL = True


def main():
    os.chdir(ROOT)

    print("\n=== 1. BRANDING CHECK ===")
    brand_terms = ["algothon", "alg-data", "predict what happens", "ps-02",
                   "hackathon", "competition", "submission", "demo video"]
    hits = []
    for p in ROOT.rglob("*"):
        if p.is_dir() or any(x in p.parts for x in (".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache")):
            continue
        if p.suffix in (".pyc", ".joblib", ".png", ".jpg", ".gif", ".ico"):
            continue
        if p.resolve() == Path(__file__).resolve():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").lower()
            for term in brand_terms:
                if term in text:
                    hits.append(f"{p.relative_to(ROOT)}: '{term}'")
        except Exception:
            pass
    # Also check notebook cells
    for nb_path in ROOT.glob("notebooks/*.ipynb"):
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            for cell in nb.get("cells", []):
                source = "".join(cell.get("source", [])).lower()
                for term in brand_terms:
                    if term in source:
                        hits.append(f"{nb_path.relative_to(ROOT)} (cell): '{term}'")
        except Exception:
            pass
    check("No branding terms", len(hits) == 0, f"{len(hits)} hits: {hits[:5]}" if hits else "")

    print("\n=== 2. PERSONAL PATH CHECK ===")
    path_hits = []
    path_pattern = re.compile(r"[Cc]:\\[Uu]sers\\|/home/\w+|/Users/\w+", re.IGNORECASE)
    for p in ROOT.rglob("*"):
        if p.is_dir() or any(x in p.parts for x in (".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache")):
            continue
        if p.suffix in (".pyc", ".joblib", ".png", ".jpg", ".gif", ".ico"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if path_pattern.search(text):
                path_hits.append(str(p.relative_to(ROOT)))
        except Exception:
            pass
    check("No personal paths in source", len(path_hits) == 0,
          f"{len(path_hits)} files: {path_hits[:5]}" if path_hits else "")

    print("\n=== 3. PYTEST ===")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
                       capture_output=True, text=True, cwd=str(ROOT))
    passed_tests = "passed" in r.stdout and "failed" not in r.stdout
    # Extract summary line
    lines = r.stdout.strip().split("\n")
    summary = lines[-1] if lines else ""
    check("All tests pass", passed_tests, summary)
    if not passed_tests:
        print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)

    print("\n=== 4. METRICS.JSON ===")
    metrics_path = ROOT / "reports" / "metrics.json"
    check("reports/metrics.json exists", metrics_path.exists())
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        check("Has holdout key", "holdout" in m)
        check("Has dataset key", "dataset" in m)

    print("\n=== 5. MODEL ARTIFACT ===")
    model_path = ROOT / "models" / "rf_engineered.joblib"
    check("Model file exists", model_path.exists())

    print("\n=== 6. STALE FILES ===")
    stale = []
    for name in ["@context.txt", "docs/SUBMISSION_CHECKLIST.md"]:
        if (ROOT / name).exists():
            stale.append(name)
    check("No stale files", len(stale) == 0, f"found: {stale}" if stale else "")

    print("\n=== 7. PACKAGE IMPORTABLE ===")
    r = subprocess.run([sys.executable, "-c", "import maint_audit; print(maint_audit.__doc__)"],
                       capture_output=True, text=True, cwd=str(ROOT))
    check("maint_audit importable", r.returncode == 0, r.stdout.strip()[:100])

    print("\n=== 8. LICENSE ===")
    check("LICENSE file exists", (ROOT / "LICENSE").exists())

    print()
    if FAIL:
        print("[FAIL] VERIFICATION FAILED - see above")
        return 1
    else:
        print("[SUCCESS] ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
