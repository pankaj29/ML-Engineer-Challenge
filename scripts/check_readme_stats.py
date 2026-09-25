#!/usr/bin/env python
"""Check the README's status table against what the repository actually does.

The test counts and coverage figure in the README are the kind of number that
is correct on the day it is written and wrong a week later. Nobody notices,
because nothing depends on them, and a reviewer who spots one stale number has
no reason to trust the rest.

    python scripts/check_readme_stats.py           # report drift
    python scripts/check_readme_stats.py --fix     # rewrite the README

Test counts are a hard failure: they do not depend on the environment.
Coverage is reported but never fails the run, because it does. A missing Git
LFS fetch or an absent Docker stack skips whole test files and moves coverage
several points, and failing the build for that blames the runner rather than
the code. The real coverage gate is `--cov-fail-under`, which runs separately.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"


def collect(path: str) -> int:
    """How many tests pytest finds under `path`."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", path, "--collect-only", "-q", "-p", "no:randomly"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    match = re.search(r"(\d+) tests? collected", result.stdout)
    if not match:
        raise RuntimeError(f"could not count tests in {path}:\n{result.stdout[-500:]}")
    return int(match.group(1))


def coverage_from_xml(path: Path) -> float | None:
    """api/ line coverage from a Cobertura report.

    The top-level `line-rate` is whatever scope the run measured. CI runs
    `--cov=api --cov=worker --cov=models`, so that figure spans all three and
    sits several points below api/ alone. Only classes whose filename is under
    api/ are counted here.
    """
    # S314 is about untrusted XML. This file is written by our own coverage
    # run, in the same job, moments earlier.
    root = ElementTree.parse(path).getroot()  # noqa: S314
    covered = valid = 0
    for klass in root.iter("class"):
        filename = (klass.get("filename") or "").replace("\\", "/")
        if not filename.startswith("api/"):
            continue
        for line in klass.iter("line"):
            valid += 1
            if line.get("hits") not in (None, "0"):
                covered += 1
    return round(covered / valid * 100, 1) if valid else None


def coverage_from_data(path: Path) -> float | None:
    """api/ line coverage from a `.coverage` database."""
    result = subprocess.run(
        [sys.executable, "-m", "coverage", "report", "--include=api/*"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    match = re.search(r"TOTAL\s+\d+\s+\d+\s+([0-9.]+)%", result.stdout)
    return float(match.group(1)) if match else None


def measured_coverage() -> float | None:
    """api/ coverage from whichever report a previous run left behind.

    `.coverage` is preferred: `coverage report --include=api/*` does the
    filtering itself, so there is no chance of reading a figure for the wrong
    scope. The XML is a fallback for a runner that keeps only that.
    """
    data = REPO_ROOT / ".coverage"
    if data.is_file():
        found = coverage_from_data(data)
        if found is not None:
            return found
    xml = REPO_ROOT / "coverage.xml"
    if xml.is_file():
        return coverage_from_xml(xml)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="Rewrite the README in place.")
    args = parser.parse_args()

    unit = collect("tests/unit")
    integration = collect("tests/integration")
    e2e = collect("tests/e2e")
    performance = collect("tests/performance")
    total = unit + integration + e2e + performance
    coverage = measured_coverage()

    print(
        f"measured: {total:,} tests ({unit} unit, {integration} integration, "
        f"{e2e} end-to-end, {performance} performance)"
    )
    print(
        f"coverage: {coverage}% on api/"
        if coverage is not None
        else "coverage: not measured (no coverage report present)"
    )

    text = README.read_text(encoding="utf-8")
    problems: list[str] = []

    expected_row = (
        f"| Tests | {total:,}: {unit} unit, {integration} integration, "
        f"{e2e} end-to-end, {performance} performance |"
    )
    row = re.search(r"^\| Tests \| .*\|$", text, re.MULTILINE)
    if row and row.group(0) != expected_row:
        problems.append(f"status table says:\n    {row.group(0)}\n  should be:\n    {expected_row}")
        text = text.replace(row.group(0), expected_row)

    expected_bullet = (
        f"- {total:,} tests: {unit} unit, {integration} integration, "
        f"{e2e} end-to-end, {performance} performance, plus Locust load"
    )
    bullet = re.search(r"^- [\d,]+ tests: .*plus Locust load$", text, re.MULTILINE)
    if bullet and bullet.group(0) != expected_bullet:
        problems.append(
            f"testing section says:\n    {bullet.group(0)}\n  should be:\n    {expected_bullet}"
        )
        text = text.replace(bullet.group(0), expected_bullet)

    # Reported, never fatal. See the module docstring.
    if coverage is not None:
        documented = re.search(r"^\| Coverage \| ([\d.]+)% on `api/`", text, re.MULTILINE)
        if documented and abs(float(documented.group(1)) - coverage) > 0.15:
            print(
                f"\nnote: README documents {documented.group(1)}% coverage, this run "
                f"measured {coverage}%. Environment-dependent, so not a failure. "
                f"Run --fix locally with the full suite to update it."
            )
            if args.fix:
                text = text.replace(
                    f"| Coverage | {documented.group(1)}% on `api/`",
                    f"| Coverage | {coverage}% on `api/`",
                )

    if args.fix:
        README.write_text(text, encoding="utf-8")
        print("\nREADME updated.")
        return 0

    if not problems:
        print("\nREADME test counts match the repository.")
        return 0

    print(f"\n{len(problems)} mismatch(es):\n")
    for problem in problems:
        print(f"  {problem}\n")
    print("Re-run with --fix to update the README.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
