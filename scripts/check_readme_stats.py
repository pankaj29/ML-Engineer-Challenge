#!/usr/bin/env python
"""Check the README's status table against what the repository actually does.

The test counts and coverage figure in the README are the kind of number that
is correct on the day it is written and wrong a week later. Nobody notices,
because nothing depends on them, and a reviewer reading "1,010 tests" against
a suite of 1,133 has no reason to trust the other numbers either.

    python scripts/check_readme_stats.py           # report drift, exit 1
    python scripts/check_readme_stats.py --fix     # rewrite the README

Counting tests means collecting them, which takes a few seconds. Coverage is
read from `coverage.xml` or `.coverage` if either is present, and skipped
otherwise, so this is usable without a full coverage run.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
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


def measured_coverage() -> float | None:
    """api/ line coverage from a previous run, or None if there isn't one."""
    xml = REPO_ROOT / "coverage.xml"
    if xml.is_file():
        text = xml.read_text(encoding="utf-8")
        match = re.search(r'line-rate="([0-9.]+)"', text)
        if match:
            return round(float(match.group(1)) * 100, 1)

    if (REPO_ROOT / ".coverage").is_file():
        result = subprocess.run(
            [sys.executable, "-m", "coverage", "report", "--include=api/*"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        match = re.search(r"TOTAL\s+\d+\s+\d+\s+([0-9.]+)%", result.stdout)
        if match:
            return float(match.group(1))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="Rewrite the README in place.")
    args = parser.parse_args()

    unit = collect("tests/unit")
    integration = collect("tests/integration")
    performance = collect("tests/performance")
    total = unit + integration + performance
    coverage = measured_coverage()

    print(
        f"measured: {total:,} tests ({unit} unit, {integration} integration, "
        f"{performance} performance)"
    )
    print(f"coverage: {coverage}%" if coverage else "coverage: not measured (no coverage data)")

    text = README.read_text(encoding="utf-8")
    original = text
    problems: list[str] = []

    expected_row = (
        f"| Tests | {total:,}: {unit} unit, {integration} integration, "
        f"{performance} performance |"
    )
    row = re.search(r"^\| Tests \| .*\|$", text, re.MULTILINE)
    if row and row.group(0) != expected_row:
        problems.append(f"status table says:\n    {row.group(0)}\n  should be:\n    {expected_row}")
        text = text.replace(row.group(0), expected_row)

    expected_bullet = (
        f"- {total:,} tests: {unit} unit, {integration} integration, "
        f"{performance} performance, plus Locust load"
    )
    bullet = re.search(r"^- [\d,]+ tests: .*plus Locust load$", text, re.MULTILINE)
    if bullet and bullet.group(0) != expected_bullet:
        problems.append(
            f"testing section says:\n    {bullet.group(0)}\n  should be:\n    {expected_bullet}"
        )
        text = text.replace(bullet.group(0), expected_bullet)

    if coverage is not None:
        documented = re.search(r"^\| Coverage \| ([\d.]+)% on `api/`", text, re.MULTILINE)
        if documented and abs(float(documented.group(1)) - coverage) > 0.15:
            problems.append(f"coverage documented as {documented.group(1)}%, measured {coverage}%")
            text = text.replace(
                f"| Coverage | {documented.group(1)}% on `api/`",
                f"| Coverage | {coverage}% on `api/`",
            )

    if not problems:
        print("\nREADME matches the repository.")
        return 0

    print(f"\n{len(problems)} mismatch(es):\n")
    for problem in problems:
        print(f"  {problem}\n")

    if args.fix:
        README.write_text(text, encoding="utf-8")
        print("README updated.")
        return 0

    print("Re-run with --fix to update the README.")
    return 1 if text != original else 0


if __name__ == "__main__":
    raise SystemExit(main())
