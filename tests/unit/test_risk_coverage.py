"""Coverage gate — acceptance criterion #2.

    전체 80%, 리스크 엔진·주문 상태머신 95%

The risk engine and order state machine are the two components every other
safety property in this system leans on (the gate no intent bypasses, and
the transition table no illegal state change survives) — so they get a
stricter bar than the rest of the codebase.

This measures coverage in a genuinely separate ``coverage`` process rather
than nesting a second ``Coverage()`` instance inside whatever is already
instrumenting *this* test run (pytest-cov, if the caller passed
``--cov=atrader``) — two active coverage trackers in one process corrupt
each other's state. Running the measurement as a subprocess is slower than
an in-process assertion would be, but it is the only version of this gate
that produces a real number instead of a guess, and it is meant to be run
as its own explicit step (see ``docs/runbook.md``), not on every routine
test invocation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

THRESHOLD_PCT = 95.0
MODULES = ("atrader.risk.engine", "atrader.execution.statemachine")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _measure_combined_coverage_pct() -> float:
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "coverage.json"
        cov_args = [arg for module in MODULES for arg in ("--cov", module)]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit",
                # Exclude this file itself — it lives under tests/unit, and
                # without this the subprocess would collect and re-run this
                # very test, spawning another subprocess, recursively.
                "--ignore=tests/unit/test_risk_coverage.py",
                *cov_args,
                f"--cov-report=json:{report_path}",
                "-q",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"the unit suite must pass before its coverage means anything:\n{result.stdout}\n{result.stderr}"
        )
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return float(data["totals"]["percent_covered"])


def test_risk_engine_and_statemachine_meet_the_95_percent_gate() -> None:
    percent = _measure_combined_coverage_pct()
    assert percent >= THRESHOLD_PCT, (
        f"{'/'.join(MODULES)} combined coverage is {percent:.1f}%, below the "
        f"{THRESHOLD_PCT}% floor acceptance criterion #2 requires"
    )
