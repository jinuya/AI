"""Static guard against non-determinism.

Spec §2.2: *"같은 입력 이벤트 시퀀스를 넣으면 같은 주문이 나와야 한다."* Replaying
recorded market data must produce a byte-identical order sequence
(acceptance criterion #3). A single stray ``datetime.now()`` or ``uuid4()``
somewhere in the order path silently breaks that, and the failure shows up as a
confusing replay diff rather than as an obvious bug.

So instead of hoping reviewers catch it, this test walks the AST of every
module and fails the build. Time comes from an injected ``Clock``; IDs come from
an injected ``IdGenerator``; randomness comes from a seeded generator held by
whoever needs it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import atrader

PACKAGE_ROOT = Path(atrader.__file__).parent

#: Modules permitted to touch the real clock / OS entropy. These are the
#: adapters that everything else injects.
ALLOWLIST: frozenset[str] = frozenset(
    {
        "core/clock.py",
        "core/ids.py",
        "core/rng.py",
    }
)

#: Dotted call suffixes that read ambient time or entropy.
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "datetime.now",
        "datetime.utcnow",
        "datetime.today",
        "date.today",
        "time.time",
        "time.time_ns",
        "time.monotonic",
        "time.monotonic_ns",
        "time.perf_counter",
        "time.perf_counter_ns",
        "uuid.uuid1",
        "uuid.uuid4",
        "secrets.randbits",
        "secrets.token_bytes",
        "secrets.token_hex",
        "os.urandom",
    }
)

#: Modules whose members are non-deterministic wholesale.
FORBIDDEN_MODULE_IMPORTS: frozenset[str] = frozenset({"random"})

#: Bare names that, when imported directly, are non-deterministic.
FORBIDDEN_IMPORTED_NAMES: frozenset[str] = frozenset(
    {"uuid1", "uuid4", "urandom", "randbits", "token_bytes", "token_hex", "monotonic_ns"}
)


def _dotted_name(node: ast.expr) -> str:
    """Render an attribute/name expression as a dotted string."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _source_files() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted_name(node.func)
            if not dotted:
                continue
            # Match on the trailing two segments so `datetime.datetime.now`
            # and `datetime.now` both trip.
            segments = dotted.split(".")
            tail = ".".join(segments[-2:])
            if tail in FORBIDDEN_CALLS:
                found.append(f"line {node.lineno}: call to {dotted}()")

        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULE_IMPORTS:
                    found.append(f"line {node.lineno}: import {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            module_root = (node.module or "").split(".")[0]
            if module_root in FORBIDDEN_MODULE_IMPORTS:
                found.append(f"line {node.lineno}: from {node.module} import ...")
                continue
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTED_NAMES:
                    found.append(f"line {node.lineno}: from {node.module} import {alias.name}")

    return found


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(PACKAGE_ROOT)))
def test_module_is_deterministic(path: Path) -> None:
    relative = path.relative_to(PACKAGE_ROOT).as_posix()
    if relative in ALLOWLIST:
        pytest.skip(f"{relative} is an allowlisted time/entropy adapter")

    violations = _violations(path)
    assert not violations, (
        f"{relative} reads ambient time or entropy, which breaks deterministic replay:\n  "
        + "\n  ".join(violations)
        + "\n\nInject a Clock (atrader.core.clock) or IdGenerator (atrader.core.ids) instead."
    )


def test_allowlist_entries_exist() -> None:
    """A stale allowlist entry would silently widen the exemption."""
    for relative in ALLOWLIST:
        assert (PACKAGE_ROOT / relative).is_file(), f"allowlisted file {relative} no longer exists"


def test_lint_detects_a_planted_violation(tmp_path: Path) -> None:
    """The linter is only useful if it actually catches something."""
    planted = tmp_path / "bad.py"
    planted.write_text(
        "import datetime\n"
        "import random\n"
        "def f():\n"
        "    return datetime.datetime.now(), random.random()\n",
        encoding="utf-8",
    )
    violations = _violations(planted)
    assert any("datetime" in v for v in violations)
    assert any("random" in v for v in violations)
