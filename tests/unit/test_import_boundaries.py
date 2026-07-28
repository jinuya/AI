"""Acceptance criterion #1 — no code path can bypass the risk engine.

Spec §7.1:

    리스크 엔진은 주문 경로상의 필수 통과 지점이다. [...] 전략이 브로커 어댑터를
    import조차 할 수 없도록 패키지 구조로 강제한다.

Spec §12-1 says this must be *statically verified*. ``import-linter`` enforces
the same contracts in CI (see ``[tool.importlinter]`` in ``pyproject.toml``),
but this test duplicates them with nothing but the stdlib so the guarantee holds
even where that tool is not installed — and so the failure message explains
*why* the boundary exists.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import atrader

PACKAGE_ROOT = Path(atrader.__file__).parent
PACKAGE_NAME = "atrader"

#: source subpackage -> subpackages it may not import, and why.
FORBIDDEN_EDGES: dict[str, dict[str, str]] = {
    "strategy": {
        "brokers": "a strategy that can reach the broker can send an unchecked order",
        "execution": "order construction and submission belong behind the risk gate",
        "risk": "strategies are gated by the risk engine, they are not callers of it",
    },
    "marketdata": {
        "strategy": "the data path must not depend on who consumes it",
        "risk": "the data path must not depend on who consumes it",
        "execution": "the data path must not depend on who consumes it",
        "brokers": "the data path must not depend on who consumes it",
        "portfolio": "the data path must not depend on who consumes it",
    },
    "features": {
        "strategy": "feature computation must not depend on its consumers",
        "risk": "feature computation must not depend on its consumers",
        "execution": "feature computation must not depend on its consumers",
        "brokers": "feature computation must not depend on its consumers",
    },
    "core": {
        # core is dependency-free; every other subpackage is off limits.
        "config": "core is the dependency-free base layer",
        "bus": "core is the dependency-free base layer",
        "storage": "core is the dependency-free base layer",
        "audit": "core is the dependency-free base layer",
        "marketdata": "core is the dependency-free base layer",
        "features": "core is the dependency-free base layer",
        "strategy": "core is the dependency-free base layer",
        "portfolio": "core is the dependency-free base layer",
        "risk": "core is the dependency-free base layer",
        "execution": "core is the dependency-free base layer",
        "brokers": "core is the dependency-free base layer",
        "backtest": "core is the dependency-free base layer",
        "monitoring": "core is the dependency-free base layer",
        "app": "core is the dependency-free base layer",
    },
}

#: Only these subpackages may talk to a broker adapter at all. Everything else
#: goes through the execution layer, which sits behind the risk gate.
BROKER_IMPORTERS: frozenset[str] = frozenset({"brokers", "execution", "backtest", "app"})


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join([PACKAGE_NAME, *parts])


def _subpackage(module_name: str) -> str | None:
    parts = module_name.split(".")
    return parts[1] if len(parts) > 1 else None


def _imported_modules(path: Path, module_name: str) -> set[str]:
    """Every ``atrader.*`` module this file imports, absolute and relative."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package_parts = module_name.split(".")
    if path.name != "__init__.py":
        package_parts = package_parts[:-1]

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == PACKAGE_NAME:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - node.level + 1]
                target = ".".join([*base, node.module] if node.module else base)
            else:
                target = node.module or ""
            if target.split(".")[0] == PACKAGE_NAME:
                found.add(target)
                # `from atrader.risk import engine` also reaches atrader.risk.engine
                found.update(f"{target}.{alias.name}" for alias in node.names)
    return found


def _source_files() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(PACKAGE_ROOT)))
def test_layering_contract(path: Path) -> None:
    module_name = _module_name(path)
    source_pkg = _subpackage(module_name)
    if source_pkg is None:
        return

    forbidden = FORBIDDEN_EDGES.get(source_pkg, {})
    if not forbidden:
        return

    violations: list[str] = []
    for imported in sorted(_imported_modules(path, module_name)):
        target_pkg = _subpackage(imported)
        if target_pkg is not None and target_pkg in forbidden:
            violations.append(f"{imported}  ({forbidden[target_pkg]})")

    assert not violations, (
        f"{module_name} violates the layering contract by importing:\n  " + "\n  ".join(violations)
    )


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(PACKAGE_ROOT)))
def test_only_execution_layer_reaches_the_broker(path: Path) -> None:
    """The concrete statement of acceptance criterion #1."""
    module_name = _module_name(path)
    source_pkg = _subpackage(module_name)
    if source_pkg is None or source_pkg in BROKER_IMPORTERS:
        return

    broker_imports = [
        m for m in sorted(_imported_modules(path, module_name)) if _subpackage(m) == "brokers"
    ]
    assert not broker_imports, (
        f"{module_name} imports the broker adapter directly: {broker_imports}. "
        "Every order must pass through the risk engine and the execution layer; "
        f"only {sorted(BROKER_IMPORTERS)} may import atrader.brokers."
    )


def test_contract_detects_a_planted_violation(tmp_path: Path) -> None:
    """A boundary check that never fires is worse than none at all."""
    planted = tmp_path / "sneaky.py"
    planted.write_text("from atrader.brokers import paper\n", encoding="utf-8")
    imports = _imported_modules(planted, "atrader.strategy.sneaky")
    assert any(_subpackage(m) == "brokers" for m in imports)
