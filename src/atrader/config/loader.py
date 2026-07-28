"""Loading and validating configuration.

Two properties this module is responsible for:

**Fail-closed.** Spec §7.1: *"리스크 설정을 로드할 수 없으면 → 시스템 부팅 거부."*
Anything wrong here raises :class:`~atrader.core.errors.ConfigError` and the
process does not start. There is no "carry on with defaults" path, because
defaults that nobody reviewed are not limits.

**Risk limits cannot come from the environment.** Spec §4.6 says limit changes
must go through a reviewed PR. An environment variable is not reviewable, so any
``ATRADER_RISK_*`` variable is a hard error rather than a silent override —
otherwise the guarantee is only as strong as whoever last edited a deploy
manifest. Infrastructure endpoints (DSN, bus URL, log level) *are* overridable,
because those legitimately differ per environment and carry no risk semantics.
"""

from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from atrader.config.schema import AppConfig, RiskConfig
from atrader.core.errors import ConfigError

__all__ = [
    "ENV_PREFIX",
    "OVERRIDABLE_ENV_VARS",
    "load_app_config",
    "load_risk_config",
    "load_yaml",
]

ENV_PREFIX = "ATRADER_"

#: Environment variable -> dotted path in the config tree. Deliberately short:
#: infrastructure wiring only, never anything with risk semantics.
OVERRIDABLE_ENV_VARS: dict[str, str] = {
    "ATRADER_ENVIRONMENT": "environment",
    "ATRADER_ACCOUNT_EQUITY": "account_equity",
    "ATRADER_STORAGE_DSN": "storage.dsn",
    "ATRADER_BUS_BACKEND": "bus.backend",
    "ATRADER_BUS_URL": "bus.url",
    "ATRADER_LOG_LEVEL": "monitoring.log_level",
    "ATRADER_METRICS_PORT": "monitoring.metrics_port",
}

REQUIRED_FILES = ("risk.yaml",)
OPTIONAL_FILES = ("app.yaml", "universe.yaml", "instruments.yaml", "strategies.yaml")


class _DecimalSafeLoader(yaml.SafeLoader):
    """YAML loader that produces :class:`Decimal` instead of ``float``.

    ``2.0`` in a YAML file must not become a binary float on its way to a risk
    limit — see :mod:`atrader.core.money`. Overriding the resolver here means
    ``config/risk.yaml`` can be written exactly as spec §13 shows it and still
    arrive as exact decimals.
    """


def _construct_decimal(loader: yaml.SafeLoader, node: yaml.Node) -> Decimal | float:
    if not isinstance(node, yaml.ScalarNode):  # pragma: no cover — floats are scalars
        raise ConfigError(f"unexpected non-scalar float node at {node.start_mark}")
    text = loader.construct_scalar(node).replace("_", "")
    try:
        return Decimal(text)
    except InvalidOperation:
        # `.inf` / `.nan` are not meaningful as limits; hand them to pydantic,
        # which will reject them against the field constraints.
        return float(text)


_DecimalSafeLoader.add_constructor("tag:yaml.org,2002:float", _construct_decimal)


def load_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML mapping, preserving decimal precision."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc

    try:
        # _DecimalSafeLoader subclasses SafeLoader, so this is not arbitrary construction.
        data = yaml.load(raw, Loader=_DecimalSafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} must contain a mapping at the top level, got {type(data).__name__}"
        )
    return data


def _set_path(tree: dict[str, Any], dotted: str, value: str) -> None:
    parts = dotted.split(".")
    cursor = tree
    for part in parts[:-1]:
        nested = cursor.setdefault(part, {})
        if not isinstance(nested, dict):
            raise ConfigError(f"cannot override {dotted}: {part} is not a mapping")
        cursor = nested
    cursor[parts[-1]] = value


def _apply_env_overlay(tree: dict[str, Any], environ: dict[str, str]) -> None:
    """Apply the allowlisted environment overrides, rejecting risk overrides."""
    forbidden = sorted(
        name
        for name in environ
        if name.startswith(f"{ENV_PREFIX}RISK") and name not in OVERRIDABLE_ENV_VARS
    )
    if forbidden:
        raise ConfigError(
            f"risk limits cannot be set from the environment: {forbidden}. "
            "Spec §4.6 requires limit changes to go through a reviewed PR — an env var "
            "in a deploy manifest is not that. Edit config/risk.yaml instead."
        )

    unknown = sorted(
        name for name in environ if name.startswith(ENV_PREFIX) and name not in OVERRIDABLE_ENV_VARS
    )
    if unknown:
        raise ConfigError(
            f"unrecognised {ENV_PREFIX}* variables: {unknown}. "
            f"Overridable variables are: {sorted(OVERRIDABLE_ENV_VARS)}. "
            "A typo'd override that is silently ignored is worse than a failed boot."
        )

    for name, dotted in OVERRIDABLE_ENV_VARS.items():
        if name in environ:
            _set_path(tree, dotted, environ[name])


def load_risk_config(config_dir: Path) -> RiskConfig:
    """Load and validate ``risk.yaml`` on its own.

    Separate entry point so the risk engine can verify its configuration without
    pulling in strategy or instrument definitions.
    """
    path = config_dir / "risk.yaml"
    if not path.is_file():
        raise ConfigError(
            f"{path} is required and missing. The system refuses to boot without risk "
            "limits rather than trading with defaults nobody approved (spec §7.1)."
        )
    data = load_yaml(path)
    block = data.get("risk", data)
    if not isinstance(block, dict):
        raise ConfigError(f"{path}: 'risk' must be a mapping")
    try:
        return RiskConfig.model_validate(block)
    except ValidationError as exc:
        raise ConfigError(f"{path} failed validation:\n{exc}") from exc


def load_app_config(config_dir: Path, *, environ: dict[str, str] | None = None) -> AppConfig:
    """Assemble the full application configuration from ``config_dir``.

    Files are merged by their top-level key, so each concern stays in its own
    reviewable file:

    ==================== ==========================================
    ``risk.yaml``        ``risk:`` — required (spec §13)
    ``app.yaml``         everything else at the top level
    ``universe.yaml``    ``universe:``
    ``instruments.yaml`` ``instruments:``
    ``strategies.yaml``  ``strategies:``
    ==================== ==========================================

    LLM settings live under ``risk.llm``, matching spec §13 rather than being
    split into their own file — the model pin is a risk control.
    """
    if not config_dir.is_dir():
        raise ConfigError(f"config directory {config_dir} does not exist")

    for name in REQUIRED_FILES:
        if not (config_dir / name).is_file():
            raise ConfigError(
                f"required config file {config_dir / name} is missing; refusing to boot"
            )

    tree: dict[str, Any] = {}
    for name in (*REQUIRED_FILES, *OPTIONAL_FILES):
        path = config_dir / name
        if not path.is_file():
            continue
        content = load_yaml(path)
        overlapping = sorted(set(content) & set(tree))
        if overlapping:
            raise ConfigError(
                f"{path} redefines keys already set by an earlier file: {overlapping}. "
                "Split configuration must not overlap — which file wins would be arbitrary."
            )
        tree.update(content)

    _apply_env_overlay(tree, dict(os.environ) if environ is None else environ)

    try:
        return AppConfig.model_validate(tree)
    except ValidationError as exc:
        raise ConfigError(f"configuration in {config_dir} failed validation:\n{exc}") from exc
