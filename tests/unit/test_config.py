"""Configuration loading and validation.

The properties under test are the ones that keep a bad limit from reaching
production: fail-closed loading, exact decimals, no environment overrides of
risk limits, and rejection of the sampling parameters spec §13 asks for but the
current API no longer accepts.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from atrader.config.loader import load_app_config, load_risk_config, load_yaml
from atrader.config.schema import (
    AppConfig,
    CircuitBreakerConfig,
    InstrumentSpec,
    LLMConfig,
    RiskConfig,
    StrategyConfig,
    UniverseConfig,
)
from atrader.core.errors import ConfigError

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"

MINIMAL_RISK_YAML = """
risk:
  account:
    daily_loss_limit_pct: 2.0
"""


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    (tmp_path / "risk.yaml").write_text(MINIMAL_RISK_YAML, encoding="utf-8")
    return tmp_path


class TestShippedConfig:
    """The configuration actually committed to the repo must be valid."""

    def test_loads(self) -> None:
        config = load_app_config(REPO_CONFIG, environ={})
        assert config.universe.symbols
        assert config.instruments
        assert config.risk.llm.model == "claude-opus-5"

    def test_percentages_are_exact_decimals_not_floats(self) -> None:
        # 2.0 as a binary float is not exactly 2.0; a risk limit must be exact.
        config = load_app_config(REPO_CONFIG, environ={})
        limit = config.risk.account.daily_loss_limit_pct
        assert isinstance(limit, Decimal)
        assert limit == Decimal("2.0")

    def test_shipped_config_does_not_set_removed_sampling_params(self) -> None:
        config = load_app_config(REPO_CONFIG, environ={})
        assert config.risk.llm.temperature is None
        assert config.risk.llm.top_p is None
        assert config.risk.llm.seed is None

    def test_llm_strategy_ships_disabled(self) -> None:
        # Spec §FR-STR-05: promotion is deliberate, never the default.
        config = load_app_config(REPO_CONFIG, environ={})
        llm = next(s for s in config.strategies if s.strategy_id == "llm_agent")
        assert llm.enabled is False


class TestFailClosed:
    def test_missing_directory(self) -> None:
        with pytest.raises(ConfigError, match="does not exist"):
            load_app_config(Path("/nonexistent/config"), environ={})

    def test_missing_risk_file_refuses_to_boot(self, tmp_path: Path) -> None:
        # Spec §7.1: no risk config means no boot, not "boot with defaults".
        with pytest.raises(ConfigError, match="required config file"):
            load_app_config(tmp_path, environ={})

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "risk.yaml").write_text("risk: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_app_config(tmp_path, environ={})

    def test_non_mapping_top_level(self, tmp_path: Path) -> None:
        (tmp_path / "risk.yaml").write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="must contain a mapping"):
            load_yaml(tmp_path / "risk.yaml")

    def test_unknown_key_is_an_error_not_a_silent_default(self, config_dir: Path) -> None:
        # A typo'd limit that silently falls back to a default is the exact
        # failure mode extra="forbid" exists to prevent.
        (config_dir / "risk.yaml").write_text(
            "risk:\n  account:\n    daily_los_limit_pct: 2.0\n", encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="failed validation"):
            load_risk_config(config_dir)

    def test_files_may_not_redefine_each_others_keys(self, config_dir: Path) -> None:
        (config_dir / "app.yaml").write_text("risk:\n  account: {}\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="redefines keys"):
            load_app_config(config_dir, environ={})


class TestEnvironmentOverlay:
    def test_infrastructure_overrides_are_allowed(self, config_dir: Path) -> None:
        config = load_app_config(
            config_dir,
            environ={"ATRADER_ENVIRONMENT": "staging", "ATRADER_LOG_LEVEL": "DEBUG"},
        )
        assert config.environment == "staging"
        assert config.monitoring.log_level == "DEBUG"

    def test_risk_limits_cannot_be_set_from_the_environment(self, config_dir: Path) -> None:
        # Spec §4.6: a limit you can raise from a deploy manifest is not a limit.
        with pytest.raises(ConfigError, match="risk limits cannot be set from the environment"):
            load_app_config(config_dir, environ={"ATRADER_RISK_ACCOUNT_MAX_LEVERAGE": "10"})

    def test_typo_in_an_override_fails_loudly(self, config_dir: Path) -> None:
        with pytest.raises(ConfigError, match="unrecognised"):
            load_app_config(config_dir, environ={"ATRADER_ENVIRONMNET": "staging"})


class TestLLMConfig:
    @pytest.mark.parametrize("field", ["temperature", "top_p", "seed"])
    def test_removed_sampling_params_are_rejected(self, field: str) -> None:
        # Spec §13 asks for temperature=0. Current models return 400 for it,
        # so the failure has to happen at boot rather than at the first call.
        with pytest.raises(ValidationError, match="removed from the Claude API"):
            LLMConfig(model="claude-opus-5", **{field: 0})  # type: ignore[arg-type]

    def test_sampling_params_still_allowed_on_a_legacy_model(self) -> None:
        config = LLMConfig(model="claude-sonnet-4-6", temperature=Decimal(0))
        assert config.temperature == Decimal(0)

    def test_unpinned_model_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not in the pinned allowlist"):
            LLMConfig(model="claude-latest")

    def test_unknown_model_allowed_with_explicit_opt_in(self) -> None:
        config = LLMConfig(model="claude-something-new", allow_unknown_model=True)
        assert config.model == "claude-something-new"

    def test_effort_is_validated(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(effort="turbo")  # type: ignore[arg-type]

    def test_confidence_threshold_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(min_confidence=Decimal("1.5"))


class TestCircuitBreakerValidation:
    def test_levels_must_escalate(self) -> None:
        # An L2 that fires before L1 means L1 is dead code.
        with pytest.raises(ValidationError, match="must increase L1 < L2 < L3"):
            CircuitBreakerConfig(l1_daily_loss_pct=Decimal(5), l2_daily_loss_pct=Decimal(2))

    def test_drawdown_levels_must_escalate(self) -> None:
        with pytest.raises(ValidationError, match="drawdown thresholds must increase"):
            CircuitBreakerConfig(l2_max_drawdown_pct=Decimal(15), l3_max_drawdown_pct=Decimal(10))

    def test_l2_after_the_hard_daily_limit_could_never_fire(self) -> None:
        with pytest.raises(ValidationError, match="could never trigger"):
            RiskConfig.model_validate(
                {
                    "account": {"daily_loss_limit_pct": 1.0},
                    "circuit_breaker": {
                        "l1_daily_loss_pct": 1.5,
                        "l2_daily_loss_pct": 2.0,
                        "l3_daily_loss_pct": 3.0,
                    },
                }
            )


class TestUniverseAndStrategyValidation:
    def test_sector_map_must_reference_known_symbols(self) -> None:
        with pytest.raises(ValidationError, match="outside the universe"):
            UniverseConfig(symbols=("AAPL",), sectors={"MSFT": "TECHNOLOGY"})

    def test_universe_symbol_without_an_instrument_spec_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no InstrumentSpec"):
            AppConfig(universe=UniverseConfig(symbols=("AAPL",)))

    def test_strategy_universe_must_be_a_subset(self) -> None:
        with pytest.raises(ValidationError, match="outside the global universe"):
            AppConfig(
                universe=UniverseConfig(symbols=("AAPL",)),
                instruments=(InstrumentSpec(symbol="AAPL"),),
                strategies=(StrategyConfig(strategy_id="s", universe=("TSLA",)),),
            )

    def test_allocations_cannot_exceed_one_hundred_percent(self) -> None:
        with pytest.raises(ValidationError, match="more than exists"):
            AppConfig(
                strategies=(
                    StrategyConfig(
                        strategy_id="a", enabled=True, capital_allocation_pct=Decimal(60)
                    ),
                    StrategyConfig(
                        strategy_id="b", enabled=True, capital_allocation_pct=Decimal(60)
                    ),
                )
            )

    def test_disabled_strategies_do_not_count_toward_allocation(self) -> None:
        config = AppConfig(
            strategies=(
                StrategyConfig(strategy_id="a", enabled=True, capital_allocation_pct=Decimal(60)),
                StrategyConfig(strategy_id="b", enabled=False, capital_allocation_pct=Decimal(60)),
            )
        )
        assert len(config.strategies) == 2


class TestImmutability:
    def test_limits_cannot_be_mutated_after_load(self) -> None:
        # The limits in force must be the limits that were reviewed.
        config = load_app_config(REPO_CONFIG, environ={})
        with pytest.raises(ValidationError):
            config.risk.account.daily_loss_limit_pct = Decimal(50)  # type: ignore[misc]
