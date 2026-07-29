"""The CLI — spec §10. Self-contained commands (backtest/replay/verify-audit)
and the strategy-wiring helpers; ``run``/``paper``/``kill``/``reconcile``
against a live process are exercised manually (see docs/runbook.md) since
they need a real server — the ``Runtime``/API layers they call are already
covered end to end in ``test_runtime.py``/``test_api.py``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from atrader.app import cli as cli_module
from atrader.audit.hashchain import GENESIS_HASH, AuditRecord, compute_hash, dump_records_jsonl
from atrader.backtest.divergence import dump_equity_curve, load_equity_curve
from atrader.config.schema import StrategyConfig
from atrader.core.errors import ATraderError
from atrader.strategy.rules.sma_crossover import SmaCrossoverStrategy

runner = CliRunner()

_RISK_YAML = "risk:\n  account:\n    max_leverage: 1.0\n"
_APP_YAML = 'environment: dev\naccount_equity: 100000\nmarket_data:\n  bar_intervals: ["1d"]\n'
_UNIVERSE_YAML = 'universe:\n  symbols: ["AAPL"]\n  sectors:\n    AAPL: TECHNOLOGY\n'
_INSTRUMENTS_YAML = (
    "instruments:\n"
    "  - symbol: AAPL\n"
    "    sector: TECHNOLOGY\n"
    '    market_open_utc: "00:00"\n'
    '    market_close_utc: "23:59"\n'
)
_STRATEGIES_YAML = (
    "strategies:\n"
    "  - strategy_id: sma_crossover\n"
    "    enabled: true\n"
    "    capital_allocation_pct: 50\n"
    '    universe: ["AAPL"]\n'
    "    params:\n"
    "      fast_period: 5\n"
    "      slow_period: 10\n"
    "      min_bars: 5\n"
)


def write_config(tmp_path: Path, *, strategies_yaml: str = _STRATEGIES_YAML) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "risk.yaml").write_text(_RISK_YAML, encoding="utf-8")
    (config_dir / "app.yaml").write_text(_APP_YAML, encoding="utf-8")
    (config_dir / "universe.yaml").write_text(_UNIVERSE_YAML, encoding="utf-8")
    (config_dir / "instruments.yaml").write_text(_INSTRUMENTS_YAML, encoding="utf-8")
    (config_dir / "strategies.yaml").write_text(strategies_yaml, encoding="utf-8")
    return config_dir


class TestLoadConfig:
    def test_missing_risk_yaml_refuses_to_boot_as_a_clean_cli_exit(self, tmp_path: Path) -> None:
        # _load_config translates ConfigError into typer.Exit — the CLI's
        # own boundary — rather than letting the raw exception surface.
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(
            cli_module.app, ["backtest", "--config", str(empty_dir), "--strategy", "sma_crossover"]
        )
        assert result.exit_code == 1
        assert "refusing to boot" in result.output

    def test_a_valid_directory_loads(self, tmp_path: Path) -> None:
        config = cli_module._load_config(write_config(tmp_path))
        assert config.universe.symbols == ("AAPL",)


class TestBuildStrategy:
    def test_sma_crossover_is_built_with_its_configured_params(self) -> None:
        sc = StrategyConfig(
            strategy_id="sma_crossover",
            enabled=True,
            universe=("AAPL",),
            params={"fast_period": 5, "slow_period": 10},
        )
        strategy, specs = cli_module._build_strategy(sc)
        assert isinstance(strategy, SmaCrossoverStrategy)
        assert strategy.fast_period == 5
        assert strategy.slow_period == 10
        assert len(specs) == 2

    def test_an_unknown_strategy_id_raises(self) -> None:
        sc = StrategyConfig(strategy_id="not_a_real_strategy", enabled=True, universe=("AAPL",))
        with pytest.raises(ATraderError):
            cli_module._build_strategy(sc)

    def test_llm_agent_is_explicitly_rejected_here(self) -> None:
        # backtest/replay do not support it; _build_strategies (run/paper)
        # routes llm_agent to _build_llm_strategy instead, never here.
        sc = StrategyConfig(strategy_id="llm_agent", enabled=True, universe=("AAPL",))
        with pytest.raises(ATraderError, match="llm_agent"):
            cli_module._build_strategy(sc)


class TestBuildStrategies:
    def test_disabled_strategies_are_skipped(self, tmp_path: Path) -> None:
        yaml = _STRATEGIES_YAML.replace("enabled: true", "enabled: false")
        config = cli_module._load_config(write_config(tmp_path, strategies_yaml=yaml))
        strategies, _ = cli_module._build_strategies(config, audit=None, clock=None)
        assert strategies == []

    def test_llm_agent_without_an_api_key_is_skipped_not_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        yaml = _STRATEGIES_YAML + (
            "  - strategy_id: llm_agent\n"
            "    enabled: true\n"
            "    capital_allocation_pct: 0\n"
            '    universe: ["AAPL"]\n'
        )
        config = cli_module._load_config(write_config(tmp_path, strategies_yaml=yaml))
        strategies, _ = cli_module._build_strategies(config, audit=None, clock=None)
        # sma_crossover still built; llm_agent silently skipped, not fatal.
        assert len(strategies) == 1
        assert isinstance(strategies[0], SmaCrossoverStrategy)

    def test_an_enabled_but_unknown_strategy_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        yaml = (
            "strategies:\n"
            "  - strategy_id: mystery_strategy\n"
            "    enabled: true\n"
            "    capital_allocation_pct: 0\n"
            '    universe: ["AAPL"]\n'
        )
        config = cli_module._load_config(write_config(tmp_path, strategies_yaml=yaml))
        strategies, _ = cli_module._build_strategies(config, audit=None, clock=None)
        assert strategies == []


class TestBacktestCommand:
    def test_runs_end_to_end_against_synthetic_data(self, tmp_path: Path) -> None:
        config_dir = write_config(tmp_path)
        result = runner.invoke(
            cli_module.app,
            [
                "backtest",
                "--config",
                str(config_dir),
                "--strategy",
                "sma_crossover",
                "--from",
                "2024-01-01",
                "--to",
                "2024-02-01",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "bars processed" in result.output
        assert "sharpe_ratio" in result.output

    def test_an_unconfigured_strategy_id_is_a_clean_error(self, tmp_path: Path) -> None:
        config_dir = write_config(tmp_path)
        result = runner.invoke(
            cli_module.app,
            ["backtest", "--config", str(config_dir), "--strategy", "does_not_exist"],
        )
        assert result.exit_code == 1

    def test_llm_agent_is_a_clean_error_not_a_crash(self, tmp_path: Path) -> None:
        yaml = _STRATEGIES_YAML + (
            "  - strategy_id: llm_agent\n"
            "    enabled: true\n"
            "    capital_allocation_pct: 0\n"
            '    universe: ["AAPL"]\n'
        )
        config_dir = write_config(tmp_path, strategies_yaml=yaml)
        result = runner.invoke(
            cli_module.app,
            ["backtest", "--config", str(config_dir), "--strategy", "llm_agent"],
        )
        assert result.exit_code == 1
        assert "llm_agent" in result.output


class TestReplayCommand:
    def test_a_deterministic_strategy_replays_byte_identical(self, tmp_path: Path) -> None:
        config_dir = write_config(tmp_path)
        result = runner.invoke(
            cli_module.app,
            [
                "replay",
                "--config",
                str(config_dir),
                "--strategy",
                "sma_crossover",
                "--from",
                "2024-01-01",
                "--to",
                "2024-01-15",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "byte-identical" in result.output

    def test_an_unconfigured_strategy_id_is_a_clean_error(self, tmp_path: Path) -> None:
        config_dir = write_config(tmp_path)
        result = runner.invoke(
            cli_module.app,
            ["replay", "--config", str(config_dir), "--strategy", "does_not_exist"],
        )
        assert result.exit_code == 1


class TestVerifyAuditCommand:
    def test_a_clean_offline_export_verifies(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        first = AuditRecord(
            seq=1,
            event_type="system.started",
            actor="system",
            payload={},
            created_at_ns=1,
            prev_hash=GENESIS_HASH,
            hash=b"",
        )
        first_hash = compute_hash(
            seq=first.seq,
            event_type=first.event_type,
            actor=first.actor,
            payload=first.payload,
            created_at_ns=first.created_at_ns,
            prev_hash=first.prev_hash,
        )
        first = replace(first, hash=first_hash)
        dump_records_jsonl([first], path)

        result = runner.invoke(cli_module.app, ["verify-audit", "--file", str(path)])
        assert result.exit_code == 0, result.output
        assert "valid=True" in result.output

    def test_a_tampered_export_fails_verification(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        bad = AuditRecord(
            seq=1,
            event_type="system.started",
            actor="system",
            payload={},
            created_at_ns=1,
            prev_hash=GENESIS_HASH,
            hash=b"not the real hash",
        )
        dump_records_jsonl([bad], path)

        result = runner.invoke(cli_module.app, ["verify-audit", "--file", str(path)])
        assert result.exit_code == 1
        assert "valid=False" in result.output

    def test_an_unreachable_api_url_is_a_clean_error(self) -> None:
        result = runner.invoke(cli_module.app, ["verify-audit", "--api-url", "http://127.0.0.1:1"])
        assert result.exit_code == 1


class _StubApiHandler(BaseHTTPRequestHandler):
    """Minimal stand-in for `atrader.app.api`'s FastAPI app — just enough to
    exercise `kill`/`reconcile`'s HTTP client code without the full Runtime
    stack (which `test_runtime.py`/`test_api.py` already cover directly)."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        request_body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/kill":
            body = {"engaged": True, "reason": "cli test", "source": "http"}
        elif self.path == "/reconcile":
            injected = bool(request_body.get("inject_break"))
            body = {
                "clean": not injected,
                "breaks": ["synthetic break"] if injected else [],
                "positions_checked": 1 if injected else 0,
                "orders_checked": 0,
            }
        else:
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass  # keep test output quiet


@pytest.fixture
def stub_api_url():  # type: ignore[no-untyped-def]
    server = HTTPServer(("127.0.0.1", 0), _StubApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestKillAndReconcileCommands:
    def test_kill_reaches_the_api_and_prints_the_response(self, stub_api_url: str) -> None:
        result = runner.invoke(
            cli_module.app, ["kill", "--reason", "smoke test", "--api-url", stub_api_url]
        )
        assert result.exit_code == 0
        assert '"engaged": true' in result.output

    def test_reconcile_reaches_the_api_and_exits_zero_when_clean(self, stub_api_url: str) -> None:
        result = runner.invoke(cli_module.app, ["reconcile", "--api-url", stub_api_url])
        assert result.exit_code == 0
        assert '"clean": true' in result.output

    def test_reconcile_inject_break_exits_nonzero_when_dirty(self, stub_api_url: str) -> None:
        result = runner.invoke(
            cli_module.app, ["reconcile", "--inject-break", "--api-url", stub_api_url]
        )
        assert result.exit_code == 1
        assert '"clean": false' in result.output

    def test_kill_against_an_unreachable_url_is_a_clean_error(self) -> None:
        result = runner.invoke(
            cli_module.app, ["kill", "--reason", "x", "--api-url", "http://127.0.0.1:1"]
        )
        assert result.exit_code == 1


class TestDivergenceReportCommand:
    """Acceptance criterion #7's instrument, wired end to end: a backtest
    writes its curve, and the report reads two of them back and returns an
    exit code a promotion step can gate on."""

    def _write_curve(self, path: Path, equities: list[str]) -> Path:
        dump_equity_curve(
            [(i * 86_400 * 1_000_000_000, Decimal(e)) for i, e in enumerate(equities)], path
        )
        return path

    def test_backtest_can_write_the_curve_the_report_reads(self, tmp_path: Path) -> None:
        config_dir = write_config(tmp_path)
        curve_path = tmp_path / "backtest-equity.jsonl"
        result = runner.invoke(
            cli_module.app,
            [
                "backtest",
                "--config",
                str(config_dir),
                "--from",
                "2024-01-01",
                "--to",
                "2024-02-01",
                "--equity-out",
                str(curve_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert curve_path.is_file()
        assert len(load_equity_curve(curve_path)) > 1

    def test_matching_curves_exit_zero(self, tmp_path: Path) -> None:
        equities = [str(100000 + 100 * i) for i in range(30)]
        backtest = self._write_curve(tmp_path / "bt.jsonl", equities)
        live = self._write_curve(tmp_path / "live.jsonl", equities)
        result = runner.invoke(
            cli_module.app,
            ["divergence-report", "--backtest", str(backtest), "--live", str(live)],
        )
        assert result.exit_code == 0, result.output
        assert "WITHIN TOLERANCE" in result.output

    def test_a_run_that_earned_half_as_much_exits_nonzero(self, tmp_path: Path) -> None:
        backtest = self._write_curve(
            tmp_path / "bt.jsonl", [str(100000 + 100 * i) for i in range(30)]
        )
        live = self._write_curve(tmp_path / "live.jsonl", [str(100000 + 50 * i) for i in range(30)])
        result = runner.invoke(
            cli_module.app,
            ["divergence-report", "--backtest", str(backtest), "--live", str(live)],
        )
        assert result.exit_code == 1
        assert "OUT OF TOLERANCE" in result.output

    def test_raising_the_tolerance_changes_the_verdict(self, tmp_path: Path) -> None:
        backtest = self._write_curve(
            tmp_path / "bt.jsonl", [str(100000 + 100 * i) for i in range(30)]
        )
        live = self._write_curve(tmp_path / "live.jsonl", [str(100000 + 90 * i) for i in range(30)])
        strict = runner.invoke(
            cli_module.app,
            [
                "divergence-report",
                "--backtest",
                str(backtest),
                "--live",
                str(live),
                "--tolerance",
                "5",
            ],
        )
        lenient = runner.invoke(
            cli_module.app,
            [
                "divergence-report",
                "--backtest",
                str(backtest),
                "--live",
                str(live),
                "--tolerance",
                "30",
            ],
        )
        assert strict.exit_code == 1
        assert lenient.exit_code == 0, lenient.output

    def test_a_missing_curve_file_is_a_clean_error_not_a_traceback(self, tmp_path: Path) -> None:
        live = self._write_curve(tmp_path / "live.jsonl", ["100000", "100100"])
        result = runner.invoke(
            cli_module.app,
            ["divergence-report", "--backtest", str(tmp_path / "nope.jsonl"), "--live", str(live)],
        )
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_a_corrupt_curve_file_is_a_clean_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("this is not json\n", encoding="utf-8")
        live = self._write_curve(tmp_path / "live.jsonl", ["100000", "100100"])
        result = runner.invoke(
            cli_module.app,
            ["divergence-report", "--backtest", str(bad), "--live", str(live)],
        )
        assert result.exit_code == 1
