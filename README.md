# atrader — AI 트레이딩 에이전트

사양서 `aitradingagentspec.md` v1.0 구현. 안전이 수익보다 우선하는 자동 매매 시스템.

## 핵심 불변식

**AI 모델은 절대 브로커 API를 직접 호출하지 않는다.** 전략은 스키마가 강제된 `TradingIntent`만
내놓고, 그 뒤의 결정론적 리스크 엔진이 실제 주문 여부를 판단한다. 이 경계는 정적으로 강제된다
(`pyproject.toml`의 import-linter 계약 + `tests/unit/test_import_boundaries.py`).

```
Market Data ──▶ Feature/Signal ──▶ Strategy ──┐
                                              │ TradingIntent
                                              ▼
                                    ┌──────────────────┐
                                    │   RISK ENGINE    │ ◀── 하드 게이트, 우회 불가
                                    └────────┬─────────┘
                                             │ Order
                                             ▼
                                    Execution / OMS ──▶ Broker Adapter
```

## 개발 환경

```bash
uv sync --all-extras          # 의존성 설치
make check                    # ruff + mypy + import-linter + pytest
```

외부 인프라(Postgres/Redis/Docker) 없이 전 테스트가 실행된다. 실백엔드 경로는
`pytest -m integration`으로 옵트인.

## 명령

```bash
uv run atrader backtest --strategy sma_crossover --from 2024-01-01 --to 2024-12-31
uv run atrader paper --config config/
uv run atrader kill --reason "..."      # 킬 스위치
uv run atrader reconcile                # 브로커 대조
uv run atrader verify-audit             # 감사 로그 해시 체인 검증
```

## 문서

- `docs/architecture.md` — 컴포넌트 경계와 데이터 흐름
- `docs/llm-determinism.md` — 사양서 §7.7의 `temperature=0` 요구와 현재 Claude API의 충돌, 그 해소 방법
- `docs/runbook.md` — 장애 시나리오별 대응
