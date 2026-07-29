# 운영 런북

이 문서는 `atrader` CLI로 시스템을 기동·감시·정지하는 절차를 다룬다. 아키텍처 설명은
`aitradingagentspec.md`와 각 모듈의 docstring에 있으므로 여기서는 반복하지 않는다.

## 0. 시작하기 전에 — 이 시스템이 실제로 하는 일

**브로커는 모의 브로커(`PaperBroker`) 하나뿐이다.** 실거래소 어댑터는 이번 범위 밖이고
(`docs/llm-determinism.md`와 마찬가지로, 처음 계획 단계에서 확정한 결정이다), 그래서
`atrader run`과 `atrader paper`는 사실상 같은 동작을 한다 — 유일한 차이는 `paper`가
기본적으로 `--duration 60`(초)로 종료 시점을 두는 것뿐이다. "실거래처럼 보이는 명령"이
있다고 해서 실제 주문이 거래소로 나가지는 않는다.

## 1. 기동 전 체크리스트

```bash
uv sync --all-extras
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/
uv run lint-imports                                   # 인수기준 #1
uv run pytest -m "not integration" --cov=atrader
```

`config/` 아래 5개 파일(`risk.yaml`이 필수, 나머지는 선택)이 있어야 한다. `risk.yaml`이
없거나 파싱에 실패하면 **부팅을 거부한다**(spec §7.1) — 이것은 버그가 아니라 설계다.

### 테스트 5계층과 인수기준 대응표

| 계층 | 위치 | 인수기준 |
|---|---|---|
| 단위 | `tests/unit/` (전체 80%, `tests/unit/test_risk_coverage.py`가 리스크엔진+상태머신 95%를 별도 프로세스로 측정해 강제) | #1(임포트 경계), #2(커버리지) |
| 속성(Hypothesis) | `tests/property/` — 리스크 한도 우회 불가(`test_risk_invariants.py`), 상태머신 64개 전이 쌍 전수 검증(`test_statemachine_invariants.py`), 포지션=체결합(`test_accounting_invariants.py`) | #2 |
| 통합 | `tests/integration/test_paper_broker.py` — 부분체결·거부·지연·순서 뒤바뀜 | — |
| 카오스 | `tests/chaos/` — `test_kill_after_send.py`(주문 전송 직후 kill), `test_reordered_events.py` | #4 |
| 리플레이 | `tests/replay/test_replay.py` — 녹화 재생 바이트 비교, 결정론이 실제로 깨지는 음성 케이스 포함 | #3 |
| 프롬프트 인젝션 | `tests/unit/test_prompt_injection.py` | #10 |
| 킬스위치 타이밍 | `tests/unit/test_killswitch_timing.py` — 실제 벽시계 시간으로 측정(시뮬레이션 시계는 이 기준에서 아무 의미가 없다) | #5 |

```bash
uv run pytest tests/unit/test_risk_coverage.py         # 인수기준 #2: 95% 강제
uv run pytest tests/replay -q                          # 인수기준 #3
uv run pytest tests/chaos -q                           # 인수기준 #4
uv run pytest tests/unit/test_killswitch_timing.py      # 인수기준 #5
uv run pytest tests/unit/test_prompt_injection.py       # 인수기준 #10
```

인수기준 #7(30일 페이퍼 트레이딩 괴리 <30%)과 #8(RTO 5분)은 시간·인프라가 필요해
자동화된 테스트로 검증할 수 없다 — 계획 단계에서부터 이번 세션 범위 밖으로 확정했다.
**측정 도구는 구현돼 있고, 실행만 남았다**:

- #7 → `atrader divergence-report` (아래 6절). `atrader.backtest.divergence`가
  계산 주체이며 `tests/unit/test_divergence.py`가 100% 커버한다.
- #8 → `atrader reconcile`(§4)과 `Reconciler`가 재기동 시 상태 대조 절차의 구현체다.

두 경우 모두 **도구가 있다는 것과 기준을 충족했다는 것은 다르다**. 실제 30일 실행과
실제 복구 훈련은 운영 단계의 몫이다.

## 2. 페이퍼 트레이딩 실행

```bash
uv run atrader run    --config config --host 127.0.0.1 --port 8000              # 무기한 실행
uv run atrader paper  --config config --port 8000 --duration 60                 # 60초 스모크 테스트
```

`run`/`paper`는 같은 프로세스 안에서 두 가지를 동시에 돈다:

1. **틱 루프** (`Runtime.run_forever`/`run_for`) — 피드 → 피처 → 전략 → 넷팅 → 서킷브레이커
   → 리스크 게이트 → OMS → 모의 브로커 → 포지션북 → 마진 모니터, 매 바마다.
2. **HTTP 제어면** (`atrader.app.api`, uvicorn) — 아래 5절.

기본 전략 구성은 `config/strategies.yaml`을 따른다(`sma_crossover` 활성, `llm_agent`는
`ANTHROPIC_API_KEY`가 없으면 자동으로 건너뛰고 WARN 로그만 남긴다 — 프로세스 전체가
죽지 않는다).

### 스모크 테스트 시 주의할 점: 바 간격

`config/app.yaml`의 `market_data.bar_intervals` 첫 번째 값(`"1m"`)이 이 런타임이 실제로
거래하는 간격이다. **`--duration 3` 같은 짧은 실행은 1분봉이 단 하나도 닫히지 않아
`bars_processed: 0`으로 끝난다 — 이것은 버그가 아니라 3초 안에 1분이 지나지 않았을 뿐이다.**
플러밍(피드→API가 응답하는지, kill/reconcile이 동작하는지)만 확인하려면 아무 지속시간이나
괜찮다. 실제로 봉이 닫히고 주문이 나가는지 보려면 최소 수 분 이상 돌리거나, 테스트 전용
설정에서 `bar_intervals: ["1s"]`로 낮춰라.

## 3. 킬 스위치 (spec §FR-MON-03, 인수기준 #5)

세 경로 — UI(별도 구현), CLI, HTTP — 모두 같은 `KillSwitch` 인스턴스를 움직인다. 어느
경로로 걸든 (1) 신규 주문 즉시 차단 (2) 미체결 전량 취소, 이 순서로 5초 안에 끝난다.

```bash
uv run atrader kill --reason "이상 감지" [--liquidate] [--api-url http://127.0.0.1:8000]
```

`--liquidate`를 주면 취소 뒤에 남은 포지션도 전량 청산 주문을 낸다 — 이 청산 주문도
`reduce_only=True`로 정상적인 리스크 게이트를 통과해 나간다(손절 방향 예외, spec §7.4).
해제(재개)는 CLI에 없다 — `DELETE /kill`을 직접 호출하거나 UI를 쓴다. 자동 재개는 없다
(사람이 원인을 이해하기 전에는 재개하지 않는다).

서킷브레이커 L3가 발동하면 런타임이 **자동으로** 킬 스위치를 건다
(`source=AUTOMATIC`, `liquidate=True`) — 이 하나만 예외적으로 사람 개입 없이 발동한다.

## 4. 정합성 확인 (spec §FR-EXE-05, 인수기준 #6)

부팅 시 무조건 한 번, 이후 `execution.reconciliation_interval_seconds`(기본 30초)마다
자동으로 돈다. **불일치를 발견해도 자동으로 고치지 않는다** — 신규 주문만 막고 사람을
기다린다.

```bash
uv run atrader reconcile [--inject-break] [--api-url http://127.0.0.1:8000]
```

`--inject-break`는 브로커가 모르는 가짜 로컬 포지션(`__diagnostic_break__`)을 하나
심어서 그 경로가 실제로 불일치를 잡아내는지 증명하는 진단 스위치다 — 운영 환경에서
쓸 일은 없다.

## 5. HTTP 제어면

| 경로 | 메서드 | 설명 |
|---|---|---|
| `/health` | GET | `RuntimeStatus` 전체 |
| `/kill` | POST / DELETE | 킬 스위치 발동 / 해제 |
| `/approve` | GET | 대기 중인 승인 요청 목록 |
| `/approve/{id}` | POST / DELETE | 승인 / 거부 |
| `/positions` | GET | 현재 보유 포지션 |
| `/pnl` | GET | 실현/미실현 손익, 자기자본 |
| `/reconcile` | POST | 즉시 정합성 확인 (§4) |
| `/audit` | GET | 감사로그 전체 (해시체인 검증용) |
| `/metrics` | GET | Prometheus 텍스트 포맷 |

사양서가 명시한 5개(`/health /kill /approve /positions /pnl`) 외의 세 경로는 CLI의
`reconcile`/`verify-audit` 명령과 모니터링이 실제로 살아있는 프로세스에 닿을 방법이
필요해서 추가했다 — 각각의 근거는 위 절과 6절에 있다.

## 6. 감사로그 검증 (spec §9.3, 인수기준 #9)

```bash
uv run atrader verify-audit                              # 실행 중인 프로세스에서 조회
uv run atrader verify-audit --file recordings/audit.jsonl # 오프라인 내보내기 검증
```

내보내기는 `atrader.audit.hashchain.dump_records_jsonl(sink.read_all(), path)`로 만든다.
7년 보관 요구(§9.3)에 맞춰 이 JSONL을 정기적으로 별도 저장소에 옮기는 절차는 이번
슬라이스 범위 밖이다 — 파일 하나 내보내고 검증하는 수단만 갖췄다.

## 7. 백테스트 / 리플레이

```bash
uv run atrader backtest --config config --strategy sma_crossover \
  --from 2024-01-01 --to 2024-12-31 [--bars recordings/aapl_2024.jsonl]

uv run atrader replay --config config --strategy sma_crossover \
  --from 2024-01-01 --to 2024-06-01 [--bars recordings/aapl_2024.jsonl]
```

`--bars`를 주지 않으면 **실제 시세가 아닌 합성 데이터**(`SimulatedFeed` + `BarAggregator`)를
생성해서 돌린다 — 이 벡터슬라이스에는 실제 과거 데이터 소스가 없기 때문이며, CLI가 그
사실을 실행 시점에 그대로 출력한다. 전략의 실제 성과를 판단하는 용도가 아니라 파이프라인
(설정 로드 → 전략 → 리스크 → 체결 → 지표 계산)이 실제로 동작하는지 확인하는 용도다.
진짜 성과 검증은 실제로 기록된 틱/바 파일(`--bars`, 또는
`atrader.marketdata.feeds.replay.write_ticks`로 만든 레코딩)로 해야 한다.

`replay`는 같은 설정으로 엔진을 **두 번 새로 만들어** 돌리고 주문 시퀀스를 바이트 단위로
비교한다(인수기준 #3) — 한 인스턴스를 재사용하면 아무것도 증명하지 못한다는 점은
`atrader.backtest.replay`의 자체 테스트(`test_replay.py`)가 의도적으로 결정론이 깨지는
케이스를 만들어 검증해 둔 것과 같은 이유다.

`llm_agent` 전략은 `backtest`/`replay`에서 아직 지원하지 않는다 — 실 API 호출 없이
재현 가능하게 돌리려면 `RecordedLLMClient`에 미리 녹화된 응답이 필요한데, 이걸 CLI
옵션으로 노출하는 건 이번 범위 밖이다. 필요하면 `atrader.backtest.engine.BacktestEngine`을
직접 스크립트에서 조립해 `RecordedLLMClient`를 넘겨라.

## 7-1. 백테스트 대비 실거래 괴리 측정 (인수기준 #7)

인수기준 #7은 "30일 페이퍼 트레이딩 결과가 백테스트 대비 괴리 30% 미만"이다. 30일이
실제로 흘러야 하므로 여기서 충족시킬 수는 없지만, **판정하는 도구는 구현돼 있다**.
운영에서 30일을 돌린 뒤 아래 세 단계로 답이 나온다.

```bash
# 1. 백테스트 자산곡선 기록
uv run atrader backtest --config config --strategy sma_crossover \
  --from 2024-01-01 --to 2024-01-31 --equity-out runs/backtest-equity.jsonl

# 2. 페이퍼 트레이딩 자산곡선 기록 (30일 실행)
uv run atrader run --config config --equity-out runs/live-equity.jsonl

# 3. 괴리 판정 — 초과 시 exit code 1 (승격 게이트로 쓸 수 있다)
uv run atrader divergence-report \
  --backtest runs/backtest-equity.jsonl --live runs/live-equity.jsonl [--tolerance 30]
```

`--equity-out`은 **일별 종가 자산 1점**만 기록한다. 바 단위로 기록하면 1초 바 기준
30일이 수백만 포인트가 되고, 애초에 인수기준도 `performance_report`의 기본값(연 252
기간)도 전부 "일" 단위로 진술돼 있다. 이 옵션은 `finally` 블록에서 쓰이므로 Ctrl-C나
킬 스위치로 끝난 세션도 곡선을 남긴다.

읽을 때 주의할 것 세 가지 (`atrader.backtest.divergence` 모듈 docstring에 근거와 함께
정리돼 있다):

| 상황 | 리포트의 처리 |
|---|---|
| 백테스트 값이 0, 실거래 값은 0이 아님 | **비교 불가**(`relative_pct=None`). 0으로 나눈 값은 "무한대"가 아니라 "수치 아님"이다. 통과로도 실패로도 세지 않으며, 게이트 대상이면 `within_tolerance`가 False가 된다 |
| 백테스트 값도 0, 실거래 값도 0 | 괴리 정확히 0 — 두 값이 완전히 일치하는 것을 "비교 불가"로 처리하면 맞는 실행을 형식논리로 떨어뜨리게 된다 |
| 두 실행의 기간 길이가 2배 넘게 차이 | `periods_aligned=False`. 1년 백테스트와 30일 실거래의 총수익률 비교는 전략 충실도와 무관한 이유로 다르므로, 판정 자체를 거부한다 |

게이트 대상은 기본적으로 `total_return_pct`·`sharpe_ratio`·`max_drawdown_pct` 셋뿐이다
(각각 "결과를 예측했나 / 과정을 예측했나 / 최악의 순간을 예측했나"). 체결 횟수와
변동성은 **원인 분석용으로 출력만 되고 판정에는 쓰이지 않는다** — 10번 대신 8번
거래해서 같은 결과를 냈다면 그건 기준 미달이 아니다.

## 8. LLM 에이전트 전략

`config/strategies.yaml`의 `llm_agent`는 기본 `enabled: false`다. 켜려면:

1. `ANTHROPIC_API_KEY` 환경변수를 설정한다 — 없으면 CLI가 WARN 로그를 남기고 그 전략만
   건너뛴다(전체 프로세스는 계속 돈다).
2. `docs/llm-determinism.md`를 반드시 읽는다 — 사양서의 `temperature=0` 요구가 현재
   Claude 모델에서 어떻게 대체되는지, `RecordedLLMClient`가 왜 실제 결정론 확보 수단인지
   설명한다.
3. 프롬프트 인젝션 방어는 `atrader.strategy.llm.prompt`(태그 격리)와
   `atrader.strategy.llm.guards`(화이트리스트·수치정합성·신뢰도) 이중 구조다 — 후자가
   진짜 방어선이다. `tests/unit/test_llm_agent.py::TestPromptInjectionResistance`가
   모델이 조작당한 상황을 가정하고도 가드가 막아내는지 직접 증명한다.

## 9. 알려진 단순화 (이번 벡터슬라이스 범위)

- **ADV**: 실시간 관측 거래량의 이동평균일 뿐, 진짜 20일 추세 ADV가 아니다(과거 데이터가
  없다).
- **상관계수 기반 노출 한도**: 구현되지 않았다 — `RiskSnapshot.correlations`가 항상
  비어 있어 그 체크는 절대 발동하지 않는다.
- **장 시간 확인**: `InstrumentSpec`의 UTC 시:분만 비교한다 — 공휴일 캘린더 없음.
- **레이트리미터와 취소**: 킬 스위치/데드맨의 전량취소는 레이트리미터를 거치지 않는다
  (의도적으로 막지 않음 — 실패 방향이 "더 많이 허용"이라 안전 쪽으로 치우친 단순화다).
- **저장소**: 기본은 인메모리(`InMemoryStorage`) — 프로세스가 죽으면 상태도 사라진다.
  SQL 리포지토리(`atrader.storage.sql`)는 구현되어 있으나 이 CLI가 아직 배선하지 않았다.

이 목록에 없는 것 — 리스크 게이트, 킬 스위치, 정합성 확인, 감사로그 해시체인, 주문
멱등성 — 은 단순화 대상이 아니다. 각각 독립적으로 테스트되어 있고 이번 런타임에도
그대로 배선되어 있다.
