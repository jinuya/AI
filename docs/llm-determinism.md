# LLM 결정론: 사양서 §7.7·§13과 현재 Claude API의 충돌

## 문제

사양서는 두 곳에서 결정론을 요구한다.

§2.2 (설계 원칙):
> 같은 입력 이벤트 시퀀스를 넣으면 같은 주문이 나와야 한다. [...] 그래서 랜덤 시드 고정,
> LLM temperature=0, 모든 외부 호출 응답 로깅이 필수다.

§7.7 (AI 안전장치):
> **결정론.** temperature=0, top_p=1, 시드 고정.

§13 부록의 `risk.llm` 블록도 `temperature: 0`을 명시한다.

**그러나 현재 Claude 모델군에서는 이 파라미터들이 존재하지 않는다.** `temperature`,
`top_p`, `top_k`는 Claude Opus 5 / Sonnet 5 / Opus 4.8 / Opus 4.7 / Fable 5에서 제거되어
**요청에 포함하면 HTTP 400**이 반환된다. 시드 파라미터는 애초에 제공된 적이 없다.

즉 사양서의 지시를 문자 그대로 구현하면 **모든 LLM 호출이 실패한다.**

## 해소 방법

사양서가 `temperature=0`으로 달성하려던 *목적*은 두 가지다.

1. **재현 가능한 백테스트** — 같은 데이터로 돌리면 같은 결과가 나와야 백테스트를 믿는다.
2. **사후 원인 분석** — 사고가 났을 때 "그때 왜 그 판단을 했는가"를 재구성할 수 있어야 한다.

둘 다 샘플링 파라미터 없이 달성 가능하며, 사실 그쪽이 더 강한 보장을 준다.

### 1. 녹화/재생 (`RecordedLLMClient`)

모든 요청과 응답을 정규화된 요청 해시를 키로 저장한다. 재생 모드에서는 API를 호출하지
않고 저장된 응답을 그대로 돌려준다.

- 백테스트와 리플레이는 **완전히** 결정론적이다. `temperature=0`으로도 얻을 수 없는 수준이다
  (temperature=0은 결정론을 보장한 적이 없다 — 부동소수점 비결정성과 서버 측 배치 효과가 있다).
- 인수기준 #3(결정론적 리플레이 바이트 비교)의 실제 구현 수단이 이것이다.
- 녹화 파일 자체가 사양서 §7.7 "전량 로깅" 요구의 산출물이 된다.

### 2. 출력 편차 모니터링

사양서 §7.7은 이미 이렇게 적고 있다:

> 그래도 완전히 결정론적이진 않으므로, 동일 입력에 대한 출력 편차를 모니터링한다.
> 편차가 커지면 모델 버전 변경을 의심한다.

이 요구는 그대로 유효하며, 오히려 유일한 신뢰 가능한 신호가 된다. `LLMConfig.variance_alert_threshold`
(기본 0.2)를 넘으면 WARN을 낸다.

### 3. `effort`로 깊이 제어

`temperature` 대신 `output_config.effort`(`low`|`medium`|`high`|`xhigh`|`max`)를 쓴다.
매매 의도 생성 기본값은 `medium`이다.

### 4. 사고(thinking)는 켜둔다

`thinking: {"type": "disabled"}`는 **쓰지 않는다.** Claude Opus 5에서 사고를 끄면 두 가지
알려진 실패 모드가 있다:

- 도구 호출이 구조화 블록 대신 **평문 텍스트로 새어나온다.** 턴은 성공으로 끝나고 호출은
  실행되지 않으며 에러도 나지 않는다 — 자동매매 루프에서 조용히 아무 일도 안 하는 최악의 형태다.
- `<thinking>` 태그가 사용자 응답에 유출된다.

Opus 5는 사고가 기본값으로 켜져 있으므로 `thinking` 필드를 건드리지 않는 것이 맞다.
비용이 걱정되면 `effort`를 낮춘다.

## 구현상의 강제

`LLMConfig`의 검증기가 `SAMPLING_FREE_MODELS`에 속한 모델에 대해 `temperature`/`top_p`/`seed`가
설정되면 **부팅을 거부한다.** 필드 자체는 사양서 추적성을 위해 스키마에 남겨두되, 잘못 켜서
운영 중 400을 맞는 일이 없도록 로드 시점에 막는다.

```
ValueError: temperature cannot be used with model 'claude-opus-5': these parameters
were removed from the Claude API and sending them returns HTTP 400. Spec §13 asks for
temperature=0 to get determinism; on current models that is achieved with
recorded/replayed responses instead — see docs/llm-determinism.md.
Control depth with `effort`.
```

## 모델 버전 핀 고정

사양서 §7.7:
> **모델 버전 핀 고정.** 날짜가 포함된 정확한 모델 버전을 명시한다. 별칭(alias)을 쓰면
> 어느 날 갑자기 모델이 바뀌어서 전략 성격이 달라진다.

현재 Claude 모델 ID에는 날짜 접미사가 없다 — `claude-opus-5`가 정확한 ID이며, 날짜를 붙이면
404가 난다. 따라서 같은 보장을 **명시적 허용목록**으로 구현한다: `KNOWN_MODEL_IDS`에 없는
문자열은 부팅 시 거부된다. 새 모델을 쓰려면 허용목록에 추가하는 리뷰된 변경이 필요하고,
그게 곧 "의도적으로 모델을 바꿨다"는 기록이 된다.

추가로 런타임에 응답의 `response.model`을 요청한 모델과 대조해, 서버 측에서 다른 모델이
응답했으면 WARN을 낸다(폴백이 발동한 경우가 여기 해당한다).

## 요약

| 사양서 요구 | 현재 구현 | 이유 |
|---|---|---|
| `temperature: 0` | 전송하지 않음, 설정 시 부팅 거부 | API에서 제거됨, 400 |
| `top_p: 1` | 동일 | 동일 |
| 시드 고정 | 해당 파라미터 없음 | API 미제공 |
| 결정론적 재현 | `RecordedLLMClient` 녹화/재생 | 더 강한 보장 |
| 출력 편차 모니터링 | `variance_alert_threshold` | 사양서 그대로 유효 |
| 모델 버전 핀 고정 | `KNOWN_MODEL_IDS` 허용목록 + 응답 모델 대조 | 날짜 접미사 부재 |
| 전량 로깅 | 감사 로그 + 녹화 파일 | 사양서 그대로 |
