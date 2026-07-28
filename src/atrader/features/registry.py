"""Feature version registry — spec §8.2.

    피처 정의가 바뀌면(예: SMA 계산 로직 수정) 이전 버전으로 만든 백테스트
    결과와 새 버전으로 만든 결과를 섞으면 안 된다. 피처 버전 불일치 시
    시작을 거부한다.

A strategy tuned against SMA-v1 and then silently handed SMA-v2 numbers is not
running the strategy that was validated — it is running an unvalidated
variant that happens to share a name. The registry makes that mismatch a boot
failure instead of a silent behaviour change: a strategy declares the feature
versions it expects, and :meth:`FeatureRegistry.validate_required` refuses to
start the system if what is actually registered disagrees.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from atrader.core.errors import ConfigError

__all__ = ["FeatureDefinition", "FeatureRegistry"]


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    name: str
    version: int
    description: str = ""


class FeatureRegistry:
    """What features exist and which version of each is currently computed."""

    __slots__ = ("_by_name",)

    def __init__(self, definitions: Iterable[FeatureDefinition] = ()) -> None:
        self._by_name: dict[str, FeatureDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: FeatureDefinition) -> None:
        existing = self._by_name.get(definition.name)
        if existing is not None and existing.version != definition.version:
            raise ConfigError(
                f"feature {definition.name!r} registered twice with different "
                f"versions ({existing.version} vs {definition.version}) — two "
                "components disagree about which implementation is running"
            )
        self._by_name[definition.name] = definition

    def version_of(self, name: str) -> int:
        definition = self._by_name.get(name)
        if definition is None:
            raise ConfigError(f"unknown feature {name!r}; register it before use")
        return definition.version

    def known_features(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def validate_required(self, required: dict[str, int]) -> None:
        """Boot-time check (spec §7.1's fail-closed principle, applied to features).

        *required* maps a feature name to the version a strategy was built and
        validated against. Anything missing or mismatched refuses startup —
        preferable to producing signals from a feature nobody has verified.
        """
        problems: list[str] = []
        for name, expected_version in sorted(required.items()):
            definition = self._by_name.get(name)
            if definition is None:
                problems.append(f"{name}: not registered")
            elif definition.version != expected_version:
                problems.append(
                    f"{name}: strategy expects v{expected_version}, "
                    f"registry has v{definition.version}"
                )
        if problems:
            raise ConfigError(
                "feature version mismatch — refusing to start: " + "; ".join(problems)
            )
