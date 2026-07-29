"""Secret resolution.

Spec §6.2: API keys are fetched at runtime, held only in memory, and never
written to a log. Hardcoding them in source or baking them into an image is out.

The providers here cover what can run without external infrastructure. A Vault
or AWS Secrets Manager provider is a drop-in implementation of
:class:`SecretProvider` — the rest of the system only ever sees the Protocol, so
adding one changes nothing outside this module.

Two habits this module enforces:

* :class:`Secret` wraps the value so an accidental ``print``/``repr``/f-string —
  the usual way a key reaches a log — shows ``Secret(ANTHROPIC_API_KEY)`` rather
  than the key itself. You have to call :meth:`Secret.reveal` on purpose.
* :func:`redact` is used by the logging filter (spec §9.3) as a second line of
  defence for values that arrive as plain strings from elsewhere.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from atrader.core.errors import ConfigError

__all__ = [
    "ChainSecretProvider",
    "EnvSecretProvider",
    "FileSecretProvider",
    "Secret",
    "SecretProvider",
    "StaticSecretProvider",
    "redact",
]


class Secret:
    """A secret value that does not leak through ``repr`` or ``str``."""

    __slots__ = ("_name", "_value")

    def __init__(self, name: str, value: str) -> None:
        self._name = name
        self._value = value

    @property
    def name(self) -> str:
        return self._name

    def reveal(self) -> str:
        """Return the raw value. Call sites should be few and obvious."""
        return self._value

    def __repr__(self) -> str:
        return f"Secret({self._name})"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Secret):
            return NotImplemented
        return self._name == other._name and self._value == other._value

    def __hash__(self) -> int:
        return hash((self._name, self._value))


@runtime_checkable
class SecretProvider(Protocol):
    """Source of secrets."""

    def get(self, name: str) -> Secret:
        """Return the named secret, or raise :class:`ConfigError` if absent."""
        ...

    def try_get(self, name: str) -> Secret | None:
        """Return the named secret, or ``None`` if it is not configured."""
        ...


class _BaseProvider:
    def get(self, name: str) -> Secret:
        found = self.try_get(name)
        if found is None:
            raise ConfigError(
                f"secret {name!r} is not available from {type(self).__name__}. "
                "Secrets are resolved at runtime and never hardcoded (spec §6.2)."
            )
        return found

    def try_get(self, name: str) -> Secret | None:  # pragma: no cover — overridden
        raise NotImplementedError


class EnvSecretProvider(_BaseProvider):
    """Read secrets from environment variables.

    Adequate for development. In production prefer a provider whose values are
    not visible in ``/proc/<pid>/environ`` or a crash dump.
    """

    __slots__ = ("_environ",)

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        import os

        self._environ = dict(os.environ) if environ is None else dict(environ)

    def try_get(self, name: str) -> Secret | None:
        value = self._environ.get(name)
        return None if value is None else Secret(name, value)


class FileSecretProvider(_BaseProvider):
    """Read secrets from one-file-per-secret directories (Docker/k8s style)."""

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root = root

    def try_get(self, name: str) -> Secret | None:
        path = self._root / name
        if not path.is_file():
            return None
        try:
            # rstrip: a trailing newline from `echo secret > file` is the single
            # most common cause of a mysteriously invalid credential.
            return Secret(name, path.read_text(encoding="utf-8").rstrip("\n"))
        except OSError as exc:
            raise ConfigError(f"cannot read secret {name} from {path}: {exc}") from exc


class StaticSecretProvider(_BaseProvider):
    """In-memory provider for tests."""

    __slots__ = ("_values",)

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self._values = dict(values or {})

    def try_get(self, name: str) -> Secret | None:
        value = self._values.get(name)
        return None if value is None else Secret(name, value)


class ChainSecretProvider(_BaseProvider):
    """Try each provider in order and return the first hit."""

    __slots__ = ("_providers",)

    def __init__(self, *providers: SecretProvider) -> None:
        if not providers:
            raise ConfigError("ChainSecretProvider needs at least one provider")
        self._providers = providers

    def try_get(self, name: str) -> Secret | None:
        for provider in self._providers:
            found = provider.try_get(name)
            if found is not None:
                return found
        return None


# ---------------------------------------------------------------------------
# Redaction (spec §9.3)
# ---------------------------------------------------------------------------

_REDACTION = "***REDACTED***"

#: Patterns that look like credentials regardless of the key they arrived under.
_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}", re.IGNORECASE),
)

#: Field names whose values are always masked, whatever they contain.
SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "apikey",
        "secret",
        "secret_key",
        "password",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "account_number",
        "account_no",
        "client_secret",
        "private_key",
    }
)


def redact(value: str) -> str:
    """Mask anything in *value* that looks like a credential."""
    for pattern in _VALUE_PATTERNS:
        value = pattern.sub(_REDACTION, value)
    return value


def is_sensitive_key(key: str) -> bool:
    """True when a field with this name should have its value masked."""
    normalised = key.lower().replace("-", "_")
    return normalised in SENSITIVE_KEYS or any(
        normalised.endswith(f"_{suffix}") for suffix in ("key", "secret", "token", "password")
    )
