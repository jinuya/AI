"""Secret resolution and masking — spec §6.2, §9.3.

    API 키는 런타임에 조회하고 메모리에만 두며 로그에 절대 쓰지 않는다.

This is the module ``.gitignore`` silently swallowed (see
``tests/unit/test_repo_completeness.py``), and it was sitting at 68% when that
came to light: ``redact``/``is_sensitive_key`` were exercised through the
logging filter, but the :class:`Secret` wrapper's own contract and every
provider except the environment one had never been executed.

Two things here are security controls rather than conveniences, and they are
tested as such. **A secret must not leak through ordinary string formatting** —
``print``, an f-string, a ``repr`` in a traceback — because that is precisely
how a key reaches a log. And **a missing secret must raise rather than return
something falsy**, because a provider that answers "" for an absent key turns
a configuration mistake into an authentication failure against a live broker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atrader.config.secrets import (
    ChainSecretProvider,
    EnvSecretProvider,
    FileSecretProvider,
    Secret,
    SecretProvider,
    StaticSecretProvider,
    is_sensitive_key,
    redact,
)
from atrader.core.errors import ConfigError

KEY = "ANTHROPIC_API_KEY"
VALUE = "sk-ant-thisisnotarealkey0000000000"


class TestSecretDoesNotLeak:
    def test_repr_shows_the_name_and_not_the_value(self) -> None:
        secret = Secret(KEY, VALUE)
        assert repr(secret) == f"Secret({KEY})"
        assert VALUE not in repr(secret)

    def test_str_does_not_leak_either(self) -> None:
        """``str`` is what an f-string calls. If it differed from ``repr`` the
        wrapper would protect the debugger and not the log line."""
        assert VALUE not in str(Secret(KEY, VALUE))

    def test_an_f_string_cannot_expose_it_by_accident(self) -> None:
        secret = Secret(KEY, VALUE)
        assert VALUE not in f"using {secret} to authenticate"

    def test_the_value_is_available_only_on_purpose(self) -> None:
        assert Secret(KEY, VALUE).reveal() == VALUE

    def test_the_name_is_readable_without_revealing(self) -> None:
        assert Secret(KEY, VALUE).name == KEY


class TestSecretEquality:
    def test_same_name_and_value_are_equal(self) -> None:
        assert Secret(KEY, VALUE) == Secret(KEY, VALUE)

    def test_a_different_value_is_not_equal(self) -> None:
        assert Secret(KEY, VALUE) != Secret(KEY, "other")

    def test_a_different_name_is_not_equal(self) -> None:
        assert Secret(KEY, VALUE) != Secret("OTHER_KEY", VALUE)

    def test_comparing_against_a_bare_string_is_never_equal(self) -> None:
        """Not merely false — ``NotImplemented``, so Python falls back to
        identity rather than letting ``secret == "sk-..."`` become a way to
        brute-force the value one guess at a time."""
        assert Secret(KEY, VALUE) != VALUE

    def test_equal_secrets_hash_alike(self) -> None:
        assert len({Secret(KEY, VALUE), Secret(KEY, VALUE)}) == 1

    def test_differing_secrets_do_not_collapse_in_a_set(self) -> None:
        assert len({Secret(KEY, VALUE), Secret(KEY, "other")}) == 2


class TestMissingSecretsRaise:
    """A provider that answers ``""`` for an absent key turns a configuration
    mistake into an authentication failure against a live broker."""

    @pytest.mark.parametrize(
        "provider",
        [
            StaticSecretProvider(),
            EnvSecretProvider({}),
            ChainSecretProvider(StaticSecretProvider()),
        ],
    )
    def test_get_raises_for_an_unknown_name(self, provider: SecretProvider) -> None:
        with pytest.raises(ConfigError, match="is not available"):
            provider.get(KEY)

    def test_the_error_names_the_provider_that_could_not_supply_it(self) -> None:
        with pytest.raises(ConfigError, match="StaticSecretProvider"):
            StaticSecretProvider().get(KEY)

    def test_try_get_answers_none_instead_of_raising(self) -> None:
        assert StaticSecretProvider().try_get(KEY) is None


class TestStaticProvider:
    def test_it_returns_what_it_was_given(self) -> None:
        provider = StaticSecretProvider({KEY: VALUE})
        assert provider.get(KEY).reveal() == VALUE

    def test_it_copies_its_input(self) -> None:
        """A caller that later clears the dict it passed must not be emptying
        the provider."""
        source = {KEY: VALUE}
        provider = StaticSecretProvider(source)
        source.clear()
        assert provider.get(KEY).reveal() == VALUE


class TestEnvProvider:
    def test_it_reads_the_mapping_it_was_given(self) -> None:
        assert EnvSecretProvider({KEY: VALUE}).get(KEY).reveal() == VALUE

    def test_it_snapshots_the_environment(self) -> None:
        env = {KEY: VALUE}
        provider = EnvSecretProvider(env)
        env[KEY] = "rotated"
        assert provider.get(KEY).reveal() == VALUE

    def test_it_falls_back_to_the_real_environment(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv(KEY, VALUE)
        assert EnvSecretProvider().get(KEY).reveal() == VALUE


class TestFileProvider:
    """One file per secret — the Docker/Kubernetes convention."""

    def test_it_reads_a_secret_from_its_own_file(self, tmp_path: Path) -> None:
        (tmp_path / KEY).write_text(VALUE, encoding="utf-8")
        assert FileSecretProvider(tmp_path).get(KEY).reveal() == VALUE

    def test_a_trailing_newline_is_stripped(self, tmp_path: Path) -> None:
        """``echo secret > file`` appends one, and it is the single most
        common cause of a credential that looks right and is rejected."""
        (tmp_path / KEY).write_text(f"{VALUE}\n", encoding="utf-8")
        assert FileSecretProvider(tmp_path).get(KEY).reveal() == VALUE

    def test_only_trailing_newlines_are_stripped(self, tmp_path: Path) -> None:
        """Leading or interior whitespace could be part of the credential —
        guessing which parts are decoration would corrupt it."""
        (tmp_path / KEY).write_text("  padded  \n", encoding="utf-8")
        assert FileSecretProvider(tmp_path).get(KEY).reveal() == "  padded  "

    def test_an_absent_file_is_simply_not_configured(self, tmp_path: Path) -> None:
        assert FileSecretProvider(tmp_path).try_get(KEY) is None

    def test_a_directory_where_a_secret_should_be_is_not_a_secret(self, tmp_path: Path) -> None:
        (tmp_path / KEY).mkdir()
        assert FileSecretProvider(tmp_path).try_get(KEY) is None

    def test_an_unreadable_file_is_reported_not_swallowed(
        self,
        tmp_path: Path,
        monkeypatch,  # type: ignore[no-untyped-def]
    ) -> None:
        """Permissions wrong on a mounted secret is a deployment fault. It
        must not look identical to 'not configured', which a caller may treat
        as an acceptable absence.

        The failure is injected rather than produced with ``chmod``: the test
        suite runs as root in CI, and root ignores the permission bits, so a
        chmod-based version of this test would pass by not failing.
        """
        path = tmp_path / KEY
        path.write_text(VALUE, encoding="utf-8")

        def refuse(*_args: object, **_kwargs: object) -> str:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "read_text", refuse)

        with pytest.raises(ConfigError, match="cannot read secret"):
            FileSecretProvider(tmp_path).try_get(KEY)

    def test_the_error_names_the_path_so_the_mount_can_be_found(
        self,
        tmp_path: Path,
        monkeypatch,  # type: ignore[no-untyped-def]
    ) -> None:
        path = tmp_path / KEY
        path.write_text(VALUE, encoding="utf-8")

        def refuse(*_args: object, **_kwargs: object) -> str:
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(Path, "read_text", refuse)

        with pytest.raises(ConfigError, match=str(path)):
            FileSecretProvider(tmp_path).get(KEY)


class TestChainProvider:
    def test_the_first_provider_with_the_key_wins(self) -> None:
        chain = ChainSecretProvider(
            StaticSecretProvider({KEY: "first"}),
            StaticSecretProvider({KEY: "second"}),
        )
        assert chain.get(KEY).reveal() == "first"

    def test_it_falls_through_to_a_later_provider(self) -> None:
        chain = ChainSecretProvider(
            StaticSecretProvider({"OTHER": "x"}),
            StaticSecretProvider({KEY: VALUE}),
        )
        assert chain.get(KEY).reveal() == VALUE

    def test_an_empty_chain_is_refused_at_construction(self) -> None:
        """A chain of nothing would answer ``None`` to every lookup, which
        reads as 'no secrets configured' rather than 'misconfigured'."""
        with pytest.raises(ConfigError, match="at least one provider"):
            ChainSecretProvider()

    def test_a_realistic_env_then_file_chain(self, tmp_path: Path) -> None:
        (tmp_path / KEY).write_text("from-file", encoding="utf-8")
        chain = ChainSecretProvider(EnvSecretProvider({}), FileSecretProvider(tmp_path))
        assert chain.get(KEY).reveal() == "from-file"


class TestRedaction:
    """Spec §9.3's second line of defence, for values that arrive as plain
    strings from somewhere that never wrapped them in a :class:`Secret`."""

    @pytest.mark.parametrize(
        "text",
        [
            "key=sk-ant-api03-abcdefghijklmnop",
            "Authorization: Bearer abcdefghijklmnopqrst",
            "token ghp_abcdefghijklmnopqrstuvwxyz",
        ],
    )
    def test_credential_shaped_values_are_masked(self, text: str) -> None:
        assert "REDACTED" in redact(text)

    def test_ordinary_text_is_left_alone(self) -> None:
        message = "submitted order 42 for AAPL at 187.50"
        assert redact(message) == message

    @pytest.mark.parametrize(
        "key", ["api_key", "API_KEY", "password", "account_number", "client_secret"]
    )
    def test_known_sensitive_field_names_are_recognised(self, key: str) -> None:
        assert is_sensitive_key(key)

    @pytest.mark.parametrize("key", ["broker_api_key", "anthropic_token", "vault_password"])
    def test_a_prefixed_field_name_is_still_recognised(self, key: str) -> None:
        """Masking only exact matches would miss every namespaced config key,
        which is most of them."""
        assert is_sensitive_key(key)

    def test_hyphens_and_case_do_not_evade_it(self) -> None:
        assert is_sensitive_key("API-KEY")

    @pytest.mark.parametrize("key", ["symbol", "quantity", "keyboard_layout"])
    def test_ordinary_field_names_are_not_masked(self, key: str) -> None:
        """Over-masking is not free: a log where half the fields say
        ***REDACTED*** stops being read."""
        assert not is_sensitive_key(key)
