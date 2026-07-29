"""No source file may be invisible to git.

This exists because it already went wrong once. ``.gitignore`` carried a
``secrets.*`` rule meant to keep credential files out of version control, and
it silently swallowed ``src/atrader/config/secrets.py`` — the
:class:`~atrader.config.secrets.SecretProvider` module. Every local check
passed (the file was on disk), ``git status`` was clean, ``git add .`` did
nothing, the push succeeded, and the repository was broken: a fresh clone
could not import ``atrader.config.secrets`` at all.

What made that bug survive was not that the file was uncommitted — it was
that the file was **ignored**, and therefore invisible. An ordinary uncommitted
file announces itself in ``git status`` on every command; an ignored one never
does, and no amount of test coverage helps because the tests run against the
copy on disk.

So this checks ignored-ness, not tracked-ness. Asserting that every source
file is already committed would fail every time anyone adds a new module
before staging it — a guard that cries wolf on normal work gets deleted, and
then it is not guarding anything. Narrowing it to the silent case gives it no
false positives and keeps it pointed at the failure it exists to prevent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: directory -> glob of files that must reach a fresh clone for the system to run.
REQUIRED: dict[str, str] = {
    "src": "**/*.py",  # the package itself
    "config": "**/*.yaml",  # spec §4.6 — boot fails closed without these
    "tests": "**/*.py",  # a test that never ships is a test that never runs
}


def ignored_paths(candidates: list[Path]) -> list[Path]:
    """Which of *candidates* git would refuse to track, via ``check-ignore``.

    ``--stdin`` in one call rather than one subprocess per file: the package
    has a hundred-odd modules and this test should not cost a hundred process
    spawns. Exit code 1 means "nothing matched", which is the healthy case,
    so it is not an error here.
    """
    if not candidates:
        return []
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(str(p) for p in candidates),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(f"git check-ignore failed: {result.stderr}")
    return [Path(line) for line in result.stdout.splitlines() if line]


@pytest.fixture(scope="module", autouse=True)
def _require_git_checkout() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout — there is no .gitignore to be caught by")


@pytest.mark.parametrize(("directory", "pattern"), sorted(REQUIRED.items()))
def test_no_required_file_is_hidden_from_git_by_gitignore(directory: str, pattern: str) -> None:
    root = REPO_ROOT / directory
    if not root.is_dir():
        pytest.skip(f"{directory}/ does not exist in this checkout")

    on_disk = [
        path.relative_to(REPO_ROOT)
        for path in root.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    ]
    assert on_disk, f"{directory}/{pattern} matched nothing — the guard would prove nothing"

    hidden = sorted(str(path) for path in ignored_paths(on_disk))

    assert not hidden, (
        f"{len(hidden)} file(s) under {directory}/ are excluded by .gitignore, so "
        f"they will never reach a fresh clone and git will never say so:\n  "
        + "\n  ".join(hidden)
        + "\n\nThis happened once with the 'secrets.*' rule swallowing "
        "src/atrader/config/secrets.py. Add a negation for the affected path "
        "(e.g. '!**/secrets.py') rather than dropping the protective rule."
    )
