"""Every file the system needs to boot must actually be in the repository.

This exists because it already went wrong once. ``.gitignore`` carried a
``secrets.*`` rule meant to keep credential files out of version control, and
it silently swallowed ``src/atrader/config/secrets.py`` — the
:class:`~atrader.config.secrets.SecretProvider` module. Every local check
passed (the file was on disk), the push succeeded, and the repository was
broken: a fresh clone could not import ``atrader.config.secrets`` at all.

A working tree is not the deliverable — the commit is. So this test compares
what is on disk against what ``git ls-files`` actually tracks, and fails on
anything the repository would be missing after a clean clone. It is cheap
insurance against a whole class of "works on my machine" that no amount of
test coverage can catch, because the tests themselves run against the
untracked file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: directory -> glob of files that must be committed for the system to run.
REQUIRED: dict[str, str] = {
    "src": "**/*.py",  # the package itself
    "config": "**/*.yaml",  # spec §4.6 — boot fails closed without these
}


def tracked_files() -> frozenset[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return frozenset(Path(line) for line in result.stdout.split("\0") if line)


@pytest.fixture(scope="module")
def tracked() -> frozenset[Path]:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout — nothing to compare against")
    return tracked_files()


@pytest.mark.parametrize(("directory", "pattern"), sorted(REQUIRED.items()))
def test_no_required_file_is_missing_from_git(
    directory: str, pattern: str, tracked: frozenset[Path]
) -> None:
    root = REPO_ROOT / directory
    if not root.is_dir():
        pytest.skip(f"{directory}/ does not exist in this checkout")

    on_disk = {
        path.relative_to(REPO_ROOT)
        for path in root.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    missing = sorted(str(path) for path in on_disk - tracked)

    assert not missing, (
        f"{len(missing)} file(s) under {directory}/ exist on disk but are not "
        f"tracked by git, so a fresh clone would not have them:\n  "
        + "\n  ".join(missing)
        + "\n\nCheck .gitignore — a broad rule (this happened with 'secrets.*') "
        "will exclude source files without any visible error. Add a negation "
        "for the affected path rather than dropping the protective rule."
    )
