"""Guard the hand-maintained source lists in the meson-python build files.

`meson.build` deliberately does not glob -- meson's position is that an explicit
list is what makes a build reproducible -- so every module is appended by hand as
it lands. That list drifted: `pfsmgraph-hmm` gained `_numeric.py`, `_params.py`
and `_viterbi.py` while `install_sources` still named only `__init__.py`, and
because the extension blocks are dormant, nothing ever built these files to
notice. ADR 0012 predicted the class of failure -- "the meson.build files are
unexercised ... they may have drifted" -- without leaving a mechanism to catch it.

The failure is silent in the worst way: the wheel builds and installs, then fails
at import. Under a meson-python editable install it is worse, because the
generated import hook serves exactly what meson installs, so an unlisted module
goes missing from a development checkout with the file sitting on disk.

`py.typed` is the case this test exists for more than the modules. It is not a
`.py` file, so a guard phrased as "every module is listed" would miss it, and
`core.md` records that a `py.typed` which fails to reach the wheel makes a type
checker discard every annotation in the package, with no error and no warning.
`align` and `hmm` have none yet; they gain one at their release commit, and this
test fails the moment the file appears until `meson.build` names it. That is
deliberately preferred to a dormant `if fs.exists()` install block: adding
unexercised build config to guard against unexercised build config rotting is not
a repair.

Scope is the top-level `install_sources` calls only. Calls nested inside
`if fs.exists(...)` describe subpackages that do not exist yet (align's
`algorithms/needleman_wunsch/`), so there is nothing on disk to compare them
against; they come into scope with the `.pyx` that guards them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Wheel content that is not a module. Extend as the family grows.
PACKAGE_DATA_NAMES = frozenset({"py.typed"})

_QUOTED = re.compile(r"'([^']*)'")
_SUBDIR = re.compile(r"subdir\s*:\s*'[^']*'")


def _meson_builds() -> list[Path]:
    return sorted(REPO_ROOT.glob("packages/*/meson.build"))


def _top_level_install_sources(text: str) -> list[str]:
    """Quoted paths from `py.install_sources(...)` calls outside any if-block."""
    calls: list[str] = []
    depth = 0
    buf: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if buf is not None:
            buf.append(line)
            if stripped.endswith(")"):
                calls.append("\n".join(buf))
                buf = None
        elif stripped.startswith("if "):
            depth += 1
        elif stripped == "endif":
            depth -= 1
        elif depth == 0 and stripped.startswith("py.install_sources("):
            buf = [line]
    return [p for call in calls for p in _QUOTED.findall(_SUBDIR.sub("", call))]


def _package_dir(meson_build: Path) -> Path:
    """`packages/pfsmgraph-hmm/meson.build` -> its `src/pfsmgraph/hmm/`."""
    member = meson_build.parent.name.split("-", 1)[1]
    return meson_build.parent / "src" / "pfsmgraph" / member


def _on_disk(package_dir: Path) -> set[str]:
    return {
        entry.name
        for entry in package_dir.iterdir()
        if entry.is_file()
        and (entry.suffix == ".py" or entry.name in PACKAGE_DATA_NAMES)
    }


IDS = [p.parent.name for p in _meson_builds()]


def test_there_are_meson_builds_to_check():
    """A parser that silently matches nothing would pass every test below."""
    assert _meson_builds(), "no packages/*/meson.build found"


@pytest.mark.parametrize("meson_build", _meson_builds(), ids=IDS)
def test_install_sources_is_not_empty(meson_build: Path):
    assert _top_level_install_sources(meson_build.read_text())


@pytest.mark.parametrize("meson_build", _meson_builds(), ids=IDS)
def test_install_sources_matches_the_package_on_disk(meson_build: Path):
    package_dir = _package_dir(meson_build)
    listed = {Path(p).name for p in _top_level_install_sources(meson_build.read_text())}
    assert listed == _on_disk(package_dir), (
        f"{meson_build.relative_to(REPO_ROOT)} is out of step with "
        f"{package_dir.relative_to(REPO_ROOT)}/"
    )


@pytest.mark.parametrize("meson_build", _meson_builds(), ids=IDS)
def test_no_meson_build_installs_a_namespace_init(meson_build: Path):
    """PEP 420: no distribution may own `pfsmgraph/__init__.py`."""
    text = meson_build.read_text()
    for path in _top_level_install_sources(text):
        assert path != "src/pfsmgraph/__init__.py"
    assert "subdir: 'pfsmgraph'," not in text
