# Build definition for pfsmgraph-hseg (meson-python).
#
# Pure orchestration over align today, on meson-python for the namespace
# reason recorded in packages/pfsmgraph-dataseq/meson.build rather than for any
# build one. If hseg turns out to have its own DP recurrence (PRD §6, §8), an
# extension block joins this file; the backend does not change.
#
# PEP 420: this file installs sources under pfsmgraph/hseg/ only. It must never
# create or install pfsmgraph/__init__.py -- the pfsmgraph level is an implicit
# namespace shared with the other four distributions.
#
# tests/test_meson_sources.py checks this list against the package on disk. That
# test is load-bearing rather than incidental: a module missing here is a
# ModuleNotFoundError at import, and that loudness is why this arrangement was
# chosen over installing the compiled members non-editable, whose characteristic
# failures (a missing `editable = false`, a stale source served from cache) are
# both silent.
#
# The project version is not set here: pyproject.toml carries a static
# [project].version, which meson-python treats as authoritative.

project('pfsmgraph-hseg', meson_version: '>=1.2')

py = import('python').find_installation(pure: true)

py.install_sources(
  ['src/pfsmgraph/hseg/__init__.py'],
  subdir: 'pfsmgraph/hseg',
)
