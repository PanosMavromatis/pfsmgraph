# Build definition for pfsmgraph-dl (meson-python).
#
# Pure-Python member, on meson-python for the namespace reason recorded in
# packages/pfsmgraph-dataseq/meson.build. rnn and transformer are plain
# submodules of this one distribution, and meson has no recursive install, so
# each subpackage needs its own install_sources call -- adding a subpackage
# without one is the failure the source test catches.
#
# PEP 420: this file installs sources under pfsmgraph/dl/ only. It must never
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

project('pfsmgraph-dl', meson_version: '>=1.2')

py = import('python').find_installation(pure: true)

py.install_sources(['src/pfsmgraph/dl/__init__.py'], subdir: 'pfsmgraph/dl')
py.install_sources(['src/pfsmgraph/dl/rnn/__init__.py'], subdir: 'pfsmgraph/dl/rnn')
py.install_sources(
  ['src/pfsmgraph/dl/transformer/__init__.py'],
  subdir: 'pfsmgraph/dl/transformer',
)
