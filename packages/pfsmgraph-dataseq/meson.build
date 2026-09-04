# Build definition for pfsmgraph-dataseq (meson-python).
#
# Pure-Python member with no compiled code and none planned. It is on
# meson-python solely so that this distribution gets an editable-install finder
# of its own -- see the namespace note in pyproject.toml. A member left on a
# plain .pth is shadowed by any sibling's meson-python finder and stops
# importing altogether.
#
# PEP 420: this file installs sources under pfsmgraph/dataseq/ only. It must never
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

project('pfsmgraph-dataseq', meson_version: '>=1.2')

py = import('python').find_installation(pure: true)

py.install_sources(
  [
    'src/pfsmgraph/dataseq/__init__.py',
    'src/pfsmgraph/dataseq/_collate.py',
    'src/pfsmgraph/dataseq/_dataset.py',
    'src/pfsmgraph/dataseq/_record.py',
    'src/pfsmgraph/dataseq/_reserved.py',
    'src/pfsmgraph/dataseq/_vocabulary.py',
    'src/pfsmgraph/dataseq/py.typed',
  ],
  subdir: 'pfsmgraph/dataseq',
)
