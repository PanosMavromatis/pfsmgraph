# Build definition for pfsmgraph-align (meson-python).
#
# Compiled member of the family: the DP kernels follow the pure-Python -> Cython
# -> Numba CUDA anti-diagonal wavefront lifecycle (CLAUDE.md "Invariants").
# meson-python is chosen for its rebuild-on-import behaviour under editable
# installs; `ninja` must be on PATH in the dev environment for that hook to fire
# (PRD §6.1). ADR 0012 removed it from the workspace-root [dependency-groups].dev
# along with meson-python and cython; it returns when that ADR is reverted. Note
# that PATH alone is not sufficient -- measured on exp/meson-python-namespace, the
# generated editable loader bakes an absolute ninja path at build time and never
# consults PATH.
#
# PEP 420: this file installs sources under pfsmgraph/align/ only. It must never
# create or install pfsmgraph/__init__.py -- the pfsmgraph level is an implicit
# namespace shared with the other four distributions.
#
# The project version is not set here: pyproject.toml carries a static
# [project].version, which meson-python treats as authoritative.

project(
  'pfsmgraph-align',
  'c',
  meson_version: '>=1.2',
  default_options: ['warning_level=2', 'buildtype=release'],
)

py = import('python').find_installation(pure: false)
fs = import('fs')

# --- Pure-Python sources -----------------------------------------------------
# meson does not glob: append each new .py module to this list as it lands.
py.install_sources(
  [
    'src/pfsmgraph/align/__init__.py',
  ],
  subdir: 'pfsmgraph/align',
)

# --- Compiled kernels ------------------------------------------------------
# The Needleman-Wunsch Cython kernel is brought in from the proof-of-concept
# during the file move (PRD §11). Target extension module:
#     pfsmgraph.align.algorithms.needleman_wunsch._cython
# While moving _cython.pyx in, also apply the deferred 2-D memoryview indexing
# fix (A[i][j] -> A[i, j]) from PRD §11.1 -- fold it into the move, not a
# follow-up commit.
#
# This block is dormant until the .pyx exists, then activates automatically.
# Cython is pulled in with add_languages() only when actually needed, so the
# project() call above stays C-only in the meantime.
nw_dir = 'src/pfsmgraph/align/algorithms/needleman_wunsch'
nw_pyx = nw_dir / '_cython.pyx'

if fs.exists(nw_pyx)
  add_languages('cython', native: false)

  # Established scientific-Python idiom for locating NumPy headers. meson may
  # warn about the absolute path; NumPy and SciPy build the same way.
  numpy_inc = run_command(
    py, ['-c', 'import numpy; print(numpy.get_include())'],
    check: true,
  ).stdout().strip()
  numpy_dep = declare_dependency(include_directories: include_directories(numpy_inc))

  py.install_sources(
    ['src/pfsmgraph/align/algorithms/__init__.py'],
    subdir: 'pfsmgraph/align/algorithms',
  )
  py.install_sources(
    [nw_dir / '__init__.py'],
    subdir: 'pfsmgraph/align/algorithms/needleman_wunsch',
  )
  py.extension_module(
    '_cython',
    nw_pyx,
    dependencies: [numpy_dep],
    cython_args: ['-3'],
    subdir: 'pfsmgraph/align/algorithms/needleman_wunsch',
    install: true,
  )
endif
