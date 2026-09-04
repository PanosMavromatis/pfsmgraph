# Build definition for pfsmgraph-hmm (meson-python).
#
# Compiled member, same rationale as pfsmgraph-align (PRD §6.1): meson-python
# for rebuild-on-import under editable installs; `ninja` must be on PATH.
#
# The DP kernels follow ADR 0002's four-phase lifecycle, so the first .pyx here
# is the Viterbi decode (revision 02), not the Baum-Welch core (revision 03).
# Each phase is a separate module living beside the phase-1 reference, never a
# replacement for it -- ADR 0003 selects between them by module path. Topology
# search via state merge/split is expected to stay plain Python (PRD §6).
#
# PEP 420: installs sources under pfsmgraph/hmm/ only, never
# pfsmgraph/__init__.py. Version comes from pyproject.toml's static
# [project].version.

project(
  'pfsmgraph-hmm',
  'c',
  meson_version: '>=1.2',
  default_options: ['warning_level=2', 'buildtype=release'],
)

py = import('python').find_installation(pure: false)
fs = import('fs')

# --- Pure-Python sources -----------------------------------------------------
# meson does not glob: append each new .py module to this list as it lands, and
# any package data too -- py.typed arrives at the release commit and is not a
# .py file. tests/test_meson_sources.py asserts this list matches what is on
# disk; it drifted once already, silently, because the dormant block below meant
# nothing ever built these files.
py.install_sources(
  [
    'src/pfsmgraph/hmm/__init__.py',
    'src/pfsmgraph/hmm/_numeric.py',
    'src/pfsmgraph/hmm/_params.py',
    'src/pfsmgraph/hmm/_viterbi.py',
  ],
  subdir: 'pfsmgraph/hmm',
)

# --- Compiled kernels --------------------------------------------------------
# ADR 0002 phase 2 of the Viterbi decode. Named _viterbi_cython rather than
# _viterbi because _viterbi.py is the phase-1 reference and the two must coexist:
# ADR 0003 runs one suite against every available backend, so a compiled kernel
# shadowing the reference would make equivalence untestable. Revision 03's
# Baum-Welch core takes _baum_welch_cython on the same pattern.
#
# Dormant until the .pyx exists, then activates automatically. Cython is pulled
# in with add_languages() only when needed, so project() above stays C-only.
viterbi_pyx = 'src/pfsmgraph/hmm/_viterbi_cython.pyx'

if fs.exists(viterbi_pyx)
  add_languages('cython', native: false)

  numpy_inc = run_command(
    py, ['-c', 'import numpy; print(numpy.get_include())'],
    check: true,
  ).stdout().strip()
  numpy_dep = declare_dependency(include_directories: include_directories(numpy_inc))

  py.extension_module(
    '_viterbi_cython',
    viterbi_pyx,
    dependencies: [numpy_dep],
    cython_args: ['-3'],
    subdir: 'pfsmgraph/hmm',
    install: true,
  )
endif
