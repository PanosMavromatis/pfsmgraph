# Build definition for pfsmgraph-hmm (meson-python).
#
# Compiled member, same rationale as pfsmgraph-align (PRD §6.1): meson-python
# for rebuild-on-import under editable installs; `ninja` must be on PATH.
#
# Only the Baum-Welch core is cythonized. Topology search via state
# merge/split is expected to stay plain Python (PRD §6). The exact module
# layout for the compiled core is left to the Lush translation, which should
# not be pre-hardened here (PRD §1.5) -- the path below is provisional and the
# block is dormant until the .pyx actually exists.
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
# meson does not glob: append each new .py module to this list as it lands
# (Baum-Welch Python reference, topology search, state merge/split, ...).
py.install_sources(
  [
    'src/pfsmgraph/hmm/__init__.py',
  ],
  subdir: 'pfsmgraph/hmm',
)

# --- Compiled Baum-Welch core ----------------------------------------------
# Provisional target: pfsmgraph.hmm._baum_welch, compiled from
# src/pfsmgraph/hmm/_baum_welch.pyx, with a plain-Python wrapper alongside.
# Confirm/relocate during the Lush translation; adjust the two paths below.
bw_pyx = 'src/pfsmgraph/hmm/_baum_welch.pyx'

if fs.exists(bw_pyx)
  add_languages('cython', native: false)

  numpy_inc = run_command(
    py, ['-c', 'import numpy; print(numpy.get_include())'],
    check: true,
  ).stdout().strip()
  numpy_dep = declare_dependency(include_directories: include_directories(numpy_inc))

  py.extension_module(
    '_baum_welch',
    bw_pyx,
    dependencies: [numpy_dep],
    cython_args: ['-3'],
    subdir: 'pfsmgraph/hmm',
    install: true,
  )
endif
