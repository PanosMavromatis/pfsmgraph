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

# `compiled` (meson.options) has to govern BOTH this and the extension block
# below. Measured 2026-09-13: skipping the extension alone still produced a
# cp312-linux_x86_64 wheel, because pure: false routes every installed file to
# platlib, and a wheel with anything in platlib is platform-tagged.
py = import('python').find_installation(pure: not get_option('compiled'))
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
    # ADR 0021 backend table and selection. Shipped, because the public calls
    # resolve backend= through it.
    'src/pfsmgraph/hmm/_backends.py',
    # Baum-Welch re-estimation for revision 03: the M-step and the EM loop over
    # the forward-backward counts. Not a DP kernel, so no lifecycle phases.
    'src/pfsmgraph/hmm/_baum_welch.py',
    # The torch backend of baum_welch: the E-step as reverse-mode gradients,
    # held to the numpy reference within the ADR 0020 tolerance. Imported only
    # when a probe or backend= names it, so torch stays optional.
    'src/pfsmgraph/hmm/_baum_welch_torch.py',
    # Forward-backward for revision 03, ADR 0002 phase 1, in the scaled
    # probability domain (ADR 0020). No apostrophes in these comments: the
    # source test reads entries as single-quoted strings.
    'src/pfsmgraph/hmm/_forward_backward.py',
    'src/pfsmgraph/hmm/_numeric.py',
    'src/pfsmgraph/hmm/_params.py',
    'src/pfsmgraph/hmm/_viterbi.py',
    # ADR 0002 phase 3. Pure Python, but it still has to be listed: meson does
    # not glob, and an unlisted module is invisible to the editable finder and
    # absent from any built wheel. Both were observed here -- writing this file
    # without listing it gave a ModuleNotFoundError at import, which is the
    # characteristic ADR 0018 failure and the reason it is loud rather than
    # silent.
    'src/pfsmgraph/hmm/_viterbi_cpu_parallel.py',
    # ADR 0002 phase 4, the CUDA backend. Pure Python for the same reason as
    # phase 3 -- numba compiles it at runtime, meson compiles nothing -- and
    # listed for the same reason: unlisted, it is invisible to the editable
    # finder and absent from the wheel.
    'src/pfsmgraph/hmm/_viterbi_cuda.py',
    # PEP 561 marker. Not a .py file, and meson does not glob, so a marker at
    # the right path but missing here reaches no wheel.
    'src/pfsmgraph/hmm/py.typed',
  ],
  subdir: 'pfsmgraph/hmm',
)

# --- Compiled kernels --------------------------------------------------------
# ADR 0002 phase 2 of the Viterbi decode. Named _viterbi_cython rather than
# _viterbi because _viterbi.py is the phase-1 reference and the two must coexist:
# ADR 0003 runs one suite against every available backend, so a compiled kernel
# shadowing the reference would make equivalence untestable. Revision 03's
# forward-backward takes _forward_backward_cython on the same pattern.
#
# Dormant until the .pyx exists, then activates automatically. Cython is pulled
# in with add_languages() only when needed, so project() above stays C-only.
viterbi_pyx = 'src/pfsmgraph/hmm/_viterbi_cython.pyx'
# ADR 0002 phase 2 of forward-backward and the batched E-step (revision 03).
forward_backward_pyx = 'src/pfsmgraph/hmm/_forward_backward_cython.pyx'

# Shared by every extension below, so it is set up once when any .pyx exists.
if get_option('compiled') and (fs.exists(viterbi_pyx) or fs.exists(forward_backward_pyx))
  add_languages('cython', native: false)

  numpy_inc = run_command(
    py, ['-c', 'import numpy; print(numpy.get_include())'],
    check: true,
  ).stdout().strip()
  numpy_dep = declare_dependency(include_directories: include_directories(numpy_inc))
endif

if get_option('compiled') and fs.exists(viterbi_pyx)
  py.extension_module(
    '_viterbi_cython',
    viterbi_pyx,
    dependencies: [numpy_dep],
    cython_args: ['-3'],
    subdir: 'pfsmgraph/hmm',
    install: true,
  )
endif

# Forward-backward sums in a fixed order and multiplies inside those sums, so a
# contracted a*b + c (one fused FMA instruction) rounds differently from the
# numpy reference: ADR 0020 section 4 forbids it. Baseline x86-64 has no FMA to
# contract into, but a -march=native build, aarch64 GCC and Apple clang do, so
# contraction is switched off here rather than left to the target. Supported
# arguments only, so a compiler without the flag is not broken by it.
if get_option('compiled') and fs.exists(forward_backward_pyx)
  cc = meson.get_compiler('c')

  py.extension_module(
    '_forward_backward_cython',
    forward_backward_pyx,
    dependencies: [numpy_dep],
    cython_args: ['-3'],
    c_args: cc.get_supported_arguments('-ffp-contract=off'),
    subdir: 'pfsmgraph/hmm',
    install: true,
  )
endif
