# feat/hmm-viterbi-cython

**Status**: active
**Created**: 2026-09-09
**Subgoal**: Implement Viterbi at ADR 0002 phase 2 (Cython), the first `.pyx` in
a distribution — `docs/plan/TODO.md`, revision 02-hmm-v0.1.0

## Goals

- [ ] Settle what this branch is allowed to claim about equivalence
  - [ ] Reconcile the subgoal's "enforced by the parameterized suite" with
        `core.md`'s finding that parameterization cannot be in force until
        `align`; whichever is wrong is corrected on `main`, not worked around
  - [ ] Triage the three `first .pyx` DEFERRED items: close the meson-python
        revert (spent by ADR 0018), schedule the tokalign indexing fix here,
        leave the `numba-cuda` pin to phase 4
- [ ] Write `_viterbi_cython.pyx` against the phase-0 specification
  - [ ] Apply the comma-form indexing fix to the tokalign template first
  - [ ] Kernel stays numeric and raises nothing, as phase 1 established
- [ ] Make it build, install and import
  - [ ] `hmm/meson.build`: extension block plus every new source in
        `install_sources` — meson does not glob
  - [ ] `_backends.py` registry row; a failed import escalates, never skips
- [ ] Establish equivalence
  - [ ] Against phase 1 on constructed inputs, including the exact-tie case the
        learned fixtures cannot exhibit
  - [ ] Against the three `.vpath.xls` oracles, one of which differs at position
        0 by design
- [ ] Update whatever the change makes stale
