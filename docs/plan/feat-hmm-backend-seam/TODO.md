# feat/hmm-backend-seam

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Write the runtime backend-selection ADR and build the seam it decides — `docs/plan/TODO.md`, revision 03-hmm-v0.2.0

## Goals

- [ ] Write ADR 0021 on runtime backend selection
  - [ ] Settle the public shape: a per-call `backend=` keyword, a module-level default, or both; and the backend names
  - [ ] Settle the default backend and what an unavailable requested backend does (raise vs fall back), against ADR 0011's strictness position
  - [ ] Settle where the registry lives, so the runtime and the repo-root test header read one source
- [ ] Build the seam in `pfsmgraph-hmm`
  - [ ] `viterbi` dispatches over all four phases through it
  - [ ] Forward-backward and EM accept the same parameter shape with only phase 1 registered
- [ ] Parameterise the ADR 0003 suites over the seam
  - [ ] Shared Viterbi tests run once per available backend through the public API
  - [ ] Fold `test_viterbi.py`'s labelled non-shared section into them
- [ ] Restore the accelerator extras (`cpu-parallel`, `gpu` with the `numpy<2.5` cap)
- [ ] Update the docs: `docs/api/hmm/`, `core.md`, the ADR index, and the `DEFERRED.md` entries
