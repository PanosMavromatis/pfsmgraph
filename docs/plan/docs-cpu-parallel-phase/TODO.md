# docs/cpu-parallel-phase

**Status**: active
**Created**: 2026-09-03
**Subgoal**: standalone — a lifecycle amendment ahead of revision 02, not itself a master-plan item

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [ ] Draft ADR 0016, amending ADR 0002
  - [ ] Insert phase 3 — Numba CPU-parallel (`prange`, anti-diagonal) — between Cython and CUDA; renumber CUDA to phase 4
  - [ ] Record the MPS/torch alternative and why it's rejected (ADR 0004 packaging tension; marginal measured speed gain; phases 1-3 already cover algorithmic validation with no GPU)
  - [ ] Settle the dependency question: plain `numba` (CPU target) becomes a hard runtime dependency of `align`/`hmm`, distinct from the optional `numba-cuda` `[gpu]` extra ADR 0004 already governs
  - [ ] Add the README index footnote on ADR 0002's row, mirroring the 0008/0012 precedent

- [ ] Ripple the renumbering into the standing docs
  - [ ] `core.md`'s "Three-phase algorithm lifecycle" invariant → four phases
  - [ ] `codex.md`'s phase-ordered review-priority list → add `_cpu_parallel*.py` between the Cython and CUDA entries
  - [ ] `docs/plan/TODO.md` revision 02's Viterbi subgoals: move "settle the anti-diagonal question" to the new phase 3, renumber the CUDA subgoal to phase 4
  - [ ] `docs/plan/planned/03-hmm-v0.2.0.md`: "ADR 0002 phases 2 and 3" → "phases 2 through 4"

- [ ] Verify and close
  - [ ] Resolve every markdown link touched by the above edits
  - [ ] `uv run pytest -q` (docs-only branch — no test-count change expected, run anyway per convention)
