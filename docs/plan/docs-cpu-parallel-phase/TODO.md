# docs/cpu-parallel-phase

**Status**: merged — PR #14 — 2026-09-03
**Created**: 2026-09-03
**Subgoal**: standalone — a lifecycle amendment ahead of revision 02, not itself a master-plan item

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [x] Draft ADR 0016, amending ADR 0002
  - [x] Insert phase 3 — Numba CPU-parallel (`prange`, anti-diagonal) — between Cython and CUDA; renumber CUDA to phase 4
  - [x] Record the MPS/torch alternative and why it's rejected (ADR 0004 packaging tension; marginal measured speed gain; phases 1-3 already cover algorithmic validation with no GPU)
  - [x] Settle the dependency question: plain `numba` (CPU target) becomes a hard runtime dependency of `align`/`hmm`, distinct from the optional `numba-cuda` `[gpu]` extra ADR 0004 already governs
  - [x] Add the README index footnote on ADR 0002's row, mirroring the 0008/0012 precedent
  > **Done:** `docs/design/adr/0016-numba-cpu-parallel-phase.md` written (Context, Decision,
  > Consequences, Alternatives, Evidence, Open — full template). ADR 0002's Status line
  > edited in place to point forward, mirroring ADR 0008's own pointer to ADR 0012 exactly
  > (in-file Status-line edit, not a body rewrite). README index: 0002's row gets a `§`
  > footnote, new row for 0016, a Reading-order bullet, and the Coverage paragraph's
  > postdates-the-PRD list extended. Two questions left open in the ADR itself rather than
  > forced here: whether anti-diagonal applies to `hmm`'s Viterbi at all (revision 02's
  > existing open finding), and the `_cpu_parallel*.py` naming convention, unexercised
  > against a real kernel.

- [x] Ripple the renumbering into the standing docs
  - [x] `core.md`'s "Three-phase algorithm lifecycle" invariant → four phases
  - [x] `codex.md`'s phase-ordered review-priority list → add `_cpu_parallel*.py` between the Cython and CUDA entries
  - [x] `docs/plan/TODO.md` revision 02's Viterbi subgoals: move "settle the anti-diagonal question" to the new phase 3, renumber the CUDA subgoal to phase 4
  - [x] `docs/plan/planned/03-hmm-v0.2.0.md`: "ADR 0002 phases 2 and 3" → "phases 2 through 4"
  > **Done:** `core.md`'s invariant bullet now names four phases and cites ADR 0016;
  > `codex.md` gained a `_cpu_parallel*.py` review-priority bullet between `_cython*.pyx`
  > and `_cuda*.py`, with the risk shape spelled out (silent race across `prange`
  > iterations vs. an indexing exception). `AGENTS.md`/`AGENTS.override.md` regenerated
  > via `/agents-docs-build` since both sources changed. Revision 02's Viterbi subgoals in
  > `docs/plan/TODO.md`: the "settle the anti-diagonal question" sub-item moved to the new
  > phase 3 (kept its own reasoning about why phases 1-2 can't need it — that reasoning
  > didn't depend on which technology phase 3 turned out to be), and a new phase-4 bullet
  > added for the CUDA work that remains, explicitly scoped to hardware-kernel concerns
  > now that the decomposition is proven in phase 3. Revision 03's forward-recurrence
  > subgoal widened from "phases 2 and 3" to "phases 2 through 4".

- [x] Verify and close
  - [x] Resolve every markdown link touched by the above edits
  - [x] `uv run pytest -q` (docs-only branch — no test-count change expected, run anyway per convention)
  > **Done:** link-checked across all eight touched files (both ADRs, the ADR README, both
  > agent docs, both plan docs, this file) — all resolve. `uv run pytest -q`: 94 passed,
  > unchanged, as expected for a docs-only branch.
