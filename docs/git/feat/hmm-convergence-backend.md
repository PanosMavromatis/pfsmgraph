# feat/hmm-convergence-backend

**Created**: 2026-09-17
**Base**: main at d0cc0fe
**Status**: active

## Purpose

Make every forward pass the topology search runs take a named backend, and accept
[ADR 0024](../../design/adr/0024-search-compiled-work.md). A profile of one `_suggest_move`
round (`m001_0005_005`, `cython` throughout) spent 10.6 of 12.7 s in the numpy forward pass
that `baum_welch`'s convergence check runs whatever `backend=` says (`_baum_welch.py:302`,
`_finish` at `:320`). ADR 0023 §7's `score_backend=` could not reach it, because `baum_welch`
makes that call itself. The fix is plumbing over kernels that already exist: no kernel, no
backend row, no compiled search.

## Scope

- Decide ADR 0024's Open item: (a) a separate `score_backend=` on `baum_welch`, default
  `"python"`, validated before work; or (b) the phase derived from `backend=`, `"python"`
  under `"torch"`. Mark the ADR Accepted.
- Implement it in `_baum_welch.py`, pass `score_backend` through `_trials.py` and
  `_search.py`, and rewrite `BaumWelchResult`'s "reference forward pass on every backend"
  sentence and the comments at `_trials.py:170` and `_search.py:175`.
- Test `==` of `BaumWelchResult`, trial and search results across the bit-identical phases;
  refusal of an invalid or `torch` check phase; default unchanged.
- Track `.scratch/hmm-lush/measurements/search_round_profile.py`, measure the round after
  the change, and measure whether a `torch` search leaves the serial path.
- Document the choice and the speed guidance in `docs/api/hmm/baum_welch.md` with executed
  output.
- `DEFERRED.md` trigger for parallelizing trials; a note under `HMMLIB-ACCOUNT.md` §12; a
  pointer from ADR 0023 §7; `core.md` sync and `AGENTS.md` rebuild.

## Context

- ADR 0024 (Proposed, 2026-09-17) — the decision and its Open items.
- ADR 0023 §7 — gave the scan the search's backend; this extends it to the check.
- ADR 0021 — named backends, no substitution; weighs against option (b).
- ADR 0020 — the bit-identity of python, cython, cpu_parallel and cuda.
- PR #48 (`feat/hmm-search-loop`) — the search this plumbs.
- Master plan `docs/plan/TODO.md`, revision 04-hmm-v0.3.0, the goal after the search loop,
  whose profile now sits under goal 4 of the branch plan.
- `.scratch/hmm-lush/HMMLIB-ACCOUNT.md` §12 — "leaving the driver interpreted costs little",
  which the profile qualifies.

## Notes
