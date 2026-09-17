# feat/hmm-convergence-backend

**Status**: active
**Created**: 2026-09-17
**Subgoal**: Settle how the topology search uses the compiled backends, and accept ADR 0024 — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [ ] Decide ADR 0024's Open item and accept the record
  - [ ] Choose **(a)** a separate `score_backend=` on `baum_welch`, default `"python"`, validated before any work (the trials' precedent, ADR 0021; recommended by the ADR), or **(b)** the check's phase derived from `backend=`, `"python"` under `"torch"`
  - [ ] Record the answer in ADR 0024 and mark it Accepted in the record and its `adr/README.md` row
- [ ] Implement the check's phase
  - [ ] `baum_welch`'s check (`_baum_welch.py:302`) and `_finish` (`:320`) take the chosen phase
  - [ ] `_re_converge_and_score` (`_trials.py:165-173`) and `_search` (`_search.py:174`) pass `score_backend` through
  - [ ] Rewrite `BaumWelchResult`'s "reference forward pass on every backend" sentence and the comments at `_trials.py:170` and `_search.py:175`; no kernel and no backend row
- [ ] Test it
  - [ ] `BaumWelchResult` fields `==` with the check on each available `forward_backward` phase against `"python"`
  - [ ] A trial's and a search's results `==` likewise
  - [ ] With (a): an invalid or `torch` check phase is refused before any work
  - [ ] The default leaves every existing test and executed output unchanged
- [ ] Measure it
  - [ ] Track `.scratch/hmm-lush/measurements/search_round_profile.py` with the master plan Note's setup, re-run the round, and replace ADR 0024's "about 2 s, not yet measured"
  - [ ] Measure whether a search on `backend="torch"` leaves the serial path on the tracked fixtures, and record it in ADR 0024
- [ ] Document it in `docs/api/hmm/baum_welch.md` with executed output: the new choice if any; python/cython/cpu_parallel/cuda identical in result and differing in speed; `torch` held to a tolerance and able to change a search's path; Cython fastest at `S <= 8`, with the numbers
- [ ] Record the follow-through
  - [ ] `DEFERRED.md` trigger "a topology search too slow to run at the model sizes in use" with ADR 0024 §3's content
  - [ ] A qualifying note under `HMMLIB-ACCOUNT.md` §12, and a pointer from ADR 0023 §7 to ADR 0024 §2
  - [ ] Sync `core.md` (the new script, test counts, `baum_welch`'s signature if it changes, 0024's status) and rebuild `AGENTS.md`
