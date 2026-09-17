# 0024. The topology search stays interpreted, and every forward pass it runs takes a named backend

- **Status:** Accepted — Proposed 2026-09-17 and accepted the same day, when **Resolved**
  chose a separate `score_backend=` on `baum_welch`.
- **Date:** 2026-09-17
- **Source:** none in the PRD — postdates it. Raised on `main` after PR #48
  (`feat/hmm-search-loop`) merged, in revision `04-hmm-v0.3.0`, as three questions: whether
  the search layer needs compiling, how Cython relates to the parallel backends when a full
  search runs, and whether parallelizing the search should be recorded for later.

## Context

Revision 04's topology search is three layers. `_search` runs rounds; `_suggest_move` lists
every merge pair and split trial of the incumbent and ranks them; `_try_split` and
`_try_merge` build one candidate with `_split_state` or `_merge_states`, re-converge it with
`baum_welch`, choose its `d` with `_scan_d` and score it. All of that is Python and numpy.
Below it sit the ADR 0002 kernels, of which revision 03 left four phases for
`forward_backward` (python, cython, cpu_parallel, cuda) and five for `baum_welch`'s E-step
(the same and torch).

**It is tempting to read those as layers**, with the interpreted search on top, Cython under
it and the parallel phases under that, and to conclude that the next speed-up is compiling the
top. Neither half holds. The phases are alternatives, one chosen per call by name, and the
search's time is not spent in its own code.

The search reaches a kernel at two points, both named by string and resolved per call through
`_backends._TABLE` (ADR 0021):

| Parameter of `_search` and every trial | Consumed by | Resolved as | Phases |
|---|---|---|---|
| `backend=` | every E-step inside `baum_welch` (`_baum_welch.py:240`) | `_resolve("baum_welch", backend)` | python, cython, cpu_parallel, cuda, torch |
| `score_backend=` | every forward pass inside `_scan_d` (`_mdl.py:164`) | `_resolve("forward_backward", backend)` | python, cython, cpu_parallel, cuda |

No phase calls another, and none falls back. python, cython, cpu_parallel and cuda are
bit-identical on one host (ADR 0020), and `baum_welch` adds counts in record order at every
`batch_size`, so any combination of those four returns the same `SearchResult` bit for bit.

**There is a third forward pass, and it takes no backend.** `baum_welch` checks convergence at
the end of every batch of cycles by calling `_corpus_description_length` with no `backend=`
(`_baum_welch.py:302`, and `_finish` at `:320`), so the numpy reference runs whatever
`backend=` says. `BaumWelchResult`'s docstring states it as a contract: "the check at the end
of each batch, and so the last entry, is the reference forward pass on every backend". That
sentence was added in `58942fe` (2026-09-14), when `torch` became the second backend, as a
description of a check that predates backends (`1fa7b57`); no reason for it is recorded. The
trials read that last entry as the candidate's unrounded data length (`_trials.py:170`), as
does the search for its start (`_search.py:175`). ADR 0023 §7 gave the *scan* the search's
backend and could not reach this call, because `baum_welch` makes it itself.

## Decision

### 1. The search is not compiled

`_search`, `_suggest_move`, the trials, the surgery, `_m_step`, quantization, the model half
and `_scan_d` stay Python and numpy. No Cython, numba or CUDA version of any of them is
written. Compiled work is the ADR 0002 kernels and nothing else, so this revision adds no
backend row and the `dp-compile` gate does not arm on it.

A compiled search loop would still call numpy, frozen dataclasses and `SeedSequence`, since
that is what it does; what compiling removes is interpreter overhead, and the profile below
puts everything above the kernels at about 1 s of a 12.7 s round. It would also be a second
implementation that has to be held equal to the first.

### 2. Every forward pass the search runs takes a named backend

The convergence check inside `baum_welch` takes a `forward_backward` phase by name, as the
scan already does, and the trials and the search pass theirs through. **It is named by a
separate `score_backend=` on `baum_welch`**, resolved against `forward_backward`'s table
before any work, as `backend=` is against `baum_welch`'s. The trials and `_search` already
take a `score_backend` for the scan and forward it, so one name governs every forward pass a
search runs. It holds these constraints:

- **No result changes on python, cython, cpu_parallel or cuda.** The check's quantity is
  bit-identical across those phases, so every `unchanged` count, every stop and every
  `description_lengths` entry is unchanged. A test asserts `BaumWelchResult` equality, and
  `SearchResult` equality, against the check on `"python"`.
- **The default stays `"python"`**, so every existing call, test and executed output in
  `docs/api/` is unchanged unless a caller asks.
- **`torch` gets no forward phase**, and nothing substitutes one for it (ADR 0021).
  `backend="torch"` pairs with any phase `forward_backward` has, and `score_backend="torch"`
  is refused with the `ValueError` `_resolve` already raises for a phase an algorithm has not
  reached.
- **The contract sentence in `BaumWelchResult` is rewritten**, together with the comments at
  `_trials.py:170` and `_search.py:175` that rely on it, and `docs/api/hmm/baum_welch.md`
  documents the choice with executed output.

### 3. Parallelizing the search is deferred, along trials, not refused

A search is not parallelized in this revision. When it is, the axis is **trials within a
round**, not a kernel's `(record, state)` cells, and the record for it is a `DEFERRED.md`
trigger, "a topology search too slow to run at the model sizes in use", carrying:

- **Why trials.** They are independent — every trial takes the same incumbent and returns its
  own `TrialResult` — and they are the whole of a round's cost.
- **Why it can be exact.** Seeds are keyed by trial identity, `(1, r, s, t)` for split trial
  `(s, t)` of round `r` (ADR 0023 §2), so no trial's draws depend on which ran first; and the
  ranking is a stable sort whose enumeration order is the tie-break. Results gathered back
  **in enumeration order** give a `SearchResult` bit-identical to the serial one.
- **The hazards.** Processes each running `cpu_parallel` oversubscribe threads, and processes
  sharing one GPU queue on it. At small `S` the natural shape is processes over `cython`
  kernels, each kernel serial.
- **The test it owes.** A parallel `SearchResult` `==` the serial one, on a seed that exercises
  ties.

### 4. Choosing a backend is documented as a speed choice, with one exception

When the search becomes public, and in `docs/api/hmm/baum_welch.md` now, the backends are
documented as interchangeable in result and differing in speed, with two facts a caller needs:

- **`backend="torch"` can change the search's path.** Its E-step is held only to ADR 0020's
  tolerance, and ADR 0023's Open item measured that a candidate's total steps by 58–71 bits
  when EM stops slightly elsewhere and the best `d` moves. No other phase can.
- **At the model sizes measured, Cython is the fastest phase.** The parallel phases split one
  timestep across its cells, and with `S <= 8` a parallel region or device launch per timestep
  costs more than the work inside it.

## Consequences

### Positive

- **The fix is plumbing over kernels that already exist**, as ADR 0023 §7 was, and removes
  about 84% of a measured round.
- **One implementation of the search.** Reviewing it against the Lush loop stays a reading of
  one file.
- **Parallelism, when it comes, needs no new decision about correctness.** §3's conditions
  already make it exact, because ADR 0023 keyed seeds by role.
- **The layering misreading is written down**, so the next proposal to compile the driver
  meets the profile first.

### Negative / costs

- **`baum_welch`'s public signature grows by one keyword**, and a published parameter
  cannot be withdrawn without a breaking release.
- **A search is still serial.** A round costs roughly its trial count times a few tenths of a
  second after §2, and the merge count grows with `S²`.
- **The documented reference-check contract changes**, for callers who asked for a phase other
  than `"python"`: bit-identical on one host, but no longer the numpy code path.

## Alternatives considered

- **A Cython port of the search layer.** Rejected: §1, and the profile. It would compile calls
  into numpy rather than arithmetic.
- **Parallelizing the kernels harder for the search.** Rejected for these sizes: at `S <= 8`
  `cpu_parallel` took 18–24 ms and `cuda` about 500 ms per forward pass over the corpus
  (`total_forward_cost.py`), where Cython is 120–330 times faster than numpy.
- **Parallelizing trials now.** Deferred: nothing yet runs a search at a size where a round's
  wall time blocks work, and it brings process management and oversubscription into a
  revision whose remaining goals are the training log, the docs audit and the release.
- **Leaving the check on numpy.** Rejected on the measurement: it is the largest cost in a
  round, and it is the reference only by a sentence that described what the code did.
- **Deriving the check's phase from `backend=`.** Rejected 2026-09-17. It needs no signature
  change, but it maps a request for one algorithm's backend onto another algorithm's phase,
  and onto `"python"` under `"torch"`: the substitution ADR 0021 forbids, done silently. It
  would also make two combinations impossible to ask for: Cython EM with a `cpu_parallel` or
  `cuda` check, and torch EM with a compiled check.
- **Computing the check on the E-step's own bits.** Rejected: `torch`'s per-cycle entries are
  its own, within tolerance, so the stopping rule would then depend on the backend.

## Evidence

- **Where a round's time goes.** One `_suggest_move` round, 2026-09-17, 4-vCPU Xeon,
  `m001_0005_005` with the `set02a_200` corpus as one record (N = 1268),
  `backend="cython"`, `score_backend="cython"`, `split_min_cycles=200`, two split trials per
  state, seed `SeedSequence(1, spawn_key=(1, 0))`, under `cProfile`: 20 moves (10 merges, 10
  splits) in **12.7 s**.

  | Where | Time |
  |---|---|
  | numpy `_forward_backward`, 236 calls | 10.6 s cumulative (84%) |
  | … of which `safe_divide`, 606,622 calls, one per timestep | 6.3 s, mostly `broadcast_shapes` and allocation |
  | `_corpus_step`, the Cython E-step and count sums, 2,360 calls | 1.15 s |
  | everything else: search, surgery, `HMMParams` validation, `_scan_d` | about 1 s |

  To be tracked as `.scratch/hmm-lush/measurements/search_round_profile.py`. The round after
  §2 is estimated at about 2 s and **not yet measured**.
- **It is the second such finding.** `merge_round_cost.py` found `_suggest_d` taking 53–62% of
  a trial through the same numpy forward pass, which ADR 0023 §7 removed.
- **The original drew the same line.** `HMMLIB-ACCOUNT.md` §12: not one topology-search
  method was compiled, since "search *drives* compiled work … so leaving the driver
  interpreted costs little". §1 agrees for the driver. The profile qualifies the conclusion:
  the driver was running the one uncompiled forward pass.
- **Bit-identity of the phases**: ADR 0020, `test_forward_backward_backends.py` (on
  `tobytes()`), and `docs/api/hmm/baum_welch.md`'s executed comparison of compiled and
  reference `description_lengths`.

## Open

- **The measured round after §2**, replacing the estimate above.
- **Whether a search on `torch` actually leaves the serial path** on the tracked fixtures,
  measured rather than inferred from ADR 0023's step sensitivity.
- **Owed with the implementation, so the record's claims are checkable where they are
  read:** the profiler tracked as `search_round_profile.py`; the `DEFERRED.md` trigger of §3;
  §4's guidance in `docs/api/hmm/baum_welch.md` with executed output; a qualifying note under
  `HMMLIB-ACCOUNT.md` §12 pointing at the profile; and a pointer from ADR 0023 §7 to §2 here,
  which extends it to the convergence check. The branch plan `feat-hmm-convergence-backend`
  lists them as goals, moved there from the master plan's revision 04 goal for this record.

## Resolved

- **How the check names its phase.** Settled 2026-09-17 on `feat/hmm-convergence-backend`:
  a separate `score_backend=` on `baum_welch`, defaulting to `"python"`, and not a phase
  derived from `backend=` (see *Alternatives considered*). The trials' precedent decided it.
  `_trials.py` already carried the keyword but passed it only to `_scan_d`, so the check was
  the one forward pass a named phase did not reach. Under ADR 0021, a separate name is how a
  caller asks for a phase, and nothing chooses one for them.
