# feat/hmm-viterbi-python

**Created**: 2026-09-03
**Base**: main at a0e9c19
**Status**: active

## Purpose

Land the project's first dynamic-programming kernel — Viterbi at
[ADR 0002](../../design/adr/0002-three-phase-algorithm-lifecycle.md) phase 1, pure
Python/numpy — together with the `HMMParams` value it operates on. The parameter object was
*specified* by [ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md) on the
`feat/hmm-public-surface` branch (PR #15) and **no commit has yet built it**: today
`packages/pfsmgraph-hmm/` holds `_numeric.py` and an empty `__init__.py`, so the package
still exports nothing. The two are inseparable here — PR #15 settled that Viterbi is a free
function *over* parameters rather than a method on a trainer, which means the decode cannot
be written before the value it takes exists.

Landing the kernel also fills the [ADR 0003](../../design/adr/0003-parameterized-backend-test-suite.md)
backend matrix for the first time. It has been empty since the reporting hook landed on
2026-09-01, not as a placeholder but as the correct steady state — `dataseq` contributes no
recurrence at any maturity. This branch is the event `_backends.py`'s module docstring names
as the one that changes that, so the session header stops reading
`backends: none registered — no DP kernel has reached ADR 0002 phase 1`.

## Scope

- `HMMParams` per ADR 0017: `init_state_p`, `transition_p`, `output_p` frozen with
  `writeable = False`; the `Vocabulary` retained as `dataseq`'s Protocol; `state_p` and the
  state entropies as cached properties over `_numeric.py`.
- The `A`-axis decision PR #15 deliberately left open — `vocab.size` (reserved block
  included) or user symbols only. It changes the `(S, S, A)` shape, so it belongs here.
- The δ-seeding defect: fixed, or faithfully reproduced. A decision either way, recorded.
- The two-layer decode PR #15 specified: private kernel
  `_viterbi(init_p, transition_p, output_p, codes)`, public `viterbi(params, record)`
  returning a `ViterbiPath`.
- The `python` row in `_backends.py`, and the repo-root backend tests that currently assert
  an empty matrix.

## Context

**Decided upstream, not reopened here.** PR #15 (`docs/plan/feat-hmm-public-surface/TODO.md`)
settled the class architecture, the two-layer split, the `Vocabulary` seam, and the
`pad_collate` deferral. PR #16 (`docs/plan/feat-hmm-numeric-utils/TODO.md`) landed
`_numeric.py`, whose `stationary_distribution` and `entropy` are what `HMMParams`'s cached
properties call.

**The two facts most likely to be lost in translation**, both already recorded as
invariants:

- [ADR 0015](../../design/adr/0015-arc-emission-mealy-formulation.md) — arc-emission
  (Mealy). The emission parameter is `output_p[i, j, symbol]`, never `B[state, symbol]`, so
  a path over *N* symbols visits *N+1* states and the emission factor **cannot** be hoisted
  out of the inner loop: it depends on both endpoints.
- `HMMLIB-ACCOUNT.md` §3 — the accumulator holds **description lengths, not
  probabilities**. They grow as the probability falls, so this is a **min-sum**, and a port
  reaching for `max` inverts every comparison.

**The defect this branch must decide about** (`HMMLIB-ACCOUNT.md` §7, marked *provenance
unknown*): `update-viterbi-path` seeds δ with raw `init-state-p` into a bit-domain
accumulator (`hmm-trainer.lsh:216-218`). Two consequences, and the second is worse than the
first. The distortion is bounded by one bit, since `init_p ≤ 1`, so it only mis-decides
between paths within a bit of each other after the first transition — but within that band
it decides them backwards, preferring *improbable* start states. And an exactly-zero
`init_p[j]` yields δ = 0.0, the **best** possible value, where it should be impossible;
`init-random` never produces an exact zero but `split-state` halves initial probabilities
and `merge-states` sums them, so revision 04 can. The identical line is *correct* in
`update-data-p`, where α is a raw probability throughout — it reads as a copy between two
methods that do not share a numeric domain.

The account's second §7 defect — `psi` as a float matrix round-tripping state indices — is
already decided by the master plan: harmless below 2²⁴ states, not reproduced.

## Notes

<!-- Running log. Append as work progresses. -->
