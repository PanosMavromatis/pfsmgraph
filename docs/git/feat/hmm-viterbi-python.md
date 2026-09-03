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

**2026-09-03 — goal 1, `HMMParams`.** `_params.py` and 44 tests; suite 160 → 204. The
package exports a public name for the first time.

The `A` axis is `vocab.size` with the reserved fibres enforced zero. The argument that
settled it was not the memory cost but a silent-failure asymmetry: `UNK` is the one reserved
code a record acquires on a documented `dataseq` path, and under the user-symbols-only
alternative `1 - USER_BASE = -5` indexes the tail without error.

Two of my own choices were corrected by measurement rather than by argument, and both are
worth remembering as a pattern:

- **`SUM_TOL` was `1e-6` and is `1e-5`.** The bound was *below* the float32 normalisation
  drift its own comment cited as the reason for it. Caught only because the test asserted
  the justification rather than the value.
- **The differential tolerance was `1e-4` and is `5e-4`**, at `1.27e-4` observed. The
  binding error is not the four-decimal print but the *relative* error it implies on a
  small probability, amplified by `d(entropy)/dp ≈ 1.9`. `test_numeric.py` had already
  derived `5e-4` for the same quantity in PR #16, so this is consistency, not slack.

Unplanned but load-bearing: the tracked `.hmm` fixtures cannot be loaded without the ADR
0011 renumbering (their `(S, S, 6)` becomes `(S, S, 12)`), and their `_alphabet` holds Lush
pointer addresses rather than names, so the symbol identities are gone. `_lush_fixtures.py`
was extracted to hold `load_params` — the reader's own docstring named revision 03 as the
trigger for sharing it, and the second consumer arrived here in 02.

**2026-09-03 — goal 2, the δ-seeding decision.** Decided: **fix it**, `delta[0] =
bits(init_state_p)`. What makes the entry worth reading is that the goal expected an
argument and got a measurement instead.

**The original's own decodes are on disk.** `save-viterbi-path` wrote a `<model>.vpath.xls`
next to each saved model — `Output / States / Entropy` per position — and all three tracked
fixtures have one; the corpus sits beside them at `set02a_200.sds`. Concatenating its 200
`.seq` files reproduces the vpath `Output` column exactly, so the port has an aligned
**decode oracle**. Running the §3 min-sum recurrence against it, using goal 1's
`load_params` and a sixty-line probe:

| model | S | original seed | corrected seed |
|---|---|---|---|
| `m001_0001_001` | 1 | 1269/1269 | 1269/1269 |
| `m001_0005_005` | 5 | 1269/1269 | 1269/1269 |
| `m008_0001_008` | 8 | 1269/1269 | 1268/1269 — position 0 only |

So the kernel is validated before goal 3 writes it, and fixing the defect costs one position
in 3807 — at `m008` position 0, where the two live start states' best arcs differ by 0.004
bits and the seed alone decides it: the original prefers `init_p` 0.3665 to 0.6335.

Two things the account could not have known, both found by measuring rather than reading:

- **The degenerate case is masked by the learned topology.** Every `init_p == 0` state is
  also unable to emit `begin` on any outgoing arc, so `+inf` absorbs before δ = 0.0 can win.
  That correlation is a property of trained models, not an invariant — and revision 04's
  `split-state` is what decouples it. The fixtures therefore cannot exhibit the defect's
  worse half, which is itself an argument for fixing rather than reproducing.
- **The symbol names are not gone**, only separated: the *corpus* `_alphabet` has them. Goal
  1's note was true of the model directory and false of the repository. Corrected in
  `core.md`; the sharpened form is better evidence for the same ADR 0001 conclusion.

The pattern from goal 1 held again, in a third form. There it was two assertions written
from reasoning and corrected by measurement; here it was a whole *decision* framed as a
judgement call that turned out to be an empirical question, and the evidence had been
sitting untracked in `.scratch/` since import. Worth carrying into goal 3: before deciding
what the fixtures can support, look at what is actually in the directory.
