# feat/hmm-viterbi-python

**Status**: active
**Created**: 2026-09-03
**Subgoal**: Implement Viterbi at ADR 0002 phase 1 (pure Python/numpy) with the ADR 0003 test suite, and register it as the first backend (revision `02-hmm-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [ ] Implement `HMMParams`, the ADR 0017 frozen parameter value
  - [ ] Settle the `A` axis — `vocab.size` (reserved block included, so the emission tensor carries fibres for `PAD`/`UNK`/… that must never be emitted) or only the user symbols. PR #15 surfaced this and deliberately did not decide it, because it changes the `(S, S, A)` shape and so belongs with the kernel rather than with the boundary.
  - [ ] Freeze the three arrays with `writeable = False` and retain the `Vocabulary` as `dataseq`'s `Protocol`, held for `size` and for identity comparison — not for any symbol string
  - [ ] `state_p` and the state entropies as cached properties over `_numeric.py`: `stationary_distribution(transition_p)`, and `entropy` of the `(S, A)` marginal `np.einsum("ij,ijk->ik", transition_p, output_p)`. ADR 0017 makes these derived rather than stored, so a reducible chain surfaces its `ValueError` on an attribute access.

- [ ] Decide the δ-seeding defect: fix it, or reproduce it faithfully
  - [ ] Weigh the two consequences separately (`HMMLIB-ACCOUNT.md` §7). The typical case is bounded by one bit and decides *backwards* within that band; the degenerate case turns an impossible start state into the best possible δ, and revision 04's `split-state`/`merge-states` can produce one where `init-random` cannot.
  - [ ] Record the decision where the ADR 0003 suite encodes the choice rather than accidentally validating the bug against itself
  - [ ] Note `psi`'s float round-trip as decided-not-reproduced — the master plan already settled it as harmless below 2²⁴ states

- [ ] Implement Viterbi at ADR 0002 phase 1 (pure Python/numpy)
  - [ ] The private kernel `_viterbi(init_p, transition_p, output_p, codes)` — a **min-sum** over bits, not a max-product (`HMMLIB-ACCOUNT.md` §3), with `output_p[i, j, symbol]` unhoistable from the inner loop because it depends on both endpoints (ADR 0015)
  - [ ] The public `viterbi(params, record) -> ViterbiPath` wrapper, taking one `SequenceRecord` and returning a result rather than writing back into it. `N` symbols visit `N+1` states; `ViterbiPath.states` is not decoded and `.label` is the only string that passes through.
  - [ ] Establish whether the three tracked `.hmm` fixtures support a differential test of the decode, as they did for the stationary solve. Their four-decimal print format is a known trap — see PR #16.

- [ ] Register `python` as the first ADR 0003 backend
  - [ ] Add the row to `_backends.py`, with `hardware=None` so a failed import escalates rather than skips
  - [ ] Update the repo-root backend tests that currently assert an empty matrix and the `EMPTY_HEADER` line
  - [ ] Confirm the parameterized suite shape works with one backend, since backend *equivalence* has nothing to compare against until phase 2
