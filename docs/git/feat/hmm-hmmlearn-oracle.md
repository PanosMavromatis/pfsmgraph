# feat/hmm-hmmlearn-oracle

**Created**: 2026-09-14
**Base**: main at 3386a5d
**Status**: active

## Purpose

Validate the numpy forward-backward and the EM loop against an external oracle before
anything else in revision 03 is built on them. An arc-emission model whose emission depends
only on the destination state is a state-emission model (ADR 0015), so that reduced case can
be checked against `hmmlearn`'s `CategoricalHMM` to machine precision. `hmmlearn` shares no
lineage with our derivation, which is what lets it catch a misreading of the model that the
planned `torch` backend, being autograd over the same derivation, would share.

## Scope

- Add `hmmlearn` to the root `dev` group as a test-only dependency; confirm it does not
  tighten the existing numpy ceiling from `numba-cuda`.
- Define the mapping between the two formulations: `hmmlearn`'s start, transition and
  emission probabilities to our `(S,)`, `(S, S)`, `(S, S, A)` arrays, the reserved-symbol
  offset, and the N+1 state geometry at the first step.
- Differential tests of the forward-backward: log-likelihood against our description
  length, and state posteriors, over random reduced models.
- Differential tests of EM: fixed iteration counts on both sides from the same start,
  compared parameter by parameter.
- Update `core.md` test counts and record what the oracle found.

## Context

- Master plan: `docs/plan/TODO.md`, revision 03-hmm-v0.2.0, the external-oracle subgoal.
- `docs/design/arc-emission-hmm-handoff.md` §3, the source of the idea.
- ADR 0015 (arc emission), ADR 0020 (scaled probability domain), ADR 0003 (why this is not
  a backend).
- Builds on `_forward_backward.py` (PR for `feat/hmm-forward-backward`) and `_baum_welch.py`
  (PR #31).

## Notes
