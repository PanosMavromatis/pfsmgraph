# feat/hmm-torch-backend

**Created**: 2026-09-14
**Base**: main at f5e3d2a
**Status**: active

## Purpose

Add `pfsmgraph-hmm`'s optional `torch` backend for revision 03: the forward pass and an autograd
E-step, behind a new `[torch]` extra. numpy computes the expected counts (ξ, γ) explicitly;
torch gets them as `.grad` on the log-parameters. The branch checks that the two agree through
ADR 0021's `backend=` seam, as an ADR 0003 cross-backend test. It also records the tolerance
that turned out to be needed, against ADR 0020's promise of "a few ulps".

## Scope

- Decide what public call `backend="torch"` attaches to (forward-backward has none; `_PUBLIC`
  is `{"viterbi"}`), and declare the `[torch]` extra, which must stay separate from `gpu`.
- torch forward pass and autograd E-step, added as a fifth `_Row` whose absence is attributable to torch.
- Count-identity test against the numpy reference; measure and record the tolerance.
- Docs: `docs/api/hmm/`, ADR 0020/0021 notes, `core.md`, master plan.

## Context

- Master plan: `docs/plan/TODO.md`, the torch subgoal of revision 03-hmm-v0.2.0 (note from PR #33).
- ADR 0020 (scaled probability domain; torch held to a stated tolerance), ADR 0021 (backend
  selection), ADR 0003 (parameterised suites), ADR 0015 (arc emission).
- Core invariant: "GPU" means two things; torch is `dl`'s and not the `gpu` extra.
- hmmlearn oracle (PR #32) validates the numpy reference independently of torch.

## Notes
