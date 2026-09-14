# feat/hmm-forward-backward

**Created**: 2026-09-14
**Base**: main at b53fdf8
**Status**: active

## Purpose

Implement forward-backward in numpy as the reference implementation for revision
03-hmm-v0.2.0 (Baum-Welch on a fixed topology). The branch computes α, β, ξ, γ and the
sequence likelihood in log space over the arc-emission (Mealy) model, with `+inf` for
impossible events, and first settles whether the log base stays 2 or becomes *e* with a
conversion at the description-length boundary. Everything later in the revision builds
on it: the M-step and EM loop, the `hmmlearn` oracle, the backend seam, the `torch`
backend, and the compiled phases.

## Scope

- Settle the log base (2 vs *e*) from what the Lush `update-data-dl` and `run-add`
  accumulate, including the semiring the phases 2-4 subgoal should name.
- A private, purely numeric α/β kernel that never raises. The arc-emission factor stays
  inside the inner loop (ADR 0015).
- ξ, γ and the sequence likelihood, with consistency checks between them.
- Tests against brute-force path enumeration on small models, plus constructed cases:
  an impossible sequence, an exactly uniform model, and zero-probability arcs.

Out of scope: the `hmmlearn` oracle (its own subgoal), the M-step and EM loop, a phase-0
`FORMALIZATION.md`, and compiled phases.

## Context

- Master plan: `docs/plan/TODO.md`, revision 03-hmm-v0.2.0, the forward-backward subgoal.
- Previous subgoal: `fix/hmm-viterbi-log2` (PR #28) settled numpy's `log2` through `bits`
  as the decode's contract. It covers the decode only, and forward-backward needs
  `log`/`exp` inside the recurrence.
- `HMMLIB-ACCOUNT.md` §3 and `hmm-trainer.lsh` (`run-add` 483-652, `update-data-dl`
  346-402) under `.scratch/hmm-lush/`.
- ADR 0015 (arc-emission), ADR 0017 (frozen parameters), ADR 0003 (test-suite shape).

## Notes
