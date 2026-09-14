# feat/hmm-em-loop

**Created**: 2026-09-14
**Base**: main at 21852cb
**Status**: active

## Purpose

Baum-Welch re-estimation and the EM loop for `pfsmgraph-hmm`, the third subgoal of
revision 03-hmm-v0.2.0. PR #30 left the scaled forward-backward and the three expected
count arrays (`C-in`, `C-t`, `C-t-y`) on `main`; this branch turns those counts into new
parameters, iterates to convergence as `run-converge` does, and ports the data description
length by reusing the forward pass rather than writing a fourth copy of the recurrence.

## Scope

- Decide what re-estimation does at zero counts, since `safe-/` turns an unvisited state
  into a zero `transition_p` row and `HMMParams` rejects one.
- The M-step as a private numeric function, tested against exact rational re-estimation
  and constructed zero-count cases.
- The EM loop and `run-converge`'s stopping rule, tested for a non-decreasing likelihood.
- The data description length, and whether quantization lands here or in revision 04.

## Context

- PR #30 (`feat/hmm-forward-backward`), plan `feat-hmm-forward-backward` under `docs/plan/`.
- ADR 0015 (arc emission), ADR 0017 (frozen parameters), ADR 0020 (scaled domain, fixed
  evaluation order, no fused multiply-add).
- `.scratch/hmm-lush/Code/HMMlib/hmm-trainer.lsh`: `update-data-dl` 346-402, `run-add`
  483-652 (M-step 620-651), `run-converge` 661-674.
- `HMMLIB-ACCOUNT.md` §9.

## Notes

- `update-data-dl` is not `_description_length`: it runs the same scaled forward pass over
  the quantized `-r` parameters ("approximate p matrices"). Sharing its code means calling
  the existing kernel on quantized parameters, and it explains why the fixtures' `_total_dl`
  sat 1.5-29 bits above the unquantized value.
- `*factor*` is `1.0` (`hmm-trainer.lsh:10`), so `run-converge` stops after 3 consecutive
  `run-add 10` batches whose `data-p` moves by less than 0.1 bits. It watches `data-p`, the
  unquantized likelihood, not the DL.
- A state with no expected outgoing count gets an all-zero `transition_p` row under
  `safe-/`, which `HMMParams` rejects by design; EM can construct that case unaided.
