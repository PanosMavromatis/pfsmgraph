# feat/hmm-forward-backward

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Implement forward-backward in numpy as the reference — `docs/plan/TODO.md`,
revision 03-hmm-v0.2.0

## Goals

- [ ] Settle the log base, 2 or *e*, before writing any code
  - [ ] Read what `update-data-dl` (`hmm-trainer.lsh:346-402`) and `run-add` (`483-652`) accumulate, and in which base
  - [ ] State what each option does to `bits`, to log-sum-exp inside the recurrence, and to the one-host numeric contract settled by `fix/hmm-viterbi-log2`
  - [ ] Record the decision, and the semiring the master plan's phases 2-4 subgoal should name
- [ ] α and β in log space as a private, purely numeric kernel
  - [ ] It never raises: an impossible sequence comes back as an infinite cost, as in `_viterbi`
  - [ ] The arc-emission factor stays inside the inner loop (ADR 0015)
- [ ] ξ, γ and the sequence likelihood
  - [ ] γ sums to 1 over states at every position, ξ marginalises to γ, and α and β give the same total
- [ ] Tests against an oracle inside the repository
  - [ ] Brute-force enumeration over every state path on small models
  - [ ] Constructed cases: an impossible sequence, an exactly uniform model, zero-probability arcs
