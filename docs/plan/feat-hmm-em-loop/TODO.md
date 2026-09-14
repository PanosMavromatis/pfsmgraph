# feat/hmm-em-loop

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Implement the M-step and the EM loop — `docs/plan/TODO.md`, revision 03-hmm-v0.2.0

## Goals

- [ ] Settle what re-estimation does where the counts are zero, before writing it
  - [ ] Read `run-add`'s M-step (`hmm-trainer.lsh:620-651`) and list every `safe-/` that can see a zero denominator
  - [ ] Decide what an unvisited state becomes, given `HMMParams` rejects a zero `transition_p` row, and whether the M-step's sums follow ADR 0020's order
- [ ] The M-step as a private numeric function from the three count arrays to new parameters
  - [ ] Tests: exact rational re-estimation, stochasticity, the reserved fibres staying zero, and constructed zero-count cases
- [ ] The EM loop and its convergence rule
  - [ ] Port `run-converge` (`661-674`): batches of `run-add 10`, stop after 3 consecutive changes in `data-p` below `0.1 · *factor*` bits, `*factor*` = 1.0
  - [ ] Decide what a training corpus is (the original trains on one flat concatenated stream) and what the loop returns
  - [ ] Tests: likelihood never decreases between cycles, and a model at a fixed point stays there
- [ ] The data description length, sharing the forward pass's code
  - [ ] Establish that `update-data-dl` is the forward pass over quantized `-r` parameters, and decide whether quantization lands here or with revision 04's MDL scoring
  - [ ] Evaluate the tracked fixtures' `_total_dl` as an oracle once the answer is known
