# feat/hmm-em-loop

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Implement the M-step and the EM loop — `docs/plan/TODO.md`, revision 03-hmm-v0.2.0

## Goals

- [x] Settle what re-estimation does where the counts are zero, before writing it
  > **Q:** When an M-step leaves state `j` with zero expected outgoing count, what should the re-estimated parameters be: keep the old row and fibres, raise a typed error, or relax `HMMParams` to allow a zero row?
  > **A:** Keep the old row and fibres. The numeric M-step applies `safe_divide` as Lush does and returns the out-counts too; the EM wrapper restores state `j`'s previous transition row and emission fibres and reports which states were degenerate. Pruning stays with revision 04.
  - [x] Read `run-add`'s M-step (`hmm-trainer.lsh:620-651`) and list every `safe-/` that can see a zero denominator
    > **Note:** Three `safe-/`. (1) `init_state_p` over `Σ C_in`, which is 1 up to rounding and zero only for an impossible or empty record, where every count is zero. (2) `transition_p[j]` over `Σ_k C_t[j,k] = Σ_{t<N} γ_t(j)`, zero whenever state `j` is never occupied before the last position: unreachable, or reachable only by the final symbol. That gives an all-zero row, which `HMMParams` rejects. (3) `output_p[j,k]` over `C_t[j,k]`, zero on every arc with no posterior mass. Its numerator is also the new `transition_p[j,k]`'s numerator, so the fibre only ever lands on a dead arc, which `HMMParams` exempts. Harmless by construction.
    > **Note:** Measured with a scratch script: a 3-state model whose state 2 is reachable only by the final `b` and emits only `c` goes from a valid row `[0.5, 0.5, 0]` to `[0, 0, 0]` in **one** cycle. Zero is absorbing under EM, since every count is a product containing the old parameter, so the row never recovers, and the arc into state 2 decays 0.18 → 0.09 → 0.04 without reaching zero.
  - [x] Decide what an unvisited state becomes, given `HMMParams` rejects a zero `transition_p` row, and whether the M-step's sums follow ADR 0020's order
    > **Note:** Measured with a scratch script: restoring a zero-count state's old row *and* fibres gives description lengths bit-identical (`==`) to Lush's zero row over 5 cycles, falling monotonically from 2.96 to 2.63 bits. Such a state has `γ_t(j) = 0` for `t < N`, so its row reaches no α or β that matters, and every row maximises EM's objective for it, so Baum's non-decrease guarantee survives. The fibres have to be restored with the row: the re-estimated ones are all zero, and a zero fibre on a live arc is exactly what `HMMParams` rejects.
    > **Note:** Sum order follows ADR 0020 with no new rule needed. Lush sums the out-count over `k` in ascending order and the init count over `j` in ascending order, then divides elementwise; `np.add.accumulate(...)[-1]` reproduces each, and `_expected_counts` already adds ξ in ascending `t`. The path from counts to parameters therefore has one fixed order, and a compiled phase can be bit-exact with it.
- [x] The M-step as a private numeric function from the three count arrays to new parameters
  > **Q:** Where should the M-step live: a new `_baum_welch.py`, or inside `_forward_backward.py` beside `_expected_counts`?
  > **A:** A new `_baum_welch.py`, holding `_m_step` now and the EM loop next. `_forward_backward.py` stays the pure DP kernel that dp-compile registers and gates; the M-step is elementwise division with no recurrence, so it has no lifecycle phases to gate.
  - [x] Tests: exact rational re-estimation, stochasticity, the reserved fibres staying zero, and constructed zero-count cases
    > **Note:** 49 tests in `tests/test_baum_welch.py`, over counts from the real E-step (4 random models and the `m008` fixture) rather than invented arrays, since the M-step's contract assumes emission counts that marginalise to transition counts. Mutation check before landing: out-count on the wrong axis failed 18 tests, `np.sum` in place of the ascending `accumulate` 2, a descending init total 3, a transposed emission denominator 9. The `np.sum` mutant is caught only by the bit-for-bit order pin, which is why that section exists.
- [ ] The EM loop and its convergence rule
  - [ ] Port `run-converge` (`661-674`): batches of `run-add 10`, stop after 3 consecutive changes in `data-p` below `0.1 · *factor*` bits, `*factor*` = 1.0
  - [ ] Decide what a training corpus is (the original trains on one flat concatenated stream) and what the loop returns
  - [ ] Tests: likelihood never decreases between cycles, and a model at a fixed point stays there
- [ ] The data description length, sharing the forward pass's code
  - [ ] Establish that `update-data-dl` is the forward pass over quantized `-r` parameters, and decide whether quantization lands here or with revision 04's MDL scoring
  - [ ] Evaluate the tracked fixtures' `_total_dl` as an oracle once the answer is known
