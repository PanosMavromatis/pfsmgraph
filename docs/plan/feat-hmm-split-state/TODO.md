# feat/hmm-split-state

**Status**: active
**Created**: 2026-09-16
**Subgoal**: Implement state split — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [ ] Specify what `split-state` does to each array before writing code
  - [ ] Transcribe the assignment order of `hmm-param.lsh:142-215` into a per-array statement of the result: which entries are copied, halved, or drawn, and which assignments later ones overwrite
  - [ ] Check the reading against the generated C, where one exists for `hmm-param.lsh`, as `util.c` settled the code-length primitives
  - [ ] State how the original's whole-alphabet `rand-p-vector` maps onto `HMMParams`' zero reserved fibres, and what that does to the number of draws per fibre
- [ ] Decide the `hmm-param.lsh:172` initial-distribution defect
  - [ ] Establish what the defect does to a split model: every uninvolved state's initial probability replaced by its stationary probability, with no renormalization. Measure whether the result still sums to one on the tracked fixtures, and whether it passes `HMMParams`' `SUM_TOL`
  - [ ] Decide reproduce, fix, or fix differently, and record it where it binds, knowing `merge-states` carries the same defect at `:262` and inherits the decision
- [ ] Settle the reproducibility contract
  - [ ] Decide whose `numpy.random.Generator` a split draws from and where it enters: a required argument to the split, to the search, or both
  - [ ] Fix the draw order as contract, since the same seed must produce the same candidate, and decide whether it follows the original's order or array order
- [ ] Implement the split as a free function over `HMMParams`, with tests
  - [ ] Write the function: fresh arrays, returning `HMMParams(init_state_p, transition_p, output_p, params.vocabulary)`
  - [ ] Test the invariants that hold for any model: row sums, zero reserved fibres, inbound mass conserved across the pair, likelihood of a sequence unchanged before randomization where that holds
  - [ ] Construct the case the fixtures cannot: a state with `init_p == 0` after a split that can still emit `begin`, and decode it, so revision 02's δ-seeding fix is exercised for the first time
  - [ ] Pin reproducibility: the same `Generator` state yields a bit-identical candidate
- [ ] Decide where `try-split` and `suggest-split` belong
  - [ ] `try-split` re-converges and chooses `d`, and `suggest-split` ranks trials: port them here, or leave them to the scored-primitives subgoal and the search loop, and record which
