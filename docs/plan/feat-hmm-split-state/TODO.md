# feat/hmm-split-state

**Status**: active
**Created**: 2026-09-16
**Subgoal**: Implement state split — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [x] Specify what `split-state` does to each array before writing code
  > **Q:** Where should the per-array specification live durably?
  > **A:** As a note here now, turned into the split function's docstring at goal 4. No new document: the split is not a DP kernel, so it gets no `FORMALIZATION.md`.
  - [x] Transcribe the assignment order of `hmm-param.lsh:142-215` into a per-array statement of the result: which entries are copied, halved, or drawn, and which assignments later ones overwrite
    > **Note:** `s` = `state-n`, `p` = the new state at index `S` (old size); `T`, `O` old, `T'`, `O'` new; later assignments overwrite earlier ones.
    > **`init'`:** `init'[i] = state_p[i]` for uninvolved `i` (`:172`, goal 2's defect); `init'[s] = init'[p] = ½·init[s]` (`:179-180`).
    > **`T'`:** the S×S block is copied first. Inbound, for every `j` *including* `s`: `T'[j,s] = T'[j,p] = ½·T[j,s]` (`:184-185`). Outbound: `T'[p,j] = T[s,j]`, **unhalved** (`:194`), so for `j ≠ s` the partner's row is an exact copy of `s`'s. Self-loops: `T'[p,s] = T'[p,p] = ½·T[s,s]` (`:203-204`), overwriting `:194`'s unhalved `T'[p,s]`. Rows `s` and `p` each sum to one, and every other row keeps `T'[j,s] + T'[j,p] = T[j,s]`.
    > **`O'`:** the S×S block is copied first. Inbound: `O'[j,p] = O[j,s]` for every `j` (`:187`); for `j ≠ s` that copy is final and `O'[j,s]` keeps its old value. Every fibre *leaving* `s` or `p` is drawn at noise 0.1, overwriting `:187`'s copy at `O'[s,p]`. That is `2S + 2` draws in the order `(s,0) (p,0) (s,1) (p,1) … (s,S-1) (p,S-1) (s,p) (p,p)`.
    > **`state_p`, `state_entropies`:** left as zero arrays; computed rather than stored under ADR 0017, so this dissolves.
    > **Two consequences.** Emissions alone break the symmetry between `s` and `p`, since inbound fibres are copies and outbound rows are identical for `j ≠ s`; revision 03's saddle finding makes the draw necessary rather than decorative. And every outbound fibre of both states can emit `begin` (~1/6 of its mass on the fixture alphabet), which is the input goal 4's δ-seeding case needs.
  - [x] Check the reading against the generated C, where one exists for `hmm-param.lsh`, as `util.c` settled the code-length primitives
    > **Note:** No generated C exists for `hmm-param.lsh`: `Code/Utility/C/util.c` is the import's only DH output. The transcription rests on plain assignments and needs no primitive's meaning settled, except `rand-p-vector`. For that one, `C_rand_p_vector` (`util.c:721-771`) confirms `1 + w·U(-1,1)` per element, then normalisation, all in `flt` (float32), matching `_numeric.rand_p_vector` apart from precision. Bit reproduction against Lush was never available, since the generators differ.
  - [x] State how the original's whole-alphabet `rand-p-vector` maps onto `HMMParams`' zero reserved fibres, and what that does to the number of draws per fibre
    > **Note:** It does nothing to the count. Lush's alphabet holds `begin` and `end` as ordinary codes 0 and 1, and `_lush_fixtures.load_params` maps them to user symbols rather than BOS/EOS (its docstring says why). So a Lush whole-alphabet fibre has exactly `n_symbols - USER_BASE` elements: draw `rand_p_vector(n_symbols - USER_BASE, 0.1, rng)` into `[..., USER_BASE:]` and leave the reserved block zero. The branch doc's context bullet, which implied the count changes, is sharpened accordingly.
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
