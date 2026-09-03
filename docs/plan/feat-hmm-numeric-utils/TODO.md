# feat/hmm-numeric-utils

**Status**: active
**Created**: 2026-09-03
**Subgoal**: Migrate the Utility code this release needs, private to the package (revision `02-hmm-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [ ] Read `Code/Utility/util.lsh` and settle where the migrated code lands
  - [ ] Read the six migrating functions and the LU trio in their own terms, against `HMMLIB-ACCOUNT.md` §3 and §4
  - [ ] Decide the module layout — whether `_numeric.py` carries all of it, or the solve and the initialisation split out
  - [ ] Check whether `Code/Utility/C/util.c` is a compiled counterpart with semantics of its own; it is untracked, ignored by `.scratch/hmm-lush/.gitignore:91`, and the tracked set only ever widens
  - [ ] Create `packages/pfsmgraph-hmm/tests/`, the package's first

- [ ] Migrate the log-domain arithmetic Viterbi's inner loop calls
  - [ ] `safe-add--log2` and `safe->--log`, and decide the fate of the `-1` log-zero sentinel (§3) — faithfully reproduced, or replaced by `-inf`
  - [ ] `safe-/` (15 call sites) and `int-delta`
  - [ ] Tests, including behaviour at the sentinel boundary

- [ ] Port the stationary-distribution solve
  - [ ] Reproduce the row-replacement trick, not just the result: row 0 of `(Pᵀ - I)` overwritten with `Σπ = 1` before `numpy.linalg.solve` (§4)
  - [ ] Test against a chain whose stationary distribution is known in closed form
  - [ ] Record that `LU-solve`, `LU-decomposition` and `LU-back-substitution` were replaced by a library call rather than translated

- [ ] Port `rand-p-vector` and `calculate-entropy`, and record what migrates nowhere
  - [ ] `rand-p-vector` for parameter initialisation — settle the RNG seam: a passed-in `numpy.random.Generator` versus module-level state
  - [ ] `calculate-entropy`
  - [ ] Record that `minimize` / `minimize-from` / `minimize-int` and `mc.lsh` had **zero** call sites from `HMMlib`, and so migrate nowhere
  - [ ] Review the member's declared `numpy>=2.1` against what this code actually uses — the first numpy in `hmm`, so the first moment the bound is checkable
