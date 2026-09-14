# fix/hmm-viterbi-log2

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Unify the logarithm across the Viterbi backends — `docs/plan/TODO.md`,
revision 03-hmm-v0.2.0

## Goals

- [ ] Confirm the premise before changing anything
  - [ ] Run `/dp-compile:phase-check viterbi`. Expect all of `formalization`, `python`, `cython`, `cpu_parallel` and `cuda` fresh, with the chain `_viterbi.py` → {`FORMALIZATION.md`, `_viterbi_cython.pyx`} → `_viterbi_cpu_parallel.py` → `_viterbi_cuda.py`
  - [ ] Measure the numpy-vs-libm `log2` disagreement rate on this host rather than taking DEFERRED's AVX-512 figure (3,937 in 4,000,000) on trust, and find a concrete model whose `total_bits` differs between phase 1 and phases 2-3
- [ ] Choose the fix and state which `log2` is contract
  - [ ] (a) One numpy table for every backend, as phase 4 builds it: costs for the distinct symbols in the record, `(S, S, |present|)`. Weigh this against phase 1's refusal of the `(S, S, A)` precompute as `O(S²·A)`. The per-record gather is bounded by the symbols used, not the alphabet, but it is still a table, and phases 2-3 would stop computing `log2` in their inner loops
  - [ ] (b) libm everywhere: route phase 1 through scalar `log2`, and phase 4's host table with it. This loses numpy vectorisation in phase 1
- [ ] Make the change in the backends and regenerate the provenance chain
  - [ ] Edit the kernels the chosen fix touches. Keep the operation order (multiply, `-log2`, add) and the first-wins ascending reduction unchanged
  - [ ] Regenerate every `dp-compile: derived-from` header downstream of an edited file, including `_viterbi_cuda.py`'s even if its body is unchanged, and `FORMALIZATION.md`'s `Derived from` hash if `_viterbi.py` changes
- [ ] Tighten the tests
  - [ ] Replace phase 4's one-ulp-per-arc bound against phases 2-3 with exact equality
  - [ ] Construct a near-tie whose two candidate costs fall in the disagreeing inputs, and assert all four backends choose the same predecessor
- [ ] Update the documentation
  - [ ] `FORMALIZATION.md`: name which `log2` is contract, replacing "remains open"
  - [ ] Correct the bit-exactness claims in the phase 2 and 3 docstrings, and phase 4's paragraph about the split
  - [ ] Remove the entry from `DEFERRED.md` and the "not bit-identical" caveat from `core.md`
