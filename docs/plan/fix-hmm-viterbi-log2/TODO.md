# fix/hmm-viterbi-log2

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Unify the logarithm across the Viterbi backends — `docs/plan/TODO.md`,
revision 03-hmm-v0.2.0

## Goals

- [x] Confirm the premise before changing anything
  - [x] Run `/dp-compile:phase-check viterbi`. Expect all of `formalization`, `python`, `cython`, `cpu_parallel` and `cuda` fresh, with the chain `_viterbi.py` → {`FORMALIZATION.md`, `_viterbi_cython.pyx`} → `_viterbi_cpu_parallel.py` → `_viterbi_cuda.py`
    > **Q:** Has `FORMALIZATION.md` been reviewed and approved, and does it stand as the contract this branch works against, including its open "Which host `log2` is contract" paragraph, which goal 2 resolves?
    > **A:** Approved.
    > **Done:** premise confirmed 2026-09-14. Nominal phase **4**; all five phases **fresh**, graph recovered and branching at the root. Recorded hashes, each computed with the target's own header line removed: `_viterbi.py` `6b26469b…` (named by both `FORMALIZATION.md` and `_viterbi_cython.pyx`), `_viterbi_cython.pyx` `79da7c8d…`, `_viterbi_cpu_parallel.py` `5f45abb4…`. `test_viterbi.py`: 89 passed, 0 skipped, header `backends: python ✓ · cython ✓ · cpu_parallel ✓ · cuda ✓` on an NVIDIA L4. All three `.vpath.xls` oracles exercised.
    > **Note:** only `_viterbi.py` lacks a provenance header, correctly: it is the root. Goal 3 should expect **stale then blocked**, not all-stale: editing the `.pyx` leaves `_viterbi_cuda.py` blocked until `_viterbi_cpu_parallel.py` is re-stamped.
  - [x] Measure the numpy-vs-libm `log2` disagreement rate on this host rather than taking DEFERRED's AVX-512 figure (3,937 in 4,000,000) on trust, and find a concrete model whose `total_bits` differs between phase 1 and phases 2-3
    > **Done:** reproduced 2026-09-14 on a 4-vCPU Xeon @ 2.20GHz with AVX-512F/BW/DQ/VL, numpy 2.4.6, numba 0.67.0. Phase 1's `bits` of a contiguous `(S, S)` array takes numpy's SIMD `log2`; numba-scalar `np.log2`, ctypes libm `log2` and `math.log2` agree with one another exactly (0 of 200,000), so phases 2 and 3 share libm's result. Every disagreement is exactly **1 ulp**. The phase-4 test's own generator and seed (`_generated_models(default_rng(20260913), 200, 9)`) reaches it in **1 of 200** models: **model #82, `S=7`, 5 user symbols, `N=16`**, phase 1 `63.37898478021349` against phases 2 and 3 `63.378984780213486` (−1 ulp), states equal. That is the model DEFERRED describes as "7-state, 16-symbol"; the 16 is the sequence length, not the alphabet.
    > **Note:** **the rate is a property of the input distribution, not a host constant**, so DEFERRED's "about 0.1%" should not be quoted without its inputs. Here: 7,763 of 4,000,000 (0.19%) for uniform `x`, 2,872 of 4,000,000 (0.07%) for kernel-shaped `x = u₁·u₂`. By range of `x` the rate climbs from 0.013% below 0.001 to **0.64% in `[0.9, 1)`**. Disagreements concentrate on high-probability arcs, which are the ones a Viterbi path is made of.
    > **Note:** **arc-level disagreement rarely reaches `total_bits`, which is why the existing seeds pass.** A separate sweep of 300 generated models (`S` 2-8, `A` 2-16, `N` 1-59) found 0 differing totals and 0 differing paths across all three later backends. A one-ulp error in a few-bit arc cost is usually lost to rounding when added to a `delta` tens of times larger. So a test that samples models will almost never exercise the split. Goal 4's near-tie has to be **constructed**: choose arc probabilities whose `log2` is known to disagree, placed where the two candidates meet. Relying on a lucky seed would repeat the mistake the existing exact assertions already make.
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
