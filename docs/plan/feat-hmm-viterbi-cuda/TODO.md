# feat/hmm-viterbi-cuda

**Status**: active
**Created**: 2026-09-13
**Subgoal**: Implement Viterbi at ADR 0002 phase 4 (Numba CUDA) —
`docs/plan/TODO.md`, revision 02-hmm-v0.1.0

## How this branch is driven

Same three invocations as phases 2 and 3, one row changed: `/dp-compile:phase-check
viterbi` reports, `/dp-compile:next-phase viterbi` resolves the target and routes, and the
skill it routes to here is `dp-compile:gpu-parallelization`. **Enter through
`/next-phase`**, never the skill directly — it includes `references/phase-detection.md`,
which carries the recovered-graph rules a phase skill's own prerequisites do not.

**The expected report before any work.** Nominal phase **3**; `formalization`, `python`,
`cython` and `cpu_parallel` fresh; `cuda` absent; target *write `cuda`*. Confirm rather
than assume — the premise is what the first goal exists to check.

## Goals

- [ ] Confirm the plugin's view of the algorithm matches this branch's premise
  - [ ] `/dp-compile:phase-check viterbi` reports nominal phase 3, nothing stale, target `cuda`
  - [ ] Suite green at 298 on this machine, header `backends: python ✓ · cython ✓ · cpu_parallel ✓`
- [ ] Settle the GPU environment before the first gated commit
  - [ ] Decide how `numba-cuda` survives `uv sync` — the gate's build command — given it is currently installed outside the lock and a plain sync uninstalls it
  - [ ] Pin the `numba-cuda` lower bound in `pfsmgraph-hmm`'s `gpu` extra and close the `DEFERRED.md` entry; check what numpy ceiling it imposes
- [ ] Implement `_viterbi_cuda.py` at ADR 0002 phase 4
  - [ ] Transliterate the phase-3 decomposition (states within one timestep; serial ascending reduction over `i`) under `cuda.jit`, keeping the kernel purely numeric
  - [ ] Register the backend in `_backends.py` with `optional_on`, and add the module to `meson.build`'s `install_sources`
- [ ] Hold it equivalent to phases 1–3
  - [ ] Differential tests in `test_viterbi.py`'s labelled non-shared section, including the constructed uniform-model tie test
  - [ ] Verify an absent device skips loudly and `PFSMGRAPH_REQUIRE_BACKENDS` escalates it
- [ ] Measure and record
  - [ ] Launch overhead against phase 3's flat ~34 ms; update `core.md`, `FORMALIZATION.md` and ADR 0016 as the result warrants
