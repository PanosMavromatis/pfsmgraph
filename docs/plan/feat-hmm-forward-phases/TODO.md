# feat/hmm-forward-phases

**Status**: active
**Created**: 2026-09-15
**Subgoal**: Carry the forward recurrence through ADR 0002 phases 2 through 4 (revision 03-hmm-v0.2.0)

## How this branch is driven

Every goal is a `dp-compile` phase. Commands come from that plugin and commits from
`workflow-claude`, the same arrangement as `feat-hmm-viterbi-cython`:

| Invocation | Role |
|---|---|
| `/dp-compile:phase-check forward_backward` | Reports only. Run at each goal's start to confirm the target, and again at the end. |
| `/dp-compile:next-phase forward_backward` | Resolves the target and routes to the skill for **the target's own phase**. The only entrance used here. |
| `dp-compile:algorithm-recover` → `cython-translation` → `cpu-parallelization` → `gpu-parallelization` | The skills `/next-phase` routes to, in goal order. Never invoked directly. |
| `/dp-compile:benchmark forward_backward` | Expected to report that no benchmark command is configured (`[commands] benchmark` is deliberately absent). |
| `/workflow-claude:smart-commit` | Commits each phase. The `dp-compile` pre-commit gate fires on every kernel path, so build and test run before the commit. |

**Enter through `/next-phase`, never a skill directly.** `DEFERRED.md` records that
`cython-translation`'s own staleness check lacks the recovered-graph carve-out that
`/next-phase` supplies through `phase-detection.md`.

**This is the reverse entrance.** `_forward_backward.py` exists and `[algorithms.forward_backward]`
is already registered (2026-09-14) with no formalization, so `/new-algorithm` has nothing to add and
`/next-phase` takes the phase-1-present branch of its phase-0 row. The graph branches at the root:
the formalization and every compiled phase derive from `_forward_backward.py`.
**An edit to that file makes the formalization and every compiled phase derived from it stale at
once**, so kernel changes between goals should be deliberate.

## Goals

- [x] Write the phase-0 `FORMALIZATION.md` for forward-backward, recovered from `_forward_backward.py` and the batched E-step, and decide the parallel decomposition (associative scan over time, or batch/state parallelism)
  > **Done:** Recovered in `c5ac642` and approved 2026-09-15: per-record recurrences and the batched E-step, evaluation order as contract, per-timestep `(record, state)` cells. It reports two findings left unfixed (the N = 0 docstring overclaim, ADR 0020's two-factor association argument) and names the `_training_log` files as oracle candidates for goal 2. TC-20 to TC-23 are specified but unimplemented.
  > **Q:** Which axis should the formalization's Parallel decomposition name for phases 3–4?
  > **A:** Each timestep's (record, state) cells, t serial, reductions serial and ascending (as the Viterbi batch). Whole records and an associative scan are recorded as not taken.
  > **Q:** Which kernel signature are phases 2–4 held to?
  > **A:** Both: per-record `_forward_backward` and batched `_e_step_batch`, as Viterbi.
  > **Q:** At N = 0 the kernel returns 0 bits where −log2(Σ init) is exact. What does the formalization specify?
  > **A:** 0 bits, recording the Lush trailing term and the docstring overclaim as a finding; no kernel or docstring change in this recovery.
  - [x] `/dp-compile:phase-check forward_backward` reports nominal phase 1, `formalization` absent, target `formalization` by the recovered entrance
  - [x] `/dp-compile:next-phase forward_backward` routes to `dp-compile:algorithm-recover`, which writes `docs/design/algorithms/forward_backward/FORMALIZATION.md` with `Derived from` naming `_forward_backward.py` and its SHA-256
  - [x] It names ADR 0020's evaluation order and no-fused-multiply-add rule as contract
  - [x] Its `Parallel decomposition` section states the chosen axis rather than "undetermined", since `cpu-parallelization` routes an open decomposition back to the document
  - [x] Reviewed and approved; `/dp-compile:phase-check forward_backward` reports it `fresh`, target `cython`
    > **Note:** Approved 2026-09-15. `phase-check` then read the recorded hash `54735d9b…` equal to the kernel's, all 12 sections filled, `python` 129/129 passed, target `cython`.
- [~] Implement phase 2 (Cython) and register it, held bit-exact to the numpy reference
  > **Q:** Should `[algorithms.forward_backward]` declare an `oracles` entry before phase 2?
  > **A:** Yes: the three tracked `_training_log` files, whose `data-dl` checks the forward pass and DL to 0.02 bits through quantised parameters (TC-01). The comment calling them "not a direct check" is corrected.
  - [x] `/dp-compile:next-phase forward_backward` passes the step-3 oracle gate. Decide whether `[algorithms.forward_backward]` should declare an `oracles` entry, since `tests/test_hmmlearn_oracle.py` exercises the kernel but the manifest names none
    > **Note:** Passed 2026-09-15. All three declared `_training_log` files are read by `test_the_data_description_length_is_the_originals_logged_value` (3/3 passed). `hmmlearn` was not declared: it is a library, and the manifest's `oracles` takes file paths.
  - [ ] Routes to `dp-compile:cython-translation`: `_forward_backward_cython.pyx` with a `# dp-compile: derived-from` header, listed in `meson.build`, rows in `_backends.py` in ADR 0021's `_Row` shape (not the skill's `Backend` template)
  - [ ] `/dp-compile:phase-check forward_backward` reports `cython` fresh, target `cpu_parallel`; committed through `/workflow-claude:smart-commit`, with the gate's build and test green
- [ ] Implement phase 3 (Numba CPU-parallel) on the decomposition goal 1 chose, held bit-exact
  - [ ] `/dp-compile:next-phase forward_backward` routes to `dp-compile:cpu-parallelization`: `_forward_backward_cpu_parallel.py` derived from the `.pyx`, `numba` optional under the `cpu-parallel` extra
  - [ ] `/dp-compile:phase-check forward_backward` reports `cpu_parallel` fresh, target `cuda`; committed through `/workflow-claude:smart-commit`
- [ ] Implement phase 4 (Numba CUDA), settling contraction behaviour, held bit-exact on a device
  - [ ] `/dp-compile:next-phase forward_backward` routes to `dp-compile:gpu-parallelization`: `_forward_backward_cuda.py` derived from the phase-3 kernel, run with `PFSMGRAPH_REQUIRE_BACKENDS=cuda` so a lost device fails rather than skips
  - [ ] `/dp-compile:phase-check forward_backward` reports complete across all four declared phases; committed through `/workflow-claude:smart-commit`
  - [ ] `/dp-compile:benchmark forward_backward` run for its report; if it confirms no benchmark command, record per-phase timings the way `viterbi_batch_speed.py` did, or say why not
