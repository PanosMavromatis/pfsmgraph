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
- [x] Implement phase 2 (Cython) and register it, held bit-exact to the numpy reference
  > **Done:** Phase 2 landed in `c637e9b`: `_forward_backward_cython.pyx` with both signatures, built with `-ffp-contract=off`, registered under `forward_backward` and `baum_welch` (so `baum_welch(backend="cython")` is public), and held byte for byte by `test_forward_backward_backends.py` (TC-20 to TC-23). The Lush `_training_log` files are now declared oracles (`c07ba5e`). Suite green at 1178.
  > **Q:** Should `[algorithms.forward_backward]` declare an `oracles` entry before phase 2?
  > **A:** Yes: the three tracked `_training_log` files, whose `data-dl` checks the forward pass and DL to 0.02 bits through quantised parameters (TC-01). The comment calling them "not a direct check" is corrected.
  - [x] `/dp-compile:next-phase forward_backward` passes the step-3 oracle gate. Decide whether `[algorithms.forward_backward]` should declare an `oracles` entry, since `tests/test_hmmlearn_oracle.py` exercises the kernel but the manifest names none
    > **Note:** Passed 2026-09-15. All three declared `_training_log` files are read by `test_the_data_description_length_is_the_originals_logged_value` (3/3 passed). `hmmlearn` was not declared: it is a library, and the manifest's `oracles` takes file paths.
  > **Q:** Build the new extension with `-ffp-contract=off` now (ADR 0020 Open), rather than relying on baseline x86-64 having no FMA?
  > **A:** Yes, via `cc.get_supported_arguments`, on `_forward_backward_cython` only; TC-21 still gets a constructed test.
  > **Q:** Where do phase 2's equivalence tests go?
  > **A:** A new bit-exact module, `tests/test_forward_backward_backends.py` (TC-20 to TC-23), plus `("cython", None)` in `test_baum_welch_backends.py`'s `KERNEL_TARGETS`.
  - [x] Routes to `dp-compile:cython-translation`: `_forward_backward_cython.pyx` with a `# dp-compile: derived-from` header, listed in `meson.build`, rows in `_backends.py` in ADR 0021's `_Row` shape (not the skill's `Backend` template)
    > **Done:** phase 2 `cython` — `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_forward_backward_cython.pyx`, derived-from `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_forward_backward.py` `sha256:54735d9bfb7261d95506a64bc64ec67cd4b52daa4b17eca393829825fa9343d2`, suite green at 1178.
    > **Note:** Mutation-tested 2026-09-15. A `-march=native -ffp-contract=fast` build failed 36 bit-exact tests (TC-21 among them) and a reversed forward sum failed 29, while all 64 `cython` cases in the tolerance-based `test_baum_welch_backends.py` passed both: only `tobytes()` comparison sees ADR 0020's order. `cdivision` removed an unreachable zero-check that reacquired the GIL; 0 of 45 kernel lines touch the C-API.
  - [x] `/dp-compile:phase-check forward_backward` reports `cython` fresh, target `cpu_parallel`; committed through `/workflow-claude:smart-commit`, with the gate's build and test green
    > **Note:** Checked 2026-09-15 after `c637e9b`: the formalization and the `.pyx` both record the kernel's current hash `54735d9b…`, `python` and `cython` passed 235/235 with all three oracles exercised, target `cpu_parallel`. The `dp-compile` gate armed on the `.pyx` in that commit (confirmed with the hook's own `kernel_paths`), so its build and test ran and passed.
- [x] Implement phase 3 (Numba CPU-parallel) on the decomposition goal 1 chose, held bit-exact
  > **Done:** `_forward_backward_cpu_parallel.py` with both signatures on the per-timestep `(record, state)` cells goal 1 chose, every reduction serial, registered under the `cpu-parallel` extra so `baum_welch(backend="cpu_parallel")` is public. Held byte for byte to the reference at 1 thread and at 4, suite green at 1297.
  - [x] `/dp-compile:next-phase forward_backward` routes to `dp-compile:cpu-parallelization`: `_forward_backward_cpu_parallel.py` derived from the `.pyx`, `numba` optional under the `cpu-parallel` extra
    > **Done:** phase 3 `cpu_parallel` — `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_forward_backward_cpu_parallel.py`, derived-from `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_forward_backward_cython.pyx` `sha256:a3dc5d01a67ce825dddcae92bc297999131531df19620713c1bd823126470feb`, suite green at 1297.
    > **Note:** Registered with `needs="numba"`, `extra="cpu-parallel"` per the manifest's `[dependencies]` row, overriding the skill's hard-dependency default, so `baum_welch(backend="cpu_parallel")` is public. Mutation-tested 2026-09-15: turning the scale factor's sum over `j` into a `prange` reduction failed 17 bit-exact tests at 4 threads (the thread-invariance test among them) and **none at 1 thread**, where numba's single chunk folds from `0.0` to the same bits; the 64 tolerance-based `cpu_parallel` cases passed it too. Only the thread-count comparison sees a reassociated reduction.
  - [x] `/dp-compile:phase-check forward_backward` reports `cpu_parallel` fresh, target `cuda`; committed through `/workflow-claude:smart-commit`
    > **Note:** Checked 2026-09-15: the chain `_forward_backward.py` → `.pyx` (`54735d9b…`) → `_forward_backward_cpu_parallel.py` (`a3dc5d01…`) matches at every link, and the formalization's `54735d9b…` matches the kernel. `python`, `cython` and `cpu_parallel` passed 290/290 with all three oracles exercised; target `cuda`. Committed through `/workflow-claude:smart-commit` together with the phase.
- [~] Implement phase 4 (Numba CUDA), settling contraction behaviour, held bit-exact on a device
  > **Q:** How should the CUDA kernel keep NVVM from fusing multiply-add, which it does by default on the L4 (24,081 of 200,000 `c + a*b` results differ from the host's unfused value, still at `opt=False`), when `cuda.jit` exposes no `fma=0`?
  > **A:** Separate launches: every product is written to device memory by one launch and summed by a later one, which no NVVM can fuse. Storing and reloading in one kernel (measured unfused, but optimiser-dependent) and forcing `fma=0` through numba-cuda internals were declined.
  > **Q:** `baum_welch`'s `device=` check told any non-torch backend it "runs on the CPU only", false for `cuda`. What does `backend="cuda"` accept?
  > **A:** Only `device=None`; anything else raises a `ValueError` saying it runs on numba-cuda's current device and `device=` applies only to `torch`.
  - [x] `/dp-compile:next-phase forward_backward` routes to `dp-compile:gpu-parallelization`: `_forward_backward_cuda.py` derived from the phase-3 kernel, run with `PFSMGRAPH_REQUIRE_BACKENDS=cuda` so a lost device fails rather than skips
    > **Done:** phase 4 `cuda` — `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_forward_backward_cuda.py`, derived-from `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_forward_backward_cpu_parallel.py` `sha256:7c719a7326b62c8c524ad3c52afd6447c432db8f67140314176d1dc41b7cfe1d`, suite green at 1418 with `PFSMGRAPH_REQUIRE_BACKENDS=cuda` on an NVIDIA L4, no skips.
    > **Note:** Settles ADR 0020's open contraction question, measured on the L4 with numba-cuda 0.30.4: NVVM **does** fuse multiply-add by default, even without `fastmath` and at `opt=False`, returning the exactly rounded `fma` in 20,000 of 20,000 checked cases. The kernel therefore never multiplies and adds in one launch. Mutation-tested: putting the forward products and their sum back into one kernel failed 38 of the 55 `cuda` bit-exact tests (TC-21 among them), while all 67 tolerance-based `cuda` cases in `test_baum_welch_backends.py` passed. Every probe ran on the device; nothing here rests on the simulator. Moving the item to ADR 0020's `Resolved` section is a separate change.
  - [x] `/dp-compile:phase-check forward_backward` reports complete across all four declared phases; committed through `/workflow-claude:smart-commit`
    > **Note:** Checked 2026-09-15: every link of the provenance graph matches (`54735d9b…` for the formalization and the `.pyx`, `a3dc5d01…` for phase 3, `7c719a73…` for phase 4), no target remains, and all four backends passed 345/345 on the L4 with `PFSMGRAPH_REQUIRE_BACKENDS=cuda`, all three oracles exercised. Committed through `/workflow-claude:smart-commit` together with the phase.
  - [ ] `/dp-compile:benchmark forward_backward` run for its report; if it confirms no benchmark command, record per-phase timings the way `viterbi_batch_speed.py` did, or say why not
