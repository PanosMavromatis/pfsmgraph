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

- [x] Confirm the plugin's view of the algorithm matches this branch's premise
  > **Q:** Has `FORMALIZATION.md` been reviewed and approved, and does it stand as the
  > contract phase 4 is written against — including its `Parallel decomposition` section
  > (states within one timestep, ascending serial reduction over `i`)?
  > **A:** Approved, but revisit the decomposition. The contract holds; phase 4 should weigh
  > a GPU-specific decomposition before implementing rather than transliterate phase 3's
  > by default.
  > **Done:** premise confirmed on this machine (2026-09-13). Nominal phase **3**;
  > `formalization`, `python`, `cython` and `cpu_parallel` fresh; `cuda` absent; target
  > **write `cuda`**. `FORMALIZATION.md` and `_viterbi_cython.pyx` both record `_viterbi.py`
  > at `sha256:48666ca8…`, `_viterbi_cpu_parallel.py` records `_viterbi_cython.pyx` at
  > `sha256:79da7c8d…`, and both recomputed hashes match — the graph still branches at the
  > root. All three declared `.vpath.xls` oracles are exercised. The formalization's entrance
  > is **recovered**, its mandatory sections all filled.
  > **Note:** `_viterbi.py` still carries no provenance header and the mtime fallback was
  > again deliberately not applied — it is the root, named by two headers.
  > **Note:** **`uv run` is unsafe on this machine until goal 2 lands, not only the commit
  > gate.** `uv run` syncs before executing, so a bare `uv run pytest` would uninstall the
  > out-of-lock `numba-cuda` exactly as `[commands] build = "uv sync"` would. The suite was
  > run as `uv run --no-sync pytest`; `numba-cuda 0.30.4` and numpy 2.4.6 were confirmed
  > still installed afterwards. The same applies to `[commands] test`, so the gate's *test*
  > step would sync too, not only its build step.
  - [x] `/dp-compile:phase-check viterbi` reports nominal phase 3, nothing stale, target `cuda`
  - [x] Suite green at 298 on this machine, header `backends: python ✓ · cython ✓ · cpu_parallel ✓`
    > **Note:** 298 passed in 5.84 s under `--no-sync`, no backend withheld. The header
    > names three backends with a GPU present, correctly — a phase not yet reached
    > contributes no row.
- [ ] Settle the GPU environment before the first gated commit
  - [ ] Decide how `numba-cuda` survives `uv sync` — the gate's build command — given it is currently installed outside the lock and a plain sync uninstalls it
  - [ ] Pin the `numba-cuda` lower bound in `pfsmgraph-hmm`'s `gpu` extra and close the `DEFERRED.md` entry; check what numpy ceiling it imposes
- [ ] Implement `_viterbi_cuda.py` at ADR 0002 phase 4
  - [ ] Decide the phase-4 decomposition before writing the kernel: keep states-within-a-timestep, or choose a GPU-specific one (e.g. a device-side time loop, or batch — which `viterbi(params, record)` cannot express until revision 03). Tie-break and min-plus stay contract either way.
  - [ ] Implement whichever decomposition that decision names under `cuda.jit`, keeping the kernel purely numeric
  - [ ] Register the backend in `_backends.py` with `optional_on`, and add the module to `meson.build`'s `install_sources`
- [ ] Hold it equivalent to phases 1–3
  - [ ] Differential tests in `test_viterbi.py`'s labelled non-shared section, including the constructed uniform-model tie test
  - [ ] Verify an absent device skips loudly and `PFSMGRAPH_REQUIRE_BACKENDS` escalates it
- [ ] Measure and record
  - [ ] Launch overhead against phase 3's flat ~34 ms; update `core.md`, `FORMALIZATION.md` and ADR 0016 as the result warrants
