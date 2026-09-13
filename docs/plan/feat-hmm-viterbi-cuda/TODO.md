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
- [x] Settle the GPU environment before the first gated commit
  > **Q:** How should `numba-cuda` survive `uv sync` / `uv run`, and so the `dp-compile`
  > gate — root `dev` group (Linux-only), `--extra gpu` in the manifest commands, or kept
  > out of the lock under `--no-sync`?
  > **A:** Root `dev` group, Linux-only — mirroring phase 3's `numba`. Consumers still opt
  > in through the `gpu` extra.
  > **Q:** Which CUDA major should the toolkit extra target — `cu13` or `cu12`?
  > **A:** `cu13`: driver 580 supports CUDA 13.0 and torch in the lock already brings CUDA
  > 13 runtime wheels, so one CUDA major per venv. Replaces the hand-installed `cu12` stack.
  > **Note:** the hand-installed stack was `numba-cuda[cu12]` via `uv pip` (its `INSTALLER`
  > reads `uv`). Bare `numba-cuda` — what the `gpu` extra declares — ships **no toolkit**:
  > NVRTC, nvJitLink and NVVM arrive only through its `cu12`/`cu13` extras. The lock's
  > `cuda-toolkit 13.0.3` is **torch's** (the dataseq interop test), requested without
  > `nvvm`, so a locked `--extra gpu` sync would plausibly leave numba-cuda unable to
  > compile a kernel — not yet measured.
  > **Note:** `numba-cuda` publishes **no macOS wheels** (manylinux and `win_amd64` only),
  > so an unmarked dev-group entry would break `uv sync` on a Mac; hence the
  > `sys_platform == 'linux'` marker. It adds **no numpy ceiling** of its own — `cuda-core`
  > requires unbounded `numpy`, and the binding cap stays numba 0.67's `<2.6`.
  - [x] Decide how `numba-cuda` survives `uv sync` — the gate's build command — given it is currently installed outside the lock and a plain sync uninstalls it
    > **Q:** numba-cuda 0.30.4 — the latest release — dies at import on numpy 2.5
    > (`np.row_stack` removed) and its metadata does not say so. Where should the
    > `numpy<2.5` cap live?
    > **A:** In `hmm`'s `gpu` extra for consumers (subgoal below), plus a Linux-marked
    > `numpy<2.5` in the root `dev` group so the workspace resolves where numba-cuda is
    > installed. A uv-only constraint was declined: it reaches no published metadata.
    > **Done:** root `dev` group gained `numba-cuda[cu13]>=0.30.4; sys_platform == 'linux'`
    > and `numpy<2.5; sys_platform == 'linux'`, each commented. `uv lock` added
    > `nvidia-nvvm` and `nvidia-cuda-cccl` 13.0.x and forked numpy by platform: 2.4.6 on
    > Linux, 2.5.2 elsewhere at Python ≥ 3.12. `uv sync` replaced the hand-installed cu12
    > stack, and `uv sync --locked --dry-run` now reports no changes — so the gate's build
    > and a bare `uv run` keep the backend. A trivial `cuda.jit` kernel compiled and ran on
    > an **NVIDIA L4** (compute capability 8.9, CUDA runtime 13.0) and matched the host
    > bit-for-bit with `inf` propagated; the suite is green at 298 under plain `uv run
    > pytest`.
    > **Note:** **a dry-run sync looking clean was not evidence the stack worked.** The
    > first locked sync (numpy 2.5.2) installed without complaint and the lock resolved,
    > yet the smoke test died with `AttributeError: module 'numpy' has no attribute
    > 'row_stack'` from `numba_cuda/numba/cuda/np/arrayobj.py:6860`, reached through the
    > `cuda.jit` import chain. The hand-installed environment only ever worked because it
    > happened to hold numpy 2.4.6. So resolution success says nothing about importability
    > here — compile and run a kernel, as the smoke test does, and expect `_backends.py`'s
    > import probe of `_viterbi_cuda` to be what catches this in the suite.
    > **Note:** `cuda-core` resolved **down**, 1.2.0 → 1.1.1, under the `cu13` extra;
    > harmless, recorded because a downgrade in a lock diff otherwise reads as a mistake.
  - [x] Pin the `numba-cuda` lower bound in `pfsmgraph-hmm`'s `gpu` extra and close the `DEFERRED.md` entry; check what numpy ceiling it imposes
    > **Q:** What should `pfsmgraph-hmm`'s published `gpu` extra declare — bare
    > `numba-cuda`, `numba-cuda[cu13]`, or separate `gpu-cu12`/`gpu-cu13` extras?
    > **A:** Bare `numba-cuda>=0.30.4` plus `numpy<2.5`. numba-cuda itself leaves the
    > toolkit to the installer, and naming a CUDA major would decide driver compatibility
    > for every consumer; the dev group's `cu13` stays a repository choice.
    > **Q:** The `DEFERRED.md` entry also pins `align`'s `gpu` extra — pin it too, or narrow
    > the entry?
    > **A:** Narrow it to `align`, whose wavefront backend is its own phase 4 and has not
    > landed; fix `codex.md`'s pointer to it.
    > **Done:** `gpu = ["numba-cuda>=0.30.4", "numpy<2.5"]`, commented with why it is bare,
    > why the floor, and whose ceiling the cap is. `DEFERRED.md`'s entry narrowed to
    > `align` rather than closed, recording that `hmm`'s phase 4 is not a wavefront so the
    > entry's own condition now names `align` alone; `codex.md:190` updated to match.
    > Resolving the extra for Linux gives `numba-cuda==0.30.4`, `numpy==2.4.6`; the suite
    > stays green at 298.
    > **Note:** **the extra's cap binds the whole workspace lock, on every platform.** uv
    > resolves every member's extras into one universal lock, so `numpy<2.5` in `hmm`'s
    > `gpu` extra collapsed the numpy fork the previous commit created: `uv lock` reported
    > `numpy v2.2.6, v2.4.6, v2.5.2 -> v2.2.6, v2.4.6`, and macOS development now resolves
    > 2.4.6 too. Consumers are unaffected — the cap reaches only whoever opts into `gpu` —
    > and the Linux-marked dev-group `numpy<2.5` is now redundant in the lock without being
    > wrong. It stays, because it states the repository's own need and would matter again
    > if the extra's cap were dropped first. This is the lock-side face of `core.md`'s
    > remark that `uv.lock` "records one resolution rather than the space of them".
- [~] Implement `_viterbi_cuda.py` at ADR 0002 phase 4
  - [x] Decide the phase-4 decomposition before writing the kernel: keep states-within-a-timestep, or choose a GPU-specific one (e.g. a device-side time loop, or batch — which `viterbi(params, record)` cannot express until revision 03). Tie-break and min-plus stay contract either way.
    > **Q:** Which phase-4 decomposition should `_viterbi_cuda.py` implement — phase 3's
    > with the logarithms computed on the host, a plain transliteration computing them on
    > the device, or a min-plus associative scan?
    > **A:** Phase 3's decomposition, logarithms on the host. A host loop over `t` launches
    > one kernel per timestep over `j`, with the reduction over `i` serial and ascending
    > inside each thread. The host computes the `-log2` arc-cost table once with numpy and
    > uploads it, and the device performs only `+` and `<` — so the kernel is bit-exact with
    > phases 1–3 by construction rather than by tolerance. The table's shape is settled
    > while implementing.
    > **Note:** **device `log2` is not bit-identical to the host's, and this is what made
    > the decomposition a correctness question rather than a speed one.** Measured
    > 2026-09-13 on the NVIDIA L4 (numba-cuda 0.30.4, CUDA 13.0): `-math.log2(p)` in a
    > `cuda.jit` kernel differs bitwise from `-np.log2(p)` for **1,087,159 of 4,000,008**
    > probability-shaped inputs, each by exactly **1 ulp**; `-log2(a*b)` differs for
    > 1,061,459 of 4,000,000. The device call lowers to libdevice, while phases 1–3 share
    > glibc/LLVM's `log2` — which is the *only* reason those three are bit-identical. A
    > transliteration would therefore make candidate costs differ by an ulp, let a near-tie
    > flip an `argmin`, and turn the tie-break contract platform-dependent. The uniform
    > model's exact ties would still pass, since identical inputs round identically on one
    > device — so TC-06 would not have caught it. Only `+` and `<` are exact IEEE
    > operations; the transcendental function is where agreement ends.
    > **Note:** **per-timestep launch cost, data already on the device**, same run: ~56 µs
    > at `S = 5`, 55 µs at `S = 16`, 62 µs at `S = 64`, 269 µs at `S = 160` — flat until
    > the work outgrows the launch. That projects an `N = 400` decode at ~22 ms for
    > `S ≤ 64` and ~108 ms at `S = 160`, against phase 3's measured flat ~34 ms. It is the
    > phase-3 diagnosis on a GPU: `t` is strictly sequential, so every exact single-record
    > decomposition pays one launch per timestep, and only revision 03's batch removes it.
    > A projection from an isolated step kernel, not yet a measurement of `_viterbi_cuda`;
    > goal 5 measures the real thing.
    > **Note:** the formalization's `Parallel decomposition` section does not yet say where
    > the logarithm is evaluated, because until now every backend evaluated it with the
    > same function. That is contract now, and it lands in `FORMALIZATION.md` with goal 5.
    > The scripts behind both notes were scratchpad-only (`cuda_probe.py`), not tracked.
  - [x] Implement whichever decomposition that decision names under `cuda.jit`, keeping the kernel purely numeric
    > **Q:** Write `_viterbi_cuda.py` with an `ImportError` when no device is present, an
    > `(S, S, U)` host log table over the symbols present, and a local filter for the
    > low-occupancy warning — and list it in `meson.build` in the same step?
    > **A:** Yes, both. The alternative probe — a device check added to
    > `_backends.detect()` — was declined, since raising at import leaves the registry
    > unchanged.
    > **Done:** phase 4 `cuda` — `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_viterbi_cuda.py`,
    > derived-from `_viterbi_cpu_parallel.py` `sha256:5f45abb4…`, entered through
    > `/dp-compile:next-phase viterbi` → `dp-compile:gpu-parallelization`. A host loop over
    > `t` launches `_step` once per timestep over `j`; `arc_bits` is `(S, S, U)` built by
    > `bits(trans[:, :, None] * out[:, :, present])` on the host, uploaded once with the
    > re-indexed record, so the device does only `+` and `<`. `N = 0` never touches the
    > device. Listed in `meson.build`'s `install_sources` in the same change. A scratchpad
    > comparison on the L4, run with warnings as errors, agreed **exactly** — `states` and
    > `total_bits` — with phases 1 and 3 on 300 random models (`S` 1–11, some arcs zeroed),
    > multi-block `S` = 63/64/65/130, `N = 0`, `S = 1`, and the uniform tie (state 0
    > throughout); `UNK` returned `+inf` without raising, and read-only inputs were accepted.
    > Suite green at 298, `test_meson_sources.py` 16/16. Not yet registered.
    > **Note:** **numba-cuda has its own `NumbaPerformanceWarning`, unrelated to numba's.**
    > It raises `numba.cuda.core.errors.NumbaPerformanceWarning`, which is not a subclass
    > of `numba.core.errors.NumbaPerformanceWarning`, so a filter on the latter matches
    > nothing and fails silently — which is exactly what happened in the scratchpad launch
    > probe, whose "Grid size 1" warnings printed straight through its filter. The warning
    > fires once per launch *configuration*, so the kernel builds `_step[blocks, threads]`
    > once per call, inside the filter, rather than once per timestep.
  - [ ] Register the backend in `_backends.py` with `optional_on`, and add the module to `meson.build`'s `install_sources`
- [ ] Hold it equivalent to phases 1–3
  - [ ] Differential tests in `test_viterbi.py`'s labelled non-shared section, including the constructed uniform-model tie test
  - [ ] Verify an absent device skips loudly and `PFSMGRAPH_REQUIRE_BACKENDS` escalates it
- [ ] Measure and record
  - [ ] Launch overhead against phase 3's flat ~34 ms; update `core.md`, `FORMALIZATION.md` and ADR 0016 as the result warrants
