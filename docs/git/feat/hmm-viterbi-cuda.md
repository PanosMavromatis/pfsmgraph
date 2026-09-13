# feat/hmm-viterbi-cuda

**Created**: 2026-09-13
**Base**: main at 33851e3
**Status**: active

## Purpose

Implement Viterbi at ADR 0002 phase 4 (Numba CUDA), the last lifecycle phase of the
decode and the last kernel subgoal of revision `02-hmm-v0.1.0` before its `docs/api/`
and release subgoals. The branch adds `_viterbi_cuda.py`, registers it as the fourth
backend, and holds it bit-equivalent to phases 1–3 with explicit differential tests in
the labelled non-shared section of `test_viterbi.py` — not ADR 0003's parameterized
suite, which cannot be in force until `align`.

## Scope

- Confirm `/dp-compile:phase-check viterbi` reports nominal phase 3, nothing stale,
  target `cuda`; confirm the suite is green at 298 on this machine.
- Settle the environment so the `dp-compile` commit gate (`uv sync`) does not uninstall
  the backend it is testing, and pin `numba-cuda`'s lower bound in `hmm`'s `gpu` extra.
- Decide the phase-4 decomposition first — phase 3's states-within-one-timestep is the
  default, but a GPU-specific one is to be weighed before implementing (decided at goal 1,
  2026-09-13) — then write `_viterbi_cuda.py` against it. The first-wins tie-break and
  the min-plus objective stay contract either way.
- Register it in `_backends.py` with `optional_on` so an absent device skips loudly;
  add it to `meson.build`'s `install_sources`.
- Differential tests against phases 1–3, including the constructed uniform-model tie test.
- Measure launch overhead against phase 3's figures and record what it shows.

## Context

- Master plan: `docs/plan/TODO.md`, revision 02-hmm-v0.1.0, the phase-4 subgoal.
- Spec: `docs/design/algorithms/viterbi/FORMALIZATION.md` — the tie-break and the
  min-plus objective are contract; its `Parallel decomposition` section names the
  decomposition to use.
- ADR 0015 (arc-emission), ADR 0016 (four-phase lifecycle and its `Resolved` section),
  ADR 0017 (frozen parameters, so device copies are of read-only arrays).
- Prior phases: PR #21 (Cython), PR #22 (CPU-parallel), and the phase-3 branch plan at
  `docs/plan/02-hmm-v0.1.0/feat-hmm-viterbi-cpu-parallel/TODO.md`.
- `docs/plan/DEFERRED.md`: "Pin the `numba-cuda` lower bound", filed under a trigger that
  has already fired but actually waiting on this phase.

## Notes

- **Environment drift found at branch creation (2026-09-13).** This machine's `.venv`
  has `numba-cuda 0.30.4` and a CUDA-12 wheel stack installed outside the lock;
  `uv sync --locked --dry-run` would uninstall `numba-cuda` and move numpy 2.4.6 → 2.5.2.
  The `dp-compile` gate runs `[commands] build = "uv sync"`, so the first gated commit of
  `_viterbi_cuda.py` would remove its own backend. Phase 3 hit the same shape and put
  `numba` in the root `dev` group.
  *Resolved 2026-09-13 (goal 2):* the root `dev` group now carries
  `numba-cuda[cu13]>=0.30.4` and `numpy<2.5`, both Linux-marked. The cap is numba-cuda's
  undeclared one — 0.30.4 dies at import on numpy 2.5 (`np.row_stack`). A `cuda.jit`
  smoke kernel runs on the NVIDIA L4 under CUDA 13.0, and a plain `uv run pytest` is safe
  again.
- GPU present: driver 580.173.02, CUDA 13.0; no system `nvcc` (numba-cuda uses the
  pip-wheel toolkit). `numba.cuda.is_available()` is `True`.
- **Decomposition decided (goal 3, 2026-09-13):** phase 3's states-within-a-timestep,
  with the `-log2` arc costs computed **on the host** and uploaded once. Device `log2`
  (libdevice) differs from glibc's by 1 ulp in ~27% of inputs, so a plain transliteration
  would not be bit-exact; with the logs on the host the device does only `+` and `<`.
- Phase 3 measured a flat ~34 ms per call regardless of `S` — launch overhead per
  timestep. A per-timestep CUDA kernel is expected to hit that wall harder; the
  decomposition with the good speed story is the batch, which waits on revision 03.
