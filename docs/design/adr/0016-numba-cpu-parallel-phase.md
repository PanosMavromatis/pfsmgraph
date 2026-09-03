# 0016. Insert a Numba CPU-parallel phase between Cython and CUDA

- **Status:** Accepted — amends [ADR 0002](0002-three-phase-algorithm-lifecycle.md) by
  inserting a phase rather than reversing it. ADR 0002's own three phases, now four,
  are otherwise unchanged.
- **Date:** 2026-09-03
- **Source:** none in the PRD — postdates it, like 0012–0015.

## Context

[ADR 0002](0002-three-phase-algorithm-lifecycle.md) defines phase 3 as "Numba CUDA,
anti-diagonal wavefront — for scale," and reaching it at all requires CUDA hardware.
That hardware is not reliably available in this project's actual working environment: no
local CUDA-capable GPU, and the fallback — GCP VMs — is not readily available on demand.
This holds phase 3 back for however long provisioning takes, and it does so at exactly
the point ADR 0002 itself identifies as most expensive to get wrong: "Debugging a wrong
recurrence inside a CUDA kernel is dramatically more expensive than debugging it in
Python." The whole point of the ordered lifecycle is to front-load correctness checking
before the expensive-to-debug phase — and a phase that cannot be reached at all defeats
that purpose regardless of how correct the recurrence is.

But phase 3, as scoped, bundles two separable concerns:

1. **Is the anti-diagonal wavefront decomposition itself correct?** Do all cells on one
   diagonal really depend only on the two preceding diagonals, with no cross-cell
   interference when computed concurrently? This is a dependency-structure question,
   checked against the phase-1 oracle.
2. **Is the hardware kernel correct and fast on actual CUDA?** Memory coalescing, warp
   occupancy, `cuda.jit` semantics, real GPU throughput. This is a hardware-execution
   question.

Only the second genuinely requires CUDA hardware. The first requires only *some* real
concurrent execution — and multi-core CPU parallelism, present on every machine including
a laptop, is sufficient for it. Numba itself already ships a mature CPU-parallel target,
`@njit(parallel=True)` with `prange`, using the identical library this family already
commits to for the CUDA phase — adopting it introduces no new toolchain, only a new
import target reached earlier.

## Decision

**A new phase is inserted between Cython and CUDA. The lifecycle is now four phases,
strictly ordered:**

1. Pure Python — for correctness. *(unchanged)*
2. Cython — for performance. *(unchanged)*
3. **Numba, CPU-parallel, anti-diagonal — for parallel correctness.**
   `@njit(parallel=True)` with `prange` over the anti-diagonal, on the CPU. Same
   decomposition phase 4 uses, same oracle-checking discipline as every other phase:
   checked against phase 1, not assumed. This is where the anti-diagonal wavefront
   decomposition itself is proven — the dependency structure, the boundary diagonals,
   the two-preceding-diagonal read — before any hardware-specific kernel work begins.
4. Numba CUDA, anti-diagonal wavefront — for scale. *(renumbered from phase 3; content
   unchanged — still gated on real CUDA hardware, but now inherits an already-validated
   decomposition from phase 3, so what remains to debug at this phase is
   hardware-kernel-specific, not algorithmic.)*

No phase is skipped, and no phase begins before the previous one is correct — phase 3's
arrival does not relax that rule for phase 4; it narrows what phase 4 has left to find.

**Plain `numba` (the CPU target) becomes a hard runtime dependency** of any package that
reaches phase 3 (`align`; `hmm`'s Baum-Welch/Viterbi core) — not an optional extra, unlike
`numba-cuda`, which stays behind the `[gpu]` extra [ADR 0004](0004-gpu-backends-and-optional-dependency-strategy.md)
already governs. This is a real, if modest, difference from Cython: Cython is a
*build-time* dependency (compiles to a `.so`, absent from the installed runtime), while
Numba JIT-compiles at call time, so a package shipping a phase-3 implementation needs
`numba` importable at runtime — and every phase is shipped (ADR 0002's own rule), so there
is no scenario where it is optional. This does not touch ADR 0004's "GPU means two
unrelated things" boundary at all: plain `numba` (CPU) is not a GPU dependency, so it
never enters the `[gpu]`-extra question `numba-cuda` and `torch` already divide between
them.

## Consequences

### Positive

- **Portable, hardware-independent, permanent.** Phase 3 needs no GPU, no CUDA toolkit,
  no cloud VM — it runs on any machine with more than one core, including CI runners and
  a GCP VM that itself has no GPU attached. Once implemented, it is never "absent
  hardware" in [ADR 0003](0003-one-parameterized-test-suite-per-algorithm.md)'s sense;
  there is no environment-degradation skip path for it to fall into, only
  implemented-vs-not.
- **Front-loads the highest-payoff debugging exactly where ADR 0002 says it matters
  most.** A wrong anti-diagonal decomposition is caught against a Python oracle on a
  laptop, not half-suspected inside a `cuda.jit` kernel that can only run once a VM
  materializes.
- **Narrows phase 4's remaining surface.** By the time a CUDA kernel is written, the
  anti-diagonal decomposition is already proven; what phase 4 debugs is
  hardware-kernel-specific, not "is this the right algorithm."
- **Rehearses the eventual kernel-writing idiom.** `prange`'s per-iteration-cell function
  is a closer shape to a `cuda.jit` kernel (a function indexed by thread/cell position)
  than a vectorized NumPy or `torch` formulation would be — see Alternatives.
- **ADR 0003's parameterized suite gains a fourth, always-available backend to
  enumerate**, at zero immediate cost — the registry already handles "a lifecycle phase
  not yet reached contributes no parameter at all," so nothing changes until a package
  actually reaches the new phase 3.

### Negative / costs

- **A third implementation to keep in agreement, before the fourth even lands.** ADR 0002
  already names "three implementations of every kernel must be kept in agreement,
  forever" as its dominant cost; this makes it four, and a traceback fix is now a
  four-place change instead of three.
- **A new hard runtime dependency.** `align`/`hmm`'s base install grows by `numba` once
  they reach phase 3 — lighter than `torch`, heavier than "no new import," and unlike
  Cython's build-only footprint, this one ships to every user regardless of whether they
  ever touch phase 4/CUDA.
- **Does not de-risk phase 4 itself.** CPU-parallel correctness says nothing about
  `cuda.jit`-specific hardware behavior — coalescing, warp divergence, real throughput.
  Phase 4 is still real, still gated on CUDA hardware; this ADR narrows what's left to
  find there, not the gate itself.
- **Time-to-fast grows longer still.** ADR 0002 already costs "phases 1 and 2 must be
  finished before phase 3 begins"; now phase 3 must also finish before phase 4 begins,
  even on a machine where CUDA hardware happens to be sitting idle and available today.

## Alternatives considered

- **PyTorch MPS (Apple Metal) as the new phase.** Rejected. It would pull `torch` into
  the DP-package side of ADR 0004's "GPU means two unrelated things" boundary for a
  reason that ADR does not argue for — ADR 0004 permits `torch` in `hmm` only for the
  Baum-Welch autograd-equivalence rationale revision 03 is scoped to, not as a general
  acceleration path for DP kernels. It also does not rehearse CUDA kernel-writing idioms
  the way `prange` does: an MPS implementation is naturally written as batched tensor
  ops, closer in shape to NumPy vectorization than to a `cuda.jit` kernel. And the
  real-world speed gain from MPS for workloads of this kind has historically been
  marginal, on hardware whose relevance keeps shrinking as newer Apple silicon
  generations ship — not a foundation worth building a permanent lifecycle phase on.
- **A vectorized NumPy diagonal-by-diagonal check, with no true parallel execution.**
  Rejected as insufficient on its own: it would prove only that iterating diagonally in
  the correct order is possible, not that concurrent, order-independent computation of a
  diagonal's cells is race-free — the actual claim the wavefront decomposition makes.
  (ADR 0002 already rejected NumPy vectorization as a *replacement* for Cython, on the
  related grounds that straightforward vectorization cannot express DP's sequential cell
  dependency; this alternative would repeat that insufficiency one layer up, as a
  correctness check rather than a shipped phase.)
- **Rent a GCP VM whenever phase 3(→4) validation is needed, and accept the wait.**
  Rejected: it does not solve the problem, it re-describes it. The reason a phase's
  correctness must be checked before its performance is trusted is defeated if the check
  itself is gated behind the same scarce resource as the thing being checked.
- **Leave CUDA as the sole GPU-scale phase and accept the local block.** Rejected:
  nothing about the anti-diagonal decomposition's correctness actually requires GPU
  hardware, so accepting the block trades away a free de-risking step for no principled
  reason — the constraint was assumed, not load-bearing, once separated from the
  hardware-kernel concern it was bundled with.

## Evidence

Reported, not independently benchmarked on this project's kernels: Apple Silicon MPS
speed gains for workloads structurally similar to anti-diagonal DP wavefronts have been
observed as marginal in prior, unrelated work — cited here as the reasoning behind
rejecting the MPS alternative above, not as a measurement this project has made. Numba's
CPU-parallel target (`prange`) is documented, stable Numba functionality already implied
by this family's commitment to Numba for the CUDA phase; this ADR asks nothing new of the
library, only earlier use of one of its targets.

## Open

- **Whether the anti-diagonal decomposition applies to `hmm`'s Viterbi at all.**
  `docs/plan/TODO.md`'s revision 02 already carries an unresolved finding that an HMM
  recurrence is 1-D over time with dense N×N state coupling, and may have no
  anti-diagonals — a batch or associative-scan decomposition instead. This ADR does not
  resolve that; it says only that *if* a kernel uses the anti-diagonal decomposition,
  phase 3 is where it is first validated under real concurrency. For `align`, phase 3
  unambiguously applies. For `hmm`, phase 3's actual content is contingent on revision
  02's still-open finding, at the same point ADR 0002's CUDA phase already deferred it to.
- **The phase-3 module naming convention.** This ADR proposes `_cpu_parallel.py` (glob
  `_cpu_parallel*.py`), mirroring the existing `_python.py` / `_cython.pyx` / `_cuda.py`
  per-phase naming seen in `.scratch/align-poc/tokalign`, but it is not yet exercised
  against a real kernel and may need revisiting once the first phase-3 implementation
  lands.
