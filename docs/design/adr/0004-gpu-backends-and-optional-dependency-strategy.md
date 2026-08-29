# 0004. GPU backends are two unrelated things; heavy dependencies stay optional

- **Status:** Accepted
- **Date:** 2025–2026 (proof-of-concept and `dl` work); formalized 2026-08-29
- **Source:** PRD §1.2, §6 — inherited, restated for the five-package family

## Context

Two packages in this family want a GPU, for entirely unrelated reasons.

- `align` and the Baum-Welch core of `hmm` want one to run the anti-diagonal wavefront
  phase of [ADR 0002](0002-three-phase-algorithm-lifecycle.md). Their need is for
  **custom DP kernels** over integer and float arrays.
- `dl` wants one to train PyTorch models. Its need is for **a tensor framework** with
  autograd.

These share a piece of hardware and nothing else. The software stacks are different
(`numba-cuda` versus `torch`), the install footprints differ by an order of magnitude,
the version and CUDA-toolkit constraints are independent, and a user who wants one
frequently does not want the other: someone aligning sequences on a GPU has no use for
several gigabytes of PyTorch, and someone training a transformer has no use for a CUDA
DP kernel compiler.

The tempting simplification — a single family-wide `[gpu]` extra meaning "make it fast"
— destroys exactly this distinction.

## Decision

**"GPU" is not a single concept in this family, and is never expressed as one.**

- The DP packages (`align`, `hmm`) use **`numba-cuda`**, exposed as their own optional
  extra.
- `dl` uses **`torch`**, which is a hard requirement of that package rather than an
  accelerator option — `dl` is a PyTorch package whether or not a GPU is present.
- **These are not unified into a single `[gpu]` extra**, dependency group, or install
  story, at any level of the family.

More generally: heavy dependencies are optional extras, and the base install of every
package stays lean. A package installs what it needs to be correct; acceleration is
opt-in.

## Consequences

### Positive

- **Users install what they actually need.** `pip install pfsmgraph-align[gpu]` does not
  drag in PyTorch, and installing `pfsmgraph-dl` does not pull a CUDA kernel compiler.
- **The two stacks version independently.** A PyTorch upgrade cannot break the alignment
  wavefront path and vice versa, which matters because both are sensitive to CUDA
  toolkit versions and would otherwise have to be co-satisfied in the workspace's single
  shared lockfile ([ADR 0006](0006-single-repository-as-a-uv-workspace.md)).
- **The lean base install is what makes graceful degradation real.** Since
  [ADR 0002](0002-three-phase-algorithm-lifecycle.md) keeps every phase shipped, a
  no-extras install is a fully working library, not a stub.
- **It keeps the meaning of "GPU support" honest in documentation.** The two claims are
  separately true and separately verifiable.

### Negative / costs

- **Two extras to explain rather than one.** A user who wants everything accelerated
  must know to ask twice, and the docs must say so explicitly rather than relying on an
  obvious single knob.
- **Neither extra can be verified by CI without appropriate hardware**, so availability
  is an environment property the test suite must handle
  ([ADR 0003](0003-one-parameterized-test-suite-per-algorithm.md)) rather than assume.
- **`torch` being a hard dependency makes `pfsmgraph-dl` a heavy install** with no lean
  variant. This is accepted: a PyTorch model package without PyTorch is not a package.

## Alternatives considered

- **A single family-wide `[gpu]` extra.** Rejected — the decision this ADR exists to
  record. It couples two unrelated stacks, and would make any `[gpu]` install pay for
  both.
- **A `pfsmgraph[all]` umbrella extra on the bare namespace placeholder.** Not adopted:
  the bare `pfsmgraph` name is a module-free placeholder holding the namespace
  ([ADR 0005](0005-namespace-prefix-and-pep-420-layout.md)), and giving it dependencies
  would make it a meta-package the family has not decided it wants.
- **Making `numba-cuda` a hard dependency of `align`/`hmm`.** Rejected: it would make a
  CUDA toolchain mandatory for users who only ever run the Python or Cython phases,
  which is most of them.
- **Making `torch` optional in `dl` behind an extra.** Rejected as incoherent — every
  public surface of `dl` is a PyTorch model.
