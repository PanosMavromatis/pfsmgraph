# 0001. Encode at the boundary

- **Status:** Accepted
- **Date:** 2025 (proof-of-concept); formalized 2026-08-29
- **Source:** PRD §1.2, §3.5 — inherited from the proof-of-concept alignment library

## Context

This ecosystem models sequences of **arbitrary multi-character symbols**, not single
characters. A symbol may be `"the"`, `"NP"`, `"onset"`, or any other string the corpus
requires. That is a deliberate modeling commitment: §1.1's "letters / words / phrases"
hierarchy is meaningless if a symbol must be one character wide.

Multi-character strings cannot serve as array indices. Every algorithm in the family
that matters — dynamic-programming alignment, Baum-Welch's transition and emission
matrices, embedding-table lookups — is fundamentally *array indexing in an inner loop*.
So the representation used for modeling and the representation used for computation
cannot be the same, and the boundary between them has to be placed somewhere explicit.

Placing it badly is the failure mode. If string handling leaks into inner loops, every
compiled or GPU backend must reimplement string semantics — and Cython and CUDA are
precisely where doing so is most painful and least portable.

## Decision

**Symbols are encoded to integers at the entry point of every public call, all inner
computation is integer-only, and results are decoded back to strings at exit.**

The boundary is the public API surface. Above it, the family speaks in strings; below
it, exclusively in contiguous non-negative integer codes. No inner routine, in any
backend, ever sees a string type.

## Consequences

### Positive

- **Compiled and GPU backends become mechanical to write.** A Cython kernel or a Numba
  CUDA kernel operates on typed integer memoryviews and device arrays. It never touches
  a Python string object, so it never needs the GIL for symbol handling and never needs
  a string-aware device-side representation. This is the consequence that justifies the
  discipline, and it is what makes the three-phase lifecycle in
  [ADR 0002](0002-three-phase-algorithm-lifecycle.md) tractable rather than aspirational.
- **The integer code is the family's shared currency.** Consumers interpret the same
  code differently — `hmm` indexes transition and emission matrices with it, `dl` looks
  up an embedding row — but the *structure* is identical in both cases: a bidirectional
  mapping between symbols and contiguous integer codes. That equivalence is the
  substantive argument for a single shared `dataseq` implementation
  ([ADR 0010](0010-dataseq-composition-merging-three-implementations.md)) rather than
  one encoder per consumer.
- **Backend equivalence becomes testable.** With a string boundary, backends could
  diverge on encoding-adjacent behavior; with an integer boundary they receive
  identical input, so [ADR 0003](0003-one-parameterized-test-suite-per-algorithm.md)
  compares like with like.
- **Persistence and interchange are well-defined.** A trained model stores codes; the
  encoder stores the mapping.

### Negative / costs

- **Every public entry point pays an encode/decode pass.** For short sequences and
  large alphabets this is measurable overhead against the algorithm itself.
- **The mapping is now a piece of state that must travel with any persisted artifact.**
  A trained model plus a lost encoder is an unreadable trained model.
- **The discipline is unenforceable by the type system** and has to be maintained by
  review. A helper that accepts a string "just this once" for convenience is how the
  boundary erodes.
- **It concentrates risk in the reserved index allocation.** Because codes are baked
  into every persisted artifact, the reserved block in
  [ADR 0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md) can never be
  renumbered after the first artifact is written.

## Alternatives considered

- **Strings throughout, with dictionary lookups in the inner loop.** Rejected: it
  forecloses the compiled and GPU phases entirely, which are the point of the DP-heavy
  packages.
- **Encoding at the outermost application layer, above the library.** Rejected: it
  pushes the reserved-block contract onto every caller and makes the library's own API
  untypeable in terms of codes. The library would also lose the ability to validate.
- **Single-character symbols only, using ordinals directly.** Rejected: it contradicts
  the modeling domain (§1.1). Hierarchical tokens are words and phrases, not letters.
- **Per-package encoders, each tuned to its consumer's needs.** Rejected in §3.5: the
  purposes differ, the structure does not, and the family had already grown three
  incompatible representations that way (see
  [ADR 0010](0010-dataseq-composition-merging-three-implementations.md)).

## Evidence

The proof-of-concept alignment library implements exactly this discipline via its
`Alphabet` type (bidirectional string ↔ integer mapping over arbitrary multi-character
symbols) feeding a `ScoringMatrix` indexed by integer symbol IDs. PRD §1.2 records that
this discipline, together with relative-import hygiene, is *why* the restructuring into
a multi-package layout was tractable at all: the code moved cleanly because the layers
were already separated along the encode boundary.
