# 0010. `dataseq` is a merge of three existing implementations, with `dl` as the base

- **Status:** **Proposed** — the composition is decided, but PRD §9 requires the encoder
  API reconciliation to be resolved *during the merge* before this record is finalized.
  Promote to `Accepted` when the merge lands.
- **Date:** 2026-08-21
- **Source:** PRD §1.5, §3.5, §8 — decision D10

## Context

[ADR 0009](0009-dataseq-as-the-base-layer.md) establishes `dataseq` as the base layer.
The question that immediately follows is what goes in it — and, unusually for a
foundational layer, the answer is not a design problem. Three implementations already
exist and have been inspected:

| Source | Maturity | Contribution |
|---|---|---|
| `dl` | Most mature; PyTorch-compatible by design | **The merge base.** Container semantics and the encoder/decoder (tokens ↔ numeric codes) |
| `hmm` (Lush) | Working but more primitive encoding | Design elements from a proven implementation; defines the interface the translation will need |
| `align` | Primitive proof-of-concept — a list of strings | No new design constraints |

There is also a sequencing question. `hmm` is the package expected to be *implemented*
first among the upper layers, because a complete working implementation already exists
in Lush and translating it is well-scoped. But translating `hmm` before `dataseq` exists
would force the translation to invent a sequence representation — producing a *fourth*
one, in a family whose defining problem is that it already has three.

## Decision

**`dataseq` is built by merging the three existing implementations, taking the `dl`
version as the base, and it is implemented first — before the Lush `hmm` translation
begins.**

**Why the `dl` implementation is the base.** It is the most mature of the three and was
built for PyTorch interoperability from the outset. Notably it required **no
`DataLoader` subclassing** — stock PyTorch `DataLoader` was usable directly. That means
the interoperability requirement is satisfied by *conforming to the existing `Dataset`
protocol* rather than by extending PyTorch's machinery, which is a much weaker and more
durable commitment.

**Why one implementation serves all consumers.** The token ↔ integer-code mapping has
different *purposes* across the family — indexing transition and emission matrices in
Baum-Welch, versus looking up rows in a vector embedding table — but the *structure* is
the same in both cases: a bidirectional mapping between symbols and contiguous integer
codes. This is [ADR 0001](0001-encode-at-the-boundary.md) generalized beyond alignment:
the integer code is the shared currency, and each consumer interprets it according to
its own needs. **Differing requirements do not justify differing representations.**

The proof-of-concept's `Alphabet` type overlaps directly with the encoder/decoder being
merged — the two express the same symbol ↔ integer-code mapping — so reconciling them is
part of this merge rather than a separate question. `ScoringMatrix` and
`AlignmentResult` are alignment-specific and stay in `align`.

## Consequences

### Positive

- **The Lush translation targets a defined interface rather than inventing one.** The
  interface `hmm` needs is not expected to differ substantially from what the `dl`
  implementation already provides, which dissolves the earlier concern that translating
  `hmm` first would harden a fourth representation.
- **The foundational layer is far more tractable than "foundational layer" usually
  implies** — a merge of three known, inspected implementations rather than a design to
  be invented.
- **PyTorch interoperability is inherited, not designed.** The base already satisfies it
  through the stock `Dataset` protocol.
- **Implementation order and dependency order coincide at the base**, then deliberately
  diverge above it: `dataseq` → `hmm` → `align` → `hseg` for implementation, while
  *release* order must still follow the graph
  ([ADR 0009](0009-dataseq-as-the-base-layer.md)). `hmm` may therefore be finished long
  before it can be released.

### Negative / costs

- **Three call sites' worth of existing code must be migrated**, and the `align`
  proof-of-concept additionally has to absorb the renumbering imposed by
  [ADR 0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md).
- **The `dl` base was designed for one consumer** and is now being asked to serve
  Baum-Welch as well. Where it turns out to be under-general, the cost lands during the
  translation.
- **`hmm` is blocked on `dataseq`.** The most tractable upper-layer package cannot start
  until the base is done.

## Alternatives considered

- **Design `dataseq` fresh from the three implementations' requirements.** Rejected:
  slower, and discards a working PyTorch-interoperable implementation for no identified
  gain.
- **Use the Lush `hmm` implementation as the base**, since it is the most proven.
  Rejected: its encoding is more primitive, and it carries no PyTorch interoperability —
  which would have to be added, having already been solved elsewhere.
- **Translate `hmm` first and extract `dataseq` afterwards.** Rejected: it hardens a
  fourth representation during the translation, then requires a second migration.
- **Separate representations per consumer, reconciled by adapters.** Rejected on the
  structural-identity argument above.

## Evidence

**Batching is not a complication.** The current `hmm` implementation does not use batch
training, so questions about reconciling batched and unbatched access do not arise at
this stage. (Note that this does *not* relax the `PAD`=0 requirement of
[ADR 0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md), since `dataseq`
serves `dl` as well.)

## Open

These are the items that keep this ADR at `Proposed`. All are to be resolved *during*
the merge, per PRD §8:

- The exact shape of the reconciled encoder API: **constructor signature**, the
  **strictness switch** (see
  [ADR 0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md)), and **how
  `align` consumes the mapping at its boundary**.
- Whether `dataseq` is pure-Python (hatchling) or carries performance-critical inner
  loops warranting compilation (meson-python) —
  [ADR 0008](0008-per-package-build-backends.md). The `dl`-derived base is presumed pure
  Python; confirm once merged.
