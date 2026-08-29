# 0011. Fixed reserved symbol block; encoding is strict by default

- **Status:** Accepted
- **Date:** 2026-08-21
- **Source:** PRD §3.6 — decision D11

## Context

[ADR 0001](0001-encode-at-the-boundary.md) makes integer codes the currency of the whole
family. Certain codes are not user symbols but structural markers: padding, gaps,
sequence boundaries, masks. Every package needs some of them, and they must mean the
same thing everywhere — a `GAP` code that `align` emits and `hseg` misreads is a silent
corruption, not an error.

The decision is unusually irreversible. Codes are baked into every persisted artifact: a
saved encoding, a trained embedding table, a stored transition matrix. Renumbering after
the fact invalidates all of them. So the allocation has to be right *before* anything is
persisted, and it has to be decided once rather than negotiated per package.

## Decision

**The reserved index allocation is fixed in `dataseq` and is not configurable.** Every
package may assume this layout without negotiation — that is the point of placing it at
the base layer ([ADR 0009](0009-dataseq-as-the-base-layer.md)).

| Index | Code | Purpose |
|---|---|---|
| 0 | `PAD` | Padding / absence |
| 1 | `UNK` | Unknown symbol |
| 2 | `BOS` | Beginning of sequence |
| 3 | `EOS` | End of sequence |
| 4 | `GAP` | Alignment gap |
| 5 | `MSK` | Masking |

**User symbols are numbered from 6.** All six codes are three characters, which keeps
aligned display of multi-character symbols uniform.

**Encoding is strict by default.** The encoder raises on unseen symbols; graceful
fallback to `UNK` is an explicit opt-in (e.g. `on_unknown="raise"` as the default,
`"unk"` where resilient inference is genuinely wanted).

### Why `PAD` is 0

PyTorch's zero-fill idiom makes this close to mandatory. `pad_sequence` defaults to
`padding_value=0`, batch buffers are naturally allocated with `torch.zeros()`, and
collate functions pad with zeros unless told otherwise. If `PAD` were any other index,
every collate path would need to pass the value explicitly, and **any zero-initialized
tensor would silently mean something other than "nothing here"** — a quiet and
hard-to-trace class of bug, sitting in the foundational layer where it would propagate
everywhere.

Note what does *not* constrain the choice: `nn.Embedding(padding_idx=…)` accepts any
index. It is the zero-fill convention, not the embedding API, that forces `PAD`=0.

That `hmm` does not currently batch is not a reason to relax this, since `dataseq`
serves `dl` as well ([ADR 0010](0010-dataseq-composition-merging-three-implementations.md)).

### Why `UNK` exists but strictness is the default

An unknown symbol appearing in **training** data indicates an upstream error — a broken
tokenizer, a corpus mismatch, an unfiltered import — not a condition to absorb silently.
Absorbing it produces a model that trained happily on partially meaningless input.
Raising by default enforces corpus curation *mechanically* rather than by convention,
while keeping the slot available for inference against real-world data, where
resilience is genuinely wanted.

### Why `MSK` is reserved before it is needed

Masked-objective training and masking-based interpretability probes (occlusion, feature
ablation) are plausible in `dl`, and interpretability is a core motivation for the
ecosystem (§1.4). Reserving the slot now costs one index; retrofitting it later would
shift every user symbol and invalidate every persisted encoding and trained embedding
table.

## Consequences

### Positive

- **No cross-package negotiation about structural codes, ever.** `GAP` is a `dataseq`
  concern rather than an `align` convention, so any package can read an alignment
  result's gaps without depending on `align`'s internals.
- **Zero-filled tensors mean "absent" throughout**, which is what every PyTorch idiom
  already assumes.
- **Corpus errors surface at encode time**, at the boundary, with the offending symbol
  in hand — rather than as degraded model quality weeks later.
- **The layout can be relied on in compiled kernels.** A Cython or CUDA kernel can treat
  "code < 6" as "structural" without consulting a runtime configuration object, which
  matters given that [ADR 0001](0001-encode-at-the-boundary.md) keeps those kernels
  free of Python objects.

### Negative / costs

- **Permanently unrenumberable once anything is persisted.** A seventh reserved code
  cannot be added at index 6 later; it would have to go above the user symbols, breaking
  the "structural codes are the low ones" property.
- **Not configurable**, so a user with an existing corpus using a different convention
  must re-encode.
- **The alphabet grows by six, and a scoring matrix is (n+6)² rather than n².**
  Immaterial for the curated alphabets this ecosystem targets, and a small price for
  never having to renumber.
- **It renumbers the existing proof-of-concept code.** The proof-of-concept alignment
  types use a different allocation (padding/BOS/EOS low, gap immediately after, user
  symbols from 4). Adopting this shifts every user symbol and changes the gap index.
  Nothing is persisted yet, so this is a **code-only change** — but it must land **as
  part of the merge, not after**
  ([ADR 0010](0010-dataseq-composition-merging-three-implementations.md)), and any
  hard-coded index assumptions in the proof-of-concept alignment code need auditing.
- **Strict-by-default will be experienced as friction** by anyone feeding in
  uncurated data, and the opt-in has to be discoverable enough that they find it rather
  than pre-filtering their corpus by hand.

## Alternatives considered

- **A configurable reserved block.** Rejected: it would put the layout back into
  per-package negotiation and force every kernel to consult configuration, defeating the
  purpose of fixing it at the base layer.
- **Omitting `UNK`.** Rejected: adding it later would renumber every user symbol.
- **Omitting `MSK` until masking is actually implemented.** Rejected for the same
  reason, at a cost of exactly one index.
- **Any index other than 0 for `PAD`.** Rejected on the zero-fill argument above.
- **Lenient encoding by default, with a strict opt-in.** Rejected: it inverts the risk,
  making the silent-corruption path the one you get by not thinking about it.
- **Placing `GAP` in `align` rather than `dataseq`.** Rejected: it would make the
  reserved block partially owned, which is the negotiation this ADR exists to prevent.

## Open

The exact spelling of the strictness switch is part of the encoder API reconciliation
still open in [ADR 0010](0010-dataseq-composition-merging-three-implementations.md).
The *semantics* — raise by default, fall back only on explicit request — are settled
here and are not up for renegotiation during that reconciliation.
