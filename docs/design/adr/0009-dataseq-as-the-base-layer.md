# 0009. `dataseq` is the dependency-graph base layer

- **Status:** Accepted
- **Date:** 2026-08-21
- **Source:** PRD §3.4 — decision D9

## Context

`align` was originally assumed to be the family's common base: `hseg`, `hmm`, and `dl`
would all depend on it, and nothing would sit beneath it.

That assumption did not survive contact with the history. Three independent data
sequence representations had already been written — one in the Lush `hmm` code, one in
the alignment component's type foundation, one in the deep-learning component — before
anyone decided they should be shared (§1.3). The duplication was not a risk to be
guarded against; it had already happened twice. Whatever holds a sequence of symbols and
maps them to integer codes ([ADR 0001](0001-encode-at-the-boundary.md)) is needed by
every package in the family, including `align` itself.

Putting that shared foundation *inside* `align` would make every consumer of the data
container also a consumer of the alignment algorithms — including `dl`, which has no use
for them.

## Decision

**`dataseq` is extracted as a fifth package and placed beneath `align`, with no
intra-family dependencies of its own.**

```
                  dataseq          (base — no intra-family dependencies)
                     │
                   align
                  ╱  │  ╲
              hseg  hmm  dl
```

`align` depends on `dataseq`. `hseg`, `hmm`, and `dl` depend on `dataseq` and `align`.
**`dataseq` depends on nothing in the family, and this is a rule rather than a
circumstance** — it is what makes the graph provably acyclic and the release order
unambiguous.

Because the base layer is where family-wide contracts belong, the reserved symbol block
is fixed *here* and assumed by everyone without negotiation
([ADR 0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md)) — including
`GAP`, which is therefore a `dataseq` concern rather than an `align` convention.

## Consequences

### Positive

- **`dl` no longer needs `align` in order to have a data container.** The dependency it
  actually has is on the container, and now it can express exactly that.
- **The graph is acyclic by construction**, which makes the release order
  unambiguous: `dataseq` → `align` → {`hseg`, `hmm`, `dl`}. A package cannot publish
  before its declared dependencies exist on PyPI, so an unambiguous order is a practical
  necessity, not an aesthetic one.
- **There is one place to put family-wide contracts.** The reserved block, the encoder
  semantics, and the `Dataset` protocol conformance all have an obvious home that every
  package can assume.
- **The fourth duplicate is prevented.** Building the base layer *first*
  ([ADR 0010](0010-dataseq-composition-merging-three-implementations.md)) removes the
  risk that the Lush translation hardens a fourth representation.

### Negative / costs

- **A fifth package to version, release, and document**, and the one every other release
  is now gated behind.
- **`dataseq` is the hardest package to change compatibly.** A breaking change there
  breaks all four others, and its version bound appears in every sibling's metadata —
  where the workspace footgun of
  [ADR 0006](0006-single-repository-as-a-uv-workspace.md) means a wrong bound will not
  surface locally.
- **The no-intra-family-dependencies rule constrains `dataseq` permanently.** Any future
  feature that would want alignment or model code has to be placed elsewhere or done
  without.
- **Release is serialized.** Nothing in the family can be published until `dataseq` has
  a real release replacing its `0.0.0` placeholder.

## Alternatives considered

- **Keep the shared types inside `align`, as originally assumed.** Rejected: it forces
  `dl` to depend on alignment algorithms it does not use, and misrepresents the actual
  dependency.
- **Duplicate the container in each package, keeping them in sync by convention.**
  Rejected — this is precisely the status quo that produced three incompatible
  implementations without anyone intending it.
- **Put the shared types in the bare `pfsmgraph` umbrella distribution.** Rejected: the
  bare name is a module-free namespace placeholder
  ([ADR 0005](0005-namespace-prefix-and-pep-420-layout.md)), and giving it content would
  make the namespace holder a real package with real dependencies.

## Open

Whether `hseg`, `hmm`, and `dl` have dependencies on *each other* is unresolved (§8).
The base and the common mid-layer are settled; the top of the graph is not. This affects
how much the atomic-commit benefit of
[ADR 0006](0006-single-repository-as-a-uv-workspace.md) is worth in practice.
