# 0007. `dl` is a single distribution; there is no third namespace tier

- **Status:** Accepted
- **Date:** 2026-06-29
- **Source:** PRD §7 — decision D6

## Context

The deep-learning component contains two model families, RNNs and transformers. Since
the rest of the ecosystem is split into separately publishable packages
([ADR 0005](0005-namespace-prefix-and-pep-420-layout.md)), the same question arises one
level down: should `rnn` and `transformer` be separate distributions?

Doing so would create `pfsmgraph.dl.rnn` and `pfsmgraph.dl.transformer` as *distinct
distributions sharing the `pfsmgraph.dl` prefix* — that is, a **second namespace level**,
with `dl` becoming a namespace package rather than a regular one.

The architectural fact that settles it: the two are not alternatives. They are intended
to co-exist **within the same model** — for example an RNN operating at the shortest
time-scale with transformers higher in the hierarchy, which is a direct expression of
the multi-level modeling the ecosystem exists for (§1.1). A user of one is very likely a
user of the other, in the same forward pass.

## Decision

**`rnn` and `transformer` ship as one distribution, `pfsmgraph-dl`, with
`pfsmgraph.dl.rnn` and `pfsmgraph.dl.transformer` as plain submodules of a regular
package.**

Consequently **`pfsmgraph` is the only namespace level in the family.** `dl` is a
regular package with a normal `__init__.py`; only the `pfsmgraph` level above it is
implicit.

## Consequences

### Positive

- **The namespace stays one level deep**, which keeps the no-`__init__.py` rule of
  [ADR 0005](0005-namespace-prefix-and-pep-420-layout.md) a single, statable
  requirement rather than a per-level judgment call.
- **It avoids compounding known friction.** Namespace packages, `src/` layout, and
  compiled editable installs already interact awkwardly — see
  [ADR 0012](0012-align-and-hmm-temporarily-on-hatchling.md), where a build backend's
  editable hook claims the whole namespace. Spanning a *second* namespace level across
  distributions would multiply that surface for no gain.
- **Cross-model code has an obvious home.** Shared layers, a hierarchical model
  composing an RNN with a transformer, and shared training utilities live in `dl`
  without a circular dependency between two distributions.
- **One version for the deep-learning layer**, which matches the reality that the two
  submodules will be changed together.

### Negative / costs

- **A user who wants only RNNs installs the transformer code too.** Negligible in
  practice: both are pure-Python model definitions over `torch`, which is the actual
  weight of the install ([ADR 0004](0004-gpu-backends-and-optional-dependency-strategy.md)).
- **`rnn` and `transformer` cannot be released independently**, so a fast-moving
  transformer implementation forces version bumps that also cover stable RNN code.
- **`dl` will be the largest package by source volume**, and internal boundaries within
  it are maintained by convention rather than enforced by packaging.

## Alternatives considered

- **Separate `pfsmgraph-dl-rnn` and `pfsmgraph-dl-transformer` distributions.**
  Rejected on two counts. It is not well-motivated given how tightly the two are coupled
  in intended use; and it would introduce a multi-level namespace (`pfsmgraph.dl.*`
  spanning distributions), compounding the namespace + `src`-layout + compiled
  editable-install friction for no benefit.
- **Folding `dl` into a single family-wide distribution.** Rejected at
  [ADR 0005](0005-namespace-prefix-and-pep-420-layout.md) — `torch` is far too heavy to
  impose on `align` users.

## Open

Separate distributions would be justified by genuinely divergent heavy dependencies or
by independent release cadence. Neither is anticipated. If one materializes — a
transformer implementation requiring a large extra dependency the RNN path does not use
— this decision should be revisited rather than worked around.
