# 0019. A declared intra-family dependency lands with its first import

- **Status:** Accepted — amends [ADR 0009](0009-dataseq-as-the-base-layer.md). Its graph
  still describes where the family's dependencies are *expected* to come from; it no
  longer describes what each member *declares*, and only declared edges fix release
  order. The base-layer rule, that `dataseq` depends on nothing in the family, is
  unchanged.
- **Date:** 2026-09-13
- **Source:** none in the PRD — postdates it. Qualifies PRD §3.3, §3.4, §5.5 and §11.

## Context

ADR 0009 draws `hseg`, `hmm` and `dl` above `align` and derives from that picture the
release order `dataseq` → `align` → {`hseg`, `hmm`, `dl`}. When the workspace was
scaffolded, every member's `pyproject.toml` declared the edges as drawn, so `hmm` has
carried `pfsmgraph-align>=0.1` since before it had any code.

`hmm` 0.1.0 is ready to release, and no module in it imports `align`. It exports a
parameter value and a decode, and neither touches alignment. The edge exists as *intent*:
PRD §1 calls alignment "a training accelerant" for topology search, and the mechanism
behind that sentence is the alignment-derived seed that `docs/plan/DEFERRED.md` records
under "`align` able to produce a multiple alignment". That revision comes after revision
04's merge/split search, which ports the original's search as it stands, so the earliest
`hmm` version that could import `align` is the one after 0.3.0.

The declared edge meanwhile makes 0.1.0 impossible to install. `align` has only its
`0.0.0` placeholder on PyPI, and its migration comes after `hmm`'s (PRD §1.5), so a
published `hmm` requiring `pfsmgraph-align>=0.1` fails to resolve for every pip user.
Nothing local shows it: a `{ workspace = true }` source satisfies any bound, and `uv.lock`
records no specifier for a workspace member (the workspace footgun, `core.md`). PRD §1.5
anticipated that `hmm` "may be finished well before it can be released". It did not ask
whether the edge holding it back was real.

## Decision

**A member declares a dependency on another family member when a module in it first
imports that member, and not before.** ADR 0009's graph remains the family's intended
structure, but a drawn edge is not a declared one, and only declared edges constrain
release order. The rule is checked at each member's **release commit**, since that is
when metadata becomes immutable. Before a release, an unimported bound harms no one.

Applied now, `pfsmgraph-hmm` drops `pfsmgraph-align>=0.1` and its `[tool.uv.sources]`
entry, and releases directly after `dataseq`. The edge returns with the
alignment-seeded topology revision, expected to be `hmm` 0.4.0, under the `DEFERRED.md`
trigger above. `hseg` and `dl` keep their declared `align` bounds until their own release
commits, where the same rule applies.

## Consequences

### Positive

- **`hmm` 0.1.0 installs.** Its declared intra-family dependency is `dataseq` alone,
  which is on PyPI.
- **Release order is what the imports force**, not what the drawing suggests. `hmm` is
  releasable now, rather than waiting on `align`, which has not been migrated.
- **Consumers do not pull in `align` for nothing.** `align` is a compiled member with its
  own accelerator extras, and a user who only decodes would have received it unused.
- **Published metadata tells the truth**, which is the one property of it that cannot be
  fixed after publishing.

### Negative / costs

- **Two graphs to keep straight.** ADR 0009's drawing is intent, and the `pyproject.toml`
  files are fact. A reader who takes the drawing as the declared graph will mispredict
  release order.
- **The edge re-enters in a minor version.** When `hmm` 0.4.0 imports `align`, upgrading
  pulls in a new family dependency, and 0.4.0's release is again gated on `align` being
  on PyPI. Under 0.x that is permitted, but it has to be stated in that release.
- **The rule is enforced by review only.** No test compares declared dependencies with
  imports, and the workspace hides a violation locally, so the release commit's review
  is the whole mechanism.

## Alternatives considered

- **Release `align` first, as ADR 0009's order says.** Rejected: `align` has not been
  migrated, and its migration comes after `hmm`'s. It would hold `hmm` back for at least
  a revision, for an edge no code uses.
- **Keep the bound and publish anyway.** Rejected: the release would not install.
- **Relax the bound to something PyPI satisfies, such as `>=0.0.0`.** Rejected: it would
  install an empty placeholder, a dependency that promises nothing. It keeps a false edge
  in immutable metadata only to make it resolvable.
- **Move `align` into an optional extra.** Rejected: an extra promises something usable
  if you opt in, and there is still nothing in `hmm` for it to enable.

## Evidence

- *Observed, 2026-09-13:* no module or test under `packages/pfsmgraph-hmm/` imports
  `pfsmgraph.align`. The name appears only in comments about ADR 0003's backend-selection
  seam.
- *Observed:* PyPI carries `pfsmgraph-align` `0.0.0` only (`core.md`, workspace footgun).
- *Reasoned:* the earliest import of `align` is the alignment-seeded revision, since
  revisions 03 and 04 port Lush code that has no alignment input (`DEFERRED.md`: zero
  hits for epsilon or silent machinery in `Code/HMMlib/`).

## Open

Whether the edge returns as a hard dependency or as an extra. The alignment seed replaces
merge/split's *starting point* rather than the search itself, so a user who seeds from a
single-state model might never need `align`. Settle it when that revision opens, from
how the seed is exposed.

**Expected answer, 2026-09-13: both, one after the other.** At 0.4.0 `align` is an
*extra*, while seeding from an alignment is one way among others to start the search. By
the first major release (1.x) it is expected to be *required*, because alignment will by
then be integral to how an `hmm` is trained. This is an expectation, not a decision, and
it is confirmed when the 0.4.0 revision opens. The second step also moves the rule's
timing. This record lets an edge land with its first import, but a *required* edge
arrives when the import stops being optional, and that is a change to what every install
pulls in, which belongs in a major version.
