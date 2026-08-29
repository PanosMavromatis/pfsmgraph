# 0005. The `pfsmgraph` prefix and a PEP 420 namespace layout

- **Status:** Accepted
- **Date:** 2026-06-29 (names secured 2026-08-21)
- **Source:** PRD §3.1, §3.2, §3.3, §4 — decisions D1, D2, D3, D4

## Context

The ecosystem comprises five components that were written over two decades, in different
languages, for different purposes, and only recently recognized as one system (§1.3).
They are genuinely separable: a user who wants sequence alignment has no need to install
PyTorch, and a user training HMMs has no need for the deep-learning layer.

Shipping them as one monolithic package would force every user to take the union of the
dependencies. Shipping them as five unrelated packages would lose the family identity
that is the point — `align` and `hseg` are interpretability instruments for the outputs
of `hmm` and `dl`, not neighbors of them (§1.4).

A prefix system resolves the tension, provided a prefix is available.

## Decision

**The family uses the prefix `pfsmgraph` (probabilistic finite-state machine + graph),
with the PEP 420 implicit-namespace-package pattern.** Distribution names are
`pfsmgraph-<pkg>`; import names are `pfsmgraph.<pkg>`.

```
uv add pfsmgraph-align           # distribution name
import pfsmgraph.align as align  # import name
```

The roster is five packages (**D3**):

| Distribution | Import | Role |
|---|---|---|
| `pfsmgraph-dataseq` | `pfsmgraph.dataseq` | Data sequence container + symbol↔code encoder; base layer |
| `pfsmgraph-align` | `pfsmgraph.align` | Sequence alignment |
| `pfsmgraph-hseg` | `pfsmgraph.hseg` | Hierarchical segmentation |
| `pfsmgraph-hmm` | `pfsmgraph.hmm` | Baum-Welch; topology search by state merge/split |
| `pfsmgraph-dl` | `pfsmgraph.dl` | PyTorch models (`rnn`, `transformer` submodules) |

**Mechanical requirement — the one that breaks everything if violated: no
`pfsmgraph/__init__.py` may exist in any package.** The `pfsmgraph` level is a
namespace, not a regular package. Each distribution contributes exactly one regular
subpackage beneath it. An `__init__.py` at the namespace level in any one distribution
shadows the namespace and breaks every other package's imports.

`pfsmgraph` is the **only** namespace level; there is no second tier
([ADR 0007](0007-dl-as-a-single-distribution.md)).

**All six names are claimed on PyPI (D4)** — the five packages plus the bare `pfsmgraph`
umbrella — via module-free `0.0.0` placeholder releases (hatchling,
`bypass-selection = true`). These placeholders are intentionally dependency-free and
must stay that way: a stub declaring `pfsmgraph-dataseq>=0.1` would fail to resolve,
because no such version exists yet.

## Consequences

### Positive

- **Independent versioning and release per package**, with install-what-you-need
  granularity.
- **Family identity is legible from the name alone.** `pfsmgraph-hseg` reads as a
  component of a system, which is the accurate signal: the alignment library is
  deliberately not positioned to compete with general-purpose alignment tools, and the
  prefix says so (§1.1). What would be a discoverability cost for a standalone tool is
  an asset here.
- **The `as align` alias neutralizes the verbosity** at call sites, so the prefix costs
  nothing in day-to-day use.
- **A well-trodden pattern.** `azure-*`, `google-cloud-*`, and the legacy `zope.*`
  ecosystems all work this way.

### Negative / costs

- **The prefix foregrounds the substrate, not the differentiator.** A finite-state
  machine *is* a graph, so `graph` is conceptually redundant; and the "hierarchical"
  axis — arguably the more interesting claim — is carried by individual package names
  like `hseg` rather than by the family name. Accepted knowingly.
- **The no-`__init__.py` rule is a permanent trap.** It is invisible, easy to violate by
  reflex or by a tool's scaffolding, and its failure mode is a confusing `ImportError`
  in a *different* package. It is recorded in `CLAUDE.md` for this reason.
- **Namespace packages interact badly with some build backends' editable installs** —
  which is not hypothetical here; see
  [ADR 0012](0012-align-and-hmm-temporarily-on-hatchling.md).
- **Six PyPI projects to maintain, release, and keep reachable.**

## Alternatives considered

- **Bare `pfsm` as the prefix.** **Unavailable** — taken on PyPI by "Python Fast Strings
  Matching" (v0.1.3). Decisively so: string matching sits directly next to sequence
  alignment, so a bare-`pfsm` prefix would be simultaneously a namespace collision and a
  thematic one, the worst combination. **This is the reason `graph` stays**; the
  redundancy is what makes the prefix available at all.
- **`phfsm` / `hpfsm`,** encoding "hierarchical" in the prefix. Rejected as
  unpronounceable.
- **One monolithic package.** Rejected: forces the union of dependencies on every user.
- **Five unrelated names with no prefix.** Rejected: discards the family identity that
  §1.4 argues is substantive rather than cosmetic.

## Evidence

Operational findings from claiming the names, worth retaining because they are not
obvious and were learned the hard way:

- **A placeholder upload is what holds a name.** A trusted-publishing "pending
  publisher" reserves *nothing* — it is invalidated if someone else registers the name
  first. Trusted publishing is a publish-*security* mechanism, not a name-holding one.
  Correct order: claim by placeholder upload, then attach a normal trusted publisher to
  the now-existing project when CI is ready.
- **PyPI rate-limits new project creation, and the production limit is tighter than the
  documented defaults** (documented: 20/hour/user, 40/hour/IP). Creating four projects
  in one session triggered `429 Too many new projects created` on the fifth; retrying
  the following day succeeded. Bulk claiming must be planned across more than one day.
- Name normalization means claiming `pfsmgraph-align` also locks `pfsmgraph_align` and
  `pfsmgraph.align`.
- TestPyPI is a separate instance; a name claimed there reserves nothing on real PyPI.
- Placeholders should be replaced with real releases within a reasonable window — PEP
  541 treats content-free projects as somewhat more reclaimable — and the account email
  must stay reachable.
