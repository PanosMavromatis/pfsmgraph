# Architecture Decision Records

Each file records one decision: what was decided, why, what it cost, and what was
rejected. **These records — not [the PRD](../PRD.md) — are authoritative** for the
decisions they cover (PRD §9). The PRD remains the narrative design document and the
account of how the ecosystem came to be; where the two disagree, the ADR wins.

Numbers are permanent and never reused. To add one, copy
[`0000-template.md`](0000-template.md) to `NNNN-kebab-case-title.md` and add a row below.

## Index

| # | Title | Status | Date | PRD source |
|---|---|---|---|---|
| [0001](0001-encode-at-the-boundary.md) | Encode at the boundary | Accepted | 2025 | §1.2, §3.5 |
| [0002](0002-three-phase-algorithm-lifecycle.md) | Three-phase algorithm lifecycle: Python, then Cython, then CUDA | Accepted § | 2025 | §1.2, §6 |
| [0003](0003-one-parameterized-test-suite-per-algorithm.md) | One parameterized test suite per algorithm, run against every backend | Accepted | 2025 | §1.2 |
| [0004](0004-gpu-backends-and-optional-dependency-strategy.md) | GPU backends are two unrelated things; heavy dependencies stay optional | Accepted | 2025–2026 | §1.2, §6 |
| [0005](0005-namespace-prefix-and-pep-420-layout.md) | The `pfsmgraph` prefix and a PEP 420 namespace layout | Accepted | 2026-06-29 | §3.1–§3.3, §4 |
| [0006](0006-single-repository-as-a-uv-workspace.md) | One repository, structured as a uv workspace | Accepted | 2026-06-29 | §5 |
| [0007](0007-dl-as-a-single-distribution.md) | `dl` is a single distribution; there is no third namespace tier | Accepted | 2026-06-29 | §7 |
| [0008](0008-per-package-build-backends.md) | Build backends are per-package; meson-python for compiled members | Superseded † | 2026-06-29 | §6, §6.1 |
| [0009](0009-dataseq-as-the-base-layer.md) | `dataseq` is the dependency-graph base layer | Accepted | 2026-08-21 | §3.4 |
| [0010](0010-dataseq-composition-merging-three-implementations.md) | `dataseq` is a merge of three existing implementations, with `dl` as the base | Accepted ‡ | 2026-08-21 | §1.5, §3.5, §8 |
| [0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md) | Fixed reserved symbol block; encoding is strict by default | Accepted | 2026-08-21 | §3.6 |
| [0012](0012-align-and-hmm-temporarily-on-hatchling.md) | `align` and `hmm` are temporarily on hatchling, not meson-python | Superseded ¶ | 2026-08 | — |
| [0013](0013-api-documentation-layout-and-tooling.md) | API documentation: repo-level `docs/api/`, hand-written, examples executed | Accepted | 2026-09-01 | — |
| [0014](0014-scratch-retention-and-per-package-scoping.md) | Imported migration source is retained in `.scratch/`, scoped per package | Accepted | 2026-09-01 | — |
| [0015](0015-arc-emission-mealy-formulation.md) | The HMM is arc-emission (Mealy): symbols are emitted on transitions | Accepted | 2026-09-03 | — |
| [0016](0016-numba-cpu-parallel-phase.md) | Insert a Numba CPU-parallel phase between Cython and CUDA | Accepted | 2026-09-03 | — |
| [0017](0017-frozen-parameter-object-for-hmm.md) | `pfsmgraph.hmm` parameters are a frozen value, not a mutable model object | Accepted | 2026-09-03 | — |
| [0018](0018-family-wide-meson-python-build-backend.md) | The build backend is family-wide: all five members on meson-python | Accepted | 2026-09-04 | — |

† Was qualified in practice by 0012 until the first Cython kernel landed. Both are now
superseded by 0018, which replaces the per-package backend with a family-wide one; the
kernel never landed, and the qualification was resolved ahead of it rather than by it.
0008 is still worth reading for its build-needs table and its setuptools evidence.

¶ Superseded by 0018 on 2026-09-04, *before* its stated expiry rather than at it. Its
problem statement holds; two of its factual claims do not — see its status line.
‡ Held at Proposed until 2026-09-01. The composition was decided when the record was
authored, but PRD §9 required the encoder API reconciliation to be resolved during the
merge itself; it was, and 0010 records the settled API.
§ Amended by 0016, which inserts a phase between Cython and CUDA and renumbers CUDA from
phase 3 to phase 4.

## Reading order

- **0001–0004** are inherited from the proof-of-concept and predate the packaging work.
  They are the most pervasive constraints in the codebase and are summarized as hard
  rules in [`docs/agents/core.md`](../../agents/core.md), which the root `CLAUDE.md`
  imports and `AGENTS.md` is generated from.
- **0005–0011** are the packaging, namespace, and base-layer decisions, in the order the
  PRD settled them.
- **0012** records where the repository deviated from 0008 between 2026-08 and
  2026-09-04, and why. Superseded by 0018, but read first if 0018's Context is to make
  sense: 0018 is largely an argument with this record.
- **0013–0014** were added after the initial set, both postdating the PRD. 0013 settles
  how this family documents its public surfaces; 0014 settles how source imported *for a
  migration* is held, and supersedes the delete-at-merge intent the repository started
  with.
- **0015** is the first record about a *model*, rather than about packaging, tooling or
  process. It is where the `hmm` migration's most consequential fact is written down, and
  it is worth reading before revision 02's plan, since every array shape in that plan
  follows from it.
- **0016** amends 0002: it inserts a Numba CPU-parallel phase between Cython and CUDA, so
  "phase 3" in anything dated after 2026-09-03 means CPU-parallel, and CUDA is phase 4.
- **0017** is the second record about the `hmm` model, and reads after 0015: that one
  fixes what the parameters *mean*, this one fixes that they are an immutable value and
  that algorithms take them rather than own them. It is worth reading before revisions 03
  and 04, both of which are planned on the assumption that it was decided.
- **0018** closes the packaging thread 0008 opened and 0012 suspended: the build backend
  is family-wide, and it is meson-python for all five members, three of which compile
  nothing and never will. The reason is the PEP 420 namespace rather than the build —
  meson-python's editable install replaces `pfsmgraph.__path__`, so any member left on a
  plain `.pth` is shadowed. It is the only record so far that supersedes rather than
  amends, and the three-record sequence 0008 → 0012 → 0018 is the clearest worked example
  in this directory of a decision surviving two revisions of its own evidence.

## Coverage of the PRD decision table

Every decision D1–D11 in PRD §2 is covered: D1–D2 and D3–D4 by 0005, D5 by 0006, D6 by
0007, D7–D8 by 0008 (both now superseded by 0018), D9 by 0009, D10 by 0010, D11 by 0011. The inherited §1.2 decisions
are covered by 0001–0004. 0012, 0013, 0014, 0015, 0016, 0017 and 0018 have no PRD counterpart —
all seven postdate the document; 0012 qualifies §6.1 and 0018 overrides it, 0013 settles a question §9 never
raised, 0014 covers a working-area policy the PRD does not describe at all, 0015 answers a
question the PRD did not know it had left open: which HMM formulation `pfsmgraph-hmm`
implements, 0016 amends the phase count §1.2/§6 originally described as three, and 0017
settles a class-architecture question the PRD leaves to the migration — whether
`pfsmgraph.hmm` inherits the imported source's mutable model/working-copy split.
