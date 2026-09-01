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
| [0002](0002-three-phase-algorithm-lifecycle.md) | Three-phase algorithm lifecycle: Python, then Cython, then CUDA | Accepted | 2025 | §1.2, §6 |
| [0003](0003-one-parameterized-test-suite-per-algorithm.md) | One parameterized test suite per algorithm, run against every backend | Accepted | 2025 | §1.2 |
| [0004](0004-gpu-backends-and-optional-dependency-strategy.md) | GPU backends are two unrelated things; heavy dependencies stay optional | Accepted | 2025–2026 | §1.2, §6 |
| [0005](0005-namespace-prefix-and-pep-420-layout.md) | The `pfsmgraph` prefix and a PEP 420 namespace layout | Accepted | 2026-06-29 | §3.1–§3.3, §4 |
| [0006](0006-single-repository-as-a-uv-workspace.md) | One repository, structured as a uv workspace | Accepted | 2026-06-29 | §5 |
| [0007](0007-dl-as-a-single-distribution.md) | `dl` is a single distribution; there is no third namespace tier | Accepted | 2026-06-29 | §7 |
| [0008](0008-per-package-build-backends.md) | Build backends are per-package; meson-python for compiled members | Accepted † | 2026-06-29 | §6, §6.1 |
| [0009](0009-dataseq-as-the-base-layer.md) | `dataseq` is the dependency-graph base layer | Accepted | 2026-08-21 | §3.4 |
| [0010](0010-dataseq-composition-merging-three-implementations.md) | `dataseq` is a merge of three existing implementations, with `dl` as the base | Accepted ‡ | 2026-08-21 | §1.5, §3.5, §8 |
| [0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md) | Fixed reserved symbol block; encoding is strict by default | Accepted | 2026-08-21 | §3.6 |
| [0012](0012-align-and-hmm-temporarily-on-hatchling.md) | `align` and `hmm` are temporarily on hatchling, not meson-python | Accepted (temporary) | 2026-08 | — |
| [0013](0013-api-documentation-layout-and-tooling.md) | API documentation: repo-level `docs/api/`, hand-written, examples executed | Accepted | 2026-09-01 | — |
| [0014](0014-scratch-retention-and-per-package-scoping.md) | Imported migration source is retained in `.scratch/`, scoped per package | Accepted | 2026-09-01 | — |

† Qualified in practice by 0012 until the first Cython kernel lands.
‡ Held at Proposed until 2026-09-01. The composition was decided when the record was
authored, but PRD §9 required the encoder API reconciliation to be resolved during the
merge itself; it was, and 0010 records the settled API.

## Reading order

- **0001–0004** are inherited from the proof-of-concept and predate the packaging work.
  They are the most pervasive constraints in the codebase and are summarized as hard
  rules in [`CLAUDE.md`](../../../CLAUDE.md).
- **0005–0011** are the packaging, namespace, and base-layer decisions, in the order the
  PRD settled them.
- **0012** records where the repository currently deviates from 0008, and why.
- **0013–0014** were added after the initial set, both postdating the PRD. 0013 settles
  how this family documents its public surfaces; 0014 settles how source imported *for a
  migration* is held, and supersedes the delete-at-merge intent the repository started
  with.

## Coverage of the PRD decision table

Every decision D1–D11 in PRD §2 is covered: D1–D2 and D3–D4 by 0005, D5 by 0006, D6 by
0007, D7–D8 by 0008, D9 by 0009, D10 by 0010, D11 by 0011. The inherited §1.2 decisions
are covered by 0001–0004. 0012, 0013 and 0014 have no PRD counterpart — all three
postdate the document; 0012 qualifies §6.1, 0013 settles a question §9 never raised, and
0014 covers a working-area policy the PRD does not describe at all.
