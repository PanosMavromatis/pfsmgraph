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
| [0009](0009-dataseq-as-the-base-layer.md) | `dataseq` is the dependency-graph base layer | Accepted ‖ | 2026-08-21 | §3.4 |
| [0010](0010-dataseq-composition-merging-three-implementations.md) | `dataseq` is a merge of three existing implementations, with `dl` as the base | Accepted ‡ | 2026-08-21 | §1.5, §3.5, §8 |
| [0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md) | Fixed reserved symbol block; encoding is strict by default | Accepted | 2026-08-21 | §3.6 |
| [0012](0012-align-and-hmm-temporarily-on-hatchling.md) | `align` and `hmm` are temporarily on hatchling, not meson-python | Superseded ¶ | 2026-08 | — |
| [0013](0013-api-documentation-layout-and-tooling.md) | API documentation: repo-level `docs/api/`, hand-written, examples executed | Accepted | 2026-09-01 | — |
| [0014](0014-scratch-retention-and-per-package-scoping.md) | Imported migration source is retained in `.scratch/`, scoped per package | Accepted | 2026-09-01 | — |
| [0015](0015-arc-emission-mealy-formulation.md) | The HMM is arc-emission (Mealy): symbols are emitted on transitions | Accepted | 2026-09-03 | — |
| [0016](0016-numba-cpu-parallel-phase.md) | Insert a Numba CPU-parallel phase between Cython and CUDA | Accepted | 2026-09-03 | — |
| [0017](0017-frozen-parameter-object-for-hmm.md) | `pfsmgraph.hmm` parameters are a frozen value, not a mutable model object | Accepted | 2026-09-03 | — |
| [0018](0018-family-wide-meson-python-build-backend.md) | The build backend is family-wide: all five members on meson-python | Accepted | 2026-09-04 | — |
| [0019](0019-declared-dependencies-follow-imports.md) | A declared intra-family dependency lands with its first import | Accepted | 2026-09-13 | — |
| [0020](0020-scaled-probability-domain-forward-backward.md) | Forward-backward runs in the scaled probability domain, in a fixed order | Accepted | 2026-09-14 | — |
| [0021](0021-runtime-backend-selection.md) | A backend is chosen per call, defaults to `python`, and is never substituted | Accepted | 2026-09-14 | — |
| [0022](0022-state-split-initialisation.md) | A state split preserves what was learned, and breaks symmetry on inbound arcs | Accepted | 2026-09-16 | — |
| [0023](0023-topology-search-loop.md) | The topology search walks the original's loop and returns the best model it visits | Accepted | 2026-09-17 | — |
| [0024](0024-search-compiled-work.md) | The topology search stays interpreted, and every forward pass it runs takes a named backend | Accepted | 2026-09-17 | — |

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
‖ Amended by 0019: the graph stands as intent, declared dependencies follow imports, and
release order follows what is declared, so `hmm` releases before `align`.

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
- **0019** amends 0009: the drawn graph is intent, and a member declares an edge on another
  member when a module in it first imports across that edge, checked at its release
  commit. Read it before any release, since it is what lets `hmm` publish ahead of
  `align`.
- **0021** answers the question 0003 left open: how a caller chooses a backend at runtime.
  It is per call, defaults to the pure-Python reference, never falls back, and lets a caller
  list what runs here. Read it before any public call that reaches a compiled kernel, and
  before touching the repo-root `_backends.py`, whose table it moves into `hmm`.
- **0022** is the third record about the model, after 0015 and 0017: how a topology split
  builds its candidate. It fixes a defect in the Lush original (`hmm-param.lsh:172`, repeated
  at `:262` in the merge) and replaces the original's outbound emission redraw with a
  per-predecessor inbound perturbation plus a light emission seed, so no learned parameter
  is discarded. Read it before writing the split or merge, `try-split`, or the search loop,
  which it obliges to give a split a minimum EM budget.
- **0023** is the search loop that 0022 obliges, and it reads after 0022. It ports the
  original's headless `repeat (suggest-move) (keep-model)` loop, which this project had
  believed did not exist. The departures are named: the walk keeps its unconditional move
  but returns the best model it visits, it stops on patience, a required round cap or a
  dead end, `d` is chosen by a bounded exact scan rather than Brent's local minimum, a
  split gets 200 EM cycles, and the total's forward pass takes the search's backend. Read
  it before touching the search, `_trials.py`'s choice of `d`, or `_mdl.py`'s signatures.
- **0024** reads after 0023. It keeps the search, the trials and the surgery
  interpreted, since a profile put 84% of a round in a numpy forward pass inside
  `baum_welch`'s convergence check rather than in the search's own code, and requires that
  check to take its own named backend, `score_backend=` on `baum_welch`, as the scan does. It also
  defers parallelizing the search, along trials within a round, and states the conditions
  under which that stays bit-identical to the serial run. Read it before compiling or
  parallelizing anything above the kernels, or changing `baum_welch`'s signature.

## Coverage of the PRD decision table

Every decision D1–D11 in PRD §2 is covered: D1–D2 and D3–D4 by 0005, D5 by 0006, D6 by
0007, D7–D8 by 0008 (both now superseded by 0018), D9 by 0009, D10 by 0010, D11 by 0011. The inherited §1.2 decisions
are covered by 0001–0004. 0012, 0013, 0014, 0015, 0016, 0017, 0018, 0019, 0020, 0021, 0022, 0023 and 0024 have no PRD counterpart —
all thirteen postdate the document; 0012 qualifies §6.1 and 0018 overrides it, 0013 settles a question §9 never
raised, 0014 covers a working-area policy the PRD does not describe at all, 0015 answers a
question the PRD did not know it had left open: which HMM formulation `pfsmgraph-hmm`
implements, 0016 amends the phase count §1.2/§6 originally described as three, and 0017
settles a class-architecture question the PRD leaves to the migration — whether
`pfsmgraph.hmm` inherits the imported source's mutable model/working-copy split, 0019
qualifies the release order §3.4 and §11 derive from the graph, and 0020 is a numeric
contract for a recurrence the PRD names only as "Baum-Welch", 0021 settles the runtime
backend-selection question that 0003 deliberately left open, 0022 departs from the
imported split surgery the PRD names only as "state merge and split", 0023 settles the
search strategy the PRD names only as "topology search", and 0024 settles where that
search's compiled work lives.
