# Master plan — pfsmgraph

**Status**: active

The two-tier plan convention for this repository:

- **Master plan** — this file, `docs/plan/TODO.md`, lives on `main`. It defines
  **revisions** (milestones) and their **subgoals**; each subgoal spawns a branch.
- **Branch plan** — `docs/plan/<type>-<slug>/TODO.md`, created by `/new-branch`, worked
  by `/hitl-step`, stamped `merged` by `/smart-merge`. It **survives on `main`** as the
  durable record of how a subgoal was executed, and is filed into its revision's
  directory by `/file-plans`.
- **PR body** — the distilled description plus a pointer to the branch plan directory.
  Not a verbatim archive.

`TODO.md` rather than `DO.md` is deliberate: branches inherit the master plan's model,
so every subgoal here runs under `/hitl-step` with its Q&A logged inline. The `dataseq`
merge reconciles three existing implementations and settles a public API that four other
packages depend on — the questions are genuinely open, and the answers are worth keeping.

**Related documents.** Decided-but-not-yet-actionable work is *not* listed here; it lives
in [`DEFERRED.md`](DEFERRED.md), indexed by the trigger that unblocks it. Open design
questions live in [PRD §8](../design/PRD.md) and the `Open` sections of the
[ADRs](../design/adr/README.md). This file tracks only work that is active now.

## Subgoals — revision 01-dataseq-v0.1.0

The workspace is scaffolded and `uv sync`s cleanly, but no algorithms are implemented.
PRD §11 puts `dataseq` first, and it is the only package that *can* go first: it is the
base layer, and the other four are blocked on the symbol↔code encoder it owns. This
revision implements it by merging the three data sequence implementations that already
exist — the `dl` version as the base, per PRD §3.5 — and carries it to a first real
release, replacing the `0.0.0` PyPI placeholder.

Settled, and not to be relitigated here: the `dl` implementation is the merge base
(§3.5); the reserved symbol block is fixed at `PAD`=0 … `MSK`=5 with user symbols from 6
and is not configurable ([ADR 0011](../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md));
and encoding is strict by default, with `UNK` fallback an explicit opt-in. What is *not*
settled — and is this revision's real design work — is the encoder API: the constructor
signature, the spelling of the strictness switch, and how `align` consumes the mapping at
its boundary. [ADR 0010](../design/adr/0010-dataseq-composition-merging-three-implementations.md)
sits at `Proposed` solely because of that, so settling it here is what promotes it.

Three items filed in [`DEFERRED.md`](DEFERRED.md) under the `dataseq` merge trigger must
land **as part of** this revision rather than after it: the ADR 0010 promotion, the
renumbering of the proof-of-concept alignment code to the reserved block, and the first
test suite written to the [ADR 0003](../design/adr/0003-one-parameterized-test-suite-per-algorithm.md)
standard. The renumbering especially — deferring it past the merge is how a code-only
edit becomes a data migration.

- [x] Merge the three `dataseq` implementations into `packages/pfsmgraph-dataseq/`, taking the `dl` version as the base (PRD §3.5): container semantics, PyTorch `Dataset` conformance, and the symbol↔code encoder/decoder. Stock `DataLoader` must remain usable without subclassing.
  > **Branch:** feat/dataseq-merge
  > **Done:** Six modules, ragged by construction, with padding confined to `pad_collate`
  > and always returned with its mask. numpy is the only runtime dependency; stock
  > `DataLoader` works without subclassing — PR #2
- [x] Settle the encoder API — constructor signature, the spelling of the strictness switch, and how `align` consumes the mapping at its boundary — then promote ADR 0010 from `Proposed` to `Accepted` and update its row in `docs/design/adr/README.md`.
  > **Branch:** feat/dataseq-merge
  > **Done:** `SymbolTable(symbols)`, strict by default with `on_unknown="unk"` as the
  > per-call opt-in, decoding total over every code including the reserved ones, and the
  > symbol→code mapping public because `align` reads it across a distribution boundary.
  > ADR 0010 accepted 2026-09-01 and its index row updated — PR #2
- [x] Renumber the proof-of-concept alignment code to the reserved block (user symbols from 6, new gap index), auditing every hard-coded index assumption. Lands here, not after.
  > **Branch:** refactor/reserved-block-renumber
  > **Done:** `tokalign` renumbered onto the block -- `GAP` 3 -> 4, user symbols 4 -> 6,
  > and `RESERVED_INDICES` replaced by module-level `Final` constants, so the block can no
  > longer be passed to the constructor. The audit's one real find was
  > `ScoringMatrix.identity` zeroing `range(RESERVED_INDICES + 1)`, which after the move
  > would have silently blanked the first user symbol's scores. 62 `tokalign` tests pass,
  > matching the pre-change baseline exactly. Making `decode` total was carved out to
  > `DEFERRED.md` under the `align` migration — PR #5
- [x] Fix `dataseq`'s third-party runtime dependencies — `dependencies = []` is a placeholder — and confirm its build backend stays hatchling ([ADR 0008](../design/adr/0008-per-package-build-backends.md)).
  > **Done:** `dependencies = ["numpy>=1.24"]`, replacing the `[]` placeholder;
  > `build-backend = "hatchling.build"` confirmed and recorded in ADR 0010 §Resolved,
  > after reading all four imported sources for a compiled inner loop belonging to
  > `dataseq`. There is none — PR #2
- [x] Write the first test suite to the ADR 0003 standard, including the `pytest_report_header` hook that prints the backend matrix and names every excluded backend with its reason.
  > **Done:** 74 tests landed, the first in this repository. **The `pytest_report_header`
  > hook did not** — there are no backends to enumerate until the first `.pyx`, and no
  > `conftest.py` exists yet. Left `[~]` rather than `[x]` so this revision cannot close
  > while the hook is still owed — PR #2
  > **Branch:** feat/backend-matrix-header
  > **Done:** the hook landed too, so the `[~]` closes. Root `conftest.py` over
  > `_backends.py`, 13 tests, suite 74 -> 87; the matrix is empty by design and says so
  > rather than printing nothing. Sited at the rootdir by measurement, not taste -- a
  > nested conftest's `pytest_report_header` is discarded silently — PR #6
- [ ] Release `pfsmgraph-dataseq` 0.1.0, replacing the `0.0.0` placeholder, and set honest lower bounds on the intra-family dependencies that name it.

## Closed revisions

Extracted by `/close-revision` once finished. The subgoals, their `> **Done:**` records,
and the branch plans that executed them all live in the revision's directory.

_None yet._
