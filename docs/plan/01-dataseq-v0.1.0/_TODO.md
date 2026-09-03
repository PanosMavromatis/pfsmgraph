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
- [x] Release `pfsmgraph-dataseq` 0.1.0, replacing the `0.0.0` placeholder, and set honest lower bounds on the intra-family dependencies that name it.
  > **Branch:** chore/release-dataseq-0.1.0
  > **Done:** 0.1.0 is on PyPI and tagged `pfsmgraph-dataseq-v0.1.0`, verified by importing
  > the published wheel rather than the local one. The version bump was the small part: the
  > member also gained a PyPI-facing README, a real LICENSE copy, `[project.urls]` and a PEP
  > 561 marker, all of which fail silently if placed wrong and none of which can be fixed in
  > place afterwards. The four dependents' bounds now read `>=0.1.0`, reviewed against a
  > version that exists. The release path is a `justfile` plus `docs/ops/release.md`, with
  > its guards written as prerequisites because `just` runs body lines after `publish` —
  > PR #8
