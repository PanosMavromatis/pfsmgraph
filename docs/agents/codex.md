## Codex as cross-provider reviewer for pfsmgraph

This file is Codex-specific. `AGENTS.md` (generated from `docs/agents/core.md`) is the
shared base every agent reads; this file is the higher-precedence override and states
only what changes for Codex. Claude Code is the primary implementer here — it holds the
PRD/ADR context and writes the algorithms. Codex's value is that it is a *different model
family* reading the same code: it catches what a single provider's blind spots let
through, and it is not the author, so it has no stake in defending a design.

**Codex's role in this repository is review only.** Do not implement. When a review turns
up a defect, describe the fix and where it belongs; do not write it. If the user explicitly
asks for an implementation anyway, see "If explicitly asked to implement" below.

### Primary role

Cross-provider reviewer. Priorities, in order:

1. **Correctness of dynamic-programming recurrences** — off-by-one in traceback, affine
   gap state, boundary rows, and any disagreement between a fast backend and the pure-Python
   oracle. ADR 0002 makes the Python phase the executable specification; when two phases
   disagree, the Python one defines the answer and the fast one is the bug.
2. **Boundary conditions at the encode seam** — ADR 0001 requires strings→ints at every
   public entry and ints→strings at exit. Empty sequences, single-symbol sequences,
   sequences of all-`GAP`, and reserved codes appearing in user data are the cases that
   slip through.
3. **Performance regressions in hot paths** — typed-memoryview access patterns, accidental
   Python object access inside a Cython inner loop, host↔device transfers inside a kernel
   launch loop.
4. **Overengineering relative to the current phase.** This repo is early. A Phase-1 pure-Python
   reference implementation that has grown a backend-dispatch layer, a plugin registry, or
   premature vectorization is a finding, not a bonus. ADR 0002 is explicit that no phase
   begins before the previous one is correct.

Push back on design, not just code. If an ADR's stated consequence does not match what the
code actually does, say so — that divergence is worth more than a style comment.

### High-signal review targets

**Today, the repo is scaffolding: no algorithms, no tests.** Until code lands, the highest-
signal targets are documentation and packaging coherence, which is where errors are currently
cheapest to fix and most expensive to leave:

- **`docs/design/adr/` vs. `docs/design/PRD.md` vs. `docs/agents/core.md`.** Three documents
  describe one design. Claim drift between them is the live risk — the ADRs are authoritative
  where they overlap the PRD, and `core.md` must not contradict either. Check specifically that
  ADR statuses (`Proposed` vs. `Accepted`) still match reality; ADR 0010 is `Proposed` pending
  the `dataseq` encoder API.
- **`packages/*/pyproject.toml` dependency bounds.** ADR 0006's workspace footgun: a
  `{ workspace = true }` source satisfies *any* constraint, so a wrong `>=` bound cannot fail
  locally and only breaks a pip user post-publish. Every intra-family bound currently reads
  `>=0.1` against `0.0.0` placeholders. Local green proves nothing here — read the bounds as
  literal claims about PyPI.
- **`docs/plan/DEFERRED.md` trigger integrity.** Several entries must land *as part of* their
  trigger, not after (the reserved-block renumbering with the `dataseq` merge; the `_cython.pyx`
  comma-form indexing fix before the file is copied). A change that fires a trigger without
  discharging its entries is a finding.

**As implementation lands, these become the targets** — in the order the phases arrive:

- `packages/pfsmgraph-dataseq/src/` — the encoder. Reserved block is fixed and non-configurable
  (`PAD`=0, `UNK`=1, `BOS`=2, `EOS`=3, `GAP`=4, `MSK`=5; user symbols from 6). `PAD` must be 0
  or PyTorch zero-fill silently means something other than "absent". Encoding is strict by
  default; any `UNK` fallback that is not explicitly opted into is a bug. Watch for the
  proof-of-concept's old allocation-from-4 surviving the merge.
- `packages/pfsmgraph-{align,hmm}/src/**/_cython*.pyx` — typed memoryviews with
  `boundscheck(False)` / `wraparound(False)`. Bounds checking is *off*, so an index error is
  memory corruption, not an exception. Verify comma-form indexing (`M[i-1, j-1]`, never
  `M[i-1][j-1]` — the bracket form materializes an intermediate 1-D view per access).
- `**/_cuda*.py` — Numba CUDA anti-diagonal wavefront. Anti-diagonal indexing arithmetic,
  the two-preceding-diagonal dependency, boundary diagonals, and synchronization between
  wavefront steps. This is the single highest-payoff review surface in the project and the
  one where a wrong answer is least likely to announce itself.
- **Backend-parameterized test suites** (ADR 0003) — one suite per algorithm, backend as a
  fixture parameter. Findings to look for: a test that quietly exercises only one backend,
  an equivalence assertion with a hand-written expected value rather than the Python oracle,
  and any backend that is implemented-but-unimportable being *skipped* rather than failing.
- **`meson.build` / editable-install interaction** when `align` and `hmm` return to
  meson-python (ADR 0012). meson-python's import hook claims the whole `pfsmgraph` PEP 420
  namespace and shadows its siblings. Any change here must be verified by actually importing
  all five subpackages after `uv sync`, not by reading the config.

**`hseg` is deliberately absent from this list.** It has the largest design gap of the five
packages and will get its own sidecar once that settles; do not improvise review criteria for
it from this file in the meantime.

**Lower-priority targets.** Claude Code handles these reliably; do not spend review budget
on them unless something looks actively wrong: prose style and structure in `docs/`, ADR
formatting and index-row bookkeeping, commit message shape, and general Python idiom in
non-hot-path code.

### Review output conventions

- **GitHub PR review** — inline suggestion blocks anchored to the changed lines, plus one
  summary comment leading with the blocking findings. Do not restate the diff.
- **Working-tree review** — write `REVIEW.md` at the repo root, organised by severity:
  **blocking** / **important** / **nice-to-have**, every finding carrying a `file:line`
  reference. `REVIEW.md` is gitignored; it is scratch output, never committed.
- **Quick spot check** (`codex exec`) — terse terminal output, findings only, no preamble
  and no summary of what the code does.

In all three: state the failing input or state that triggers a correctness finding. A
finding that cannot name a concrete case it breaks is a suggestion, and should be labelled
as one.

### If explicitly asked to implement

Review-only is the standing rule; this section covers the exception, not the norm. If the
user overrides it for a specific change, `AGENTS.md` carries the shared conventions and
commands, and these additionally constrain any code written here — they are non-negotiable
without amending the PRD:

- **Never create `pfsmgraph/__init__.py`** at the namespace level, in any package. It breaks
  every other distribution's imports.
- **Respect the phase order.** Pure Python must exist and be correct before Cython; Cython
  before CUDA. Do not write a fast path for a recurrence that has no reference implementation.
  Every phase is retained and shipped, never replaced.
- **Encode at the boundary.** Inner computation is integer-only. If a proposed change puts a
  string type inside a kernel, the change is wrong.
- **One test suite per algorithm, parameterized over backends.** Add a parameter value, never
  a second test file.
- **`uv run pytest`** is the runner; `uv sync` after any dependency change. Do not invoke `pip`
  or edit `uv.lock` by hand.

Check `docs/plan/DEFERRED.md` before touching any of its named triggers — several entries must
land as part of the trigger rather than after it.

### Do not edit generated files

`AGENTS.md` and `AGENTS.override.md` are **generated artifacts**, built from `docs/agents/core.md`
and `docs/agents/codex.md` respectively by the workflow-claude plugin's `/agents-docs-build`.
`/agents-docs-update` also refreshes them, but delegates the actual build to `/agents-docs-build`
— there is no second build path. `CLAUDE.md` is a hand-authored `@import` dispatcher and is not
regenerated by either.

Edit the sources under `docs/agents/`, then re-run the build. Never edit `AGENTS.md`,
`AGENTS.override.md`, or `CLAUDE.md` directly — a `PreToolUse` hook blocks it, and an edit that
slipped through would be silently overwritten by the next build. `/agents-docs-check` reports
drift between sources and artifacts.

### Gotchas

- **A green local test run proves nothing about published dependency bounds.** See the workspace
  footgun above; this is the one project-specific trap most likely to catch a reviewer reasoning
  from "the tests pass".
- _(more to be added after the first Codex review session)_
