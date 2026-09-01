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

**`.scratch/` is not a review target.** On the `feat/dataseq-merge` branch, `.scratch/` holds
the three existing `dataseq` implementations imported for side-by-side comparison, together
with our own account of them and a Python transliteration of the Lush original. **That
transliteration is deliberately idiomatic rather than literal** — it reproduces the original's
*implementation decisions* in ordinary Python instead of transliterating Lush constructs, and
every departure carries a `DEVIATION` comment naming the original behaviour and the reason.
So do not read a divergence from the `.lsh` as infidelity: fidelity is claimed for the
decisions, not the constructs, and `ACCOUNT.md` is where the original is described. The
priorities above — the encode seam most of all — generate noise when applied to any of it: a
finding against scratch input is a finding against the thing being compared, not against
anything that ships. Nothing in `.scratch/` is part of any
distribution and nothing outside it may import from it. **It is no longer deleted when this
branch merges** (changed 2026-08-31): the same imports seed `hmm` and `align` 0.1.0, so the
tree is retained and each import's `.gitignore` is re-scoped as the migration target changes.
`.scratch/align-poc/.gitignore` is written in explicit phases for that reason, with only its
`dataseq` block active — so a file that is present on disk but untracked there is scoped out
of the current phase, not overlooked.
Review the merged result under `packages/pfsmgraph-dataseq/`, never the inputs. The directory
states its own lifetime in `.scratch/README.md`.

**What arrives there is a whole third-party working tree, not an extracted module.** The merge
base is imported as `.scratch/dl/MelodyHPO/` — a defunct standalone project, of which
`melody_hpo/data/` is the part being merged and the rest (models, training, evaluation,
generation) is context. Two of the high-signal targets above will match inside it and must not:
its `pyproject.toml` is *not* one of the `packages/*/pyproject.toml` whose dependency bounds are
literal claims about PyPI — it belongs to a project that is not published and not built here —
and its `tests/` is not a backend-parameterized suite under ADR 0003, so measuring it against
that standard produces findings against code that ships nowhere. Anchor both targets on
`packages/`, and treat any path under `.scratch/` as out of scope by default.

Note also that **each import carries its own deny-by-default `.gitignore`**, and each admits a
small fraction of what is on disk: `.scratch/dl/` tracks 34 files out of 2.2 GB,
`.scratch/hmm-lush/` 143 out of 929 MB, `.scratch/py-rudimentary/` 73 out of 1.7 GB, and
`.scratch/align-poc/` 9 out of 194 MB, plus two documents of our own at the `.scratch/` root
(`README.md` and `RESERVED-BLOCK.md`). The per-import counts include our own written analysis,
which lives alongside the source it describes. *(Counts measured 2026-08-31; the previous
figures for the first two were each high by one.)* If something in an imported tree looks conspicuously absent, that is the
intended behaviour and not a finding — the exclusions carry their reasons inline in each of those
files, and what they turn away is overwhelmingly not source: virtualenvs and tool caches in the
first, saved model checkpoints from 2008–2011 training runs in the second.

Two exclusions in `hmm-lush` are worth knowing before reading, because both look like gaps in the
translation record and are not. `Code/SeqData/C/` is absent because every `.c` in it opens
`WARNING: Automatically generated code ... by the DH compiler` — Lush's own compiler emitting C
from the `.lsh` beside it, so it is a build artefact rather than a hand-written fast path, and
there is no Python/C equivalence to check there. And `Code/_Old Lisp Code/` is absent because it
is an interpreted Common Lisp predecessor that the owner has ruled out as a source; the Lush
version under `Code/` is the original the translation must be faithful to.

**`.scratch/py-rudimentary/` holds two repositories, and the asymmetry between them is
deliberate.** `segalign/` is the third `dataseq` implementation; `SegAlign-Draft/` is the
predecessor it was refactored from, and exactly one file of it is tracked
(`glob/ss2_alignment.py`). That is not an incomplete import. What the draft contributes is a
*negative* — it has no sequence abstraction at all — and one signature taking `List[Any]` is
what makes that claim checkable after `.scratch/` is deleted. Its `tcoffee/` package (six
modules of T-Coffee multiple alignment) is absent for the same reason `Training/` is absent
from `hmm-lush`: it is real work, but it is `pfsmgraph-align`'s scope rather than `dataseq`'s.

The tracking bar is also higher here than for `hmm-lush`, and the reason is provenance, not
importance: `hmm-lush` was under no version control and was irreplaceable, whereas both trees
here are clean checkouts of live GitHub repositories at revisions recorded in
`.scratch/README.md`. Anything turned away costs one `git clone`. One caveat that file records
and a reviewer would otherwise not see: `segalign`'s working copy is **dirty** at `ca97809` in
`glob/needleman_wunsch.py`, so that one tracked file will not match GitHub. It is tracked only
because `src/segalign/__init__.py` imports `glob`, without which the merge target does not
import and its tests do not run.

What *is* worth reviewing there is the written comparison rather than the code: if the semantic
account of the Lush original and the translation beside it disagree, that is a real finding, and
the account wins. It was written first precisely so the comparison could not be run against our
reading of the original instead of the original.

This generalises to every written analysis under `.scratch/` — `.scratch/dl/ANALYSIS.md` was
the first, joined by `.scratch/hmm-lush/ACCOUNT.md` and `COMPARISON.md`, and by whatever
goal 4 writes for `py-rudimentary` and `align-poc`.

**One caution specific to `.scratch/align-poc/`.** It holds `tokalign`, the proof-of-concept
alignment library that PRD §1.2 and ADRs 0001–0004 were written *from*. It is therefore not
evidence to be weighed against those records the way the other three imports are — it is
their source. Where it and an ADR disagree, the ADR is still the later word (`Alphabet` puts
user symbols at 4; ADR 0011 moves them to 6), but a divergence there is a deliberate
renumbering rather than a defect, and `Alphabet` already satisfies ADR 0011 on strictness,
`gap_index` and `decode`. Those documents are
load-bearing in a way the imported code is not: they decide what `packages/pfsmgraph-dataseq/`
becomes and they are the draft of ADR 0010's decision section, so a claim in one that the code
does not support is a real finding even though the code it describes ships nowhere. Check the
claims, not the style.

**One structural point about `COMPARISON.md`, because it governs which findings against it are
valid.** Its §2 lists the places the `dl` base must be overridden; its §3 lists divergences that
are deliberately *not* adopted, because an Accepted ADR settles them. The split is load-bearing:
the ADRs outrank every imported source — the three `dataseq` implementations, the base
included, and `align-poc/tokalign` (see `AGENTS.md` under "Invariants") — so "the merge
ignores Lush's X" is a finding only when X falls in §2. Against §3 it is the intended
behaviour, most visibly for the reserved block, where the implementations disagree with each
other *and* with [ADR 0011](../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md),
and the ADR wins regardless of what any of them does. If a §3 item looks wrongly classified,
that is worth raising; that it was not adopted is not.

**Two counting caveats, because both are easy to file as errors and neither is one.** First,
there are **four imported sources but three `dataseq` implementations**: `tokalign` is an
alignment library contributing the *encoder* half, which is why ADR 0010 is titled
"merging-three-implementations" while its text separately requires reconciling `Alphabet`.
A document saying "three implementations" is therefore correct, not stale. Second, the
precedence rule reads backwards for `tokalign` specifically — ADRs 0001–0004 were written
*from* it, so its divergences are overwhelmingly later decisions rather than defects, and it
is the only source with a real `gap_index`, strict-by-default encoding and a `decode` at all.
Reporting it as the most deviant of the four inverts the actual picture.

That caveat protects deliberate divergences, **not** everything in the file. Goal 4 found two
real defects there, and both are worth confirming rather than dismissing: `RESERVED_INDICES`
is annotated as a dataclass field rather than a `ClassVar`, so the reserved block ADR 0011
fixes is a positional constructor argument; and `decode` raises `KeyError` on every reserved
code, because `_idx_to_sym` is built from the gap index up. Neither is a decision — one is an
annotation slip, the other an unfinished table — which is exactly what distinguishes them
from the divergences the caveat covers. Both are recorded in
`.scratch/align-poc/COMPARISON.md` §3.

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
