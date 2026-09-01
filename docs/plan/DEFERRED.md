# Deferred actions

Work that is **decided but not yet actionable**, each waiting on a specific trigger.
This is not a backlog of ideas and not a list of open design questions — an entry
belongs here only when the decision is already made and something concrete must happen
when its trigger fires.

Open *questions* live in [PRD §8](../design/PRD.md) and in the `Open` sections of the
[ADRs](../design/adr/README.md). Where an item below has an authoritative record
elsewhere, that record wins; this file is an index by *trigger*, not a second source of
truth.

---

## Trigger: the `dataseq` merge

The next piece of work in PRD order. It carries the most freight of any trigger here,
and several of these must land *as part of* the merge rather than after it.

- **Promote [ADR 0010](../design/adr/0010-dataseq-composition-merging-three-implementations.md)
  from `Proposed` to `Accepted`.** It is `Proposed` only because the encoder API
  reconciliation is unresolved: the constructor signature, the spelling of the
  strictness switch, and how `align` consumes the mapping at its boundary. Settle those
  during the merge, then update the ADR and the index row.
  **Settled (2026-09-01), and the entry is closed.** All three were settled and
  implemented: `SymbolTable(symbols)` with `from_sequences`; per-call
  `encode(..., on_unknown="raise" | "unk")`; and `code()` plus a `sym_to_code` read-only
  mapping published as cross-distribution API. The Q&A is recorded under goal 6 in
  `docs/plan/feat-dataseq-merge/TODO.md`. The settled API is written into the ADR's
  decision section, its `## Open` section has become `## Resolved`, and the index row and
  its footnote in `docs/design/adr/README.md` read `Accepted`.
- **Renumber the proof-of-concept alignment code to the reserved block.** The
  proof-of-concept allocates user symbols from 4, with a different gap index;
  [ADR 0011](../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md)
  requires user symbols from 6. Nothing is persisted yet, so this is a code-only change
  — **but it must land as part of the merge, not after**, and every hard-coded index
  assumption in the alignment code needs auditing. Deferring it past the merge is how it
  becomes a data migration instead of an edit.
  **Wider than it reads (2026-08-31).** The `dl` merge base allocates user symbols from
  **3** (`PAD` 0, `BOS` 1, `EOS` 2), a *second* wrong offset distinct from the
  proof-of-concept's 4, and it is missing `UNK`, `GAP`, and `MSK` entirely — `GAP` being
  the symbol `align` exists to produce. So this entry covers at least two implementations,
  and the Lush one may add a third. Its padding is also written as the literal `[0]`, never
  as `PAD`, so it agrees with ADR 0011 only by coincidence. See
  `.scratch/dl/ANALYSIS.md` §3.1 and §2.3.
  **Settled (2026-09-01), and the entry is closed.** It took two branches, and the
  note above is why. The `dl` half was satisfied inside the merge itself: the container
  that landed in `packages/pfsmgraph-dataseq/` hard-codes the block in `_reserved.py` as
  module constants, so the base's user-symbols-from-3 offset never reached `main`
  (PR #2). The proof-of-concept half is `refactor/reserved-block-renumber` --
  `tokalign`'s `Alphabet` now puts `GAP` at 4 with user symbols from 6, and its
  `RESERVED_INDICES` field is *gone* rather than renumbered, because it was a plain
  dataclass field and therefore a constructor parameter, which ADR 0011 forbids. The
  audit found one real defect: `ScoringMatrix.identity` zeroed
  `range(RESERVED_INDICES + 1)`, an expression encoding gap-sitting-just-past-the-block,
  which after the move would have blanked the first *user* symbol's scores in silence.
  The "the Lush one may add a third" guess was right -- Lush allocates user symbols from
  2 -- but nothing here fixes it: that tree is unmigrated `.scratch/` source, and its
  renumbering happens inside the `hmm` translation, not before it. One piece was carved
  out rather than done: making `tokalign`'s `decode` total, filed below under the `align`
  migration.
- **Confirm `dataseq`'s build backend.** Its `pyproject.toml` presumes pure-Python
  (hatchling) on the strength of the `dl`-derived base; switch to meson-python only if a
  compiled inner loop is found ([ADR 0008](../design/adr/0008-per-package-build-backends.md)).
  **Settled (2026-08-31): hatchling holds, and the entry is closed.** All four sources were
  read; none has a compiled inner loop that belongs to `dataseq`. The landed container is
  pure Python over numpy. `tokalign` does carry Cython, but in its alignment algorithms —
  that is `align`'s migration and `align`'s build backend, not this one's.
- **Fix `dataseq`'s third-party runtime dependencies.** `dependencies = []` is a
  placeholder — whether `numpy`, `torch`, or neither belongs there is determined by what
  the merge base actually needs.
  **Settled (2026-08-31): `numpy` only, and the entry is closed.** `dependencies =
  ["numpy>=1.24"]`. The provisional "neither" holds for `torch` and `pandas` — torch left
  with the dataset views and pandas with ingestion, and a stock `DataLoader` is verified to
  work against a container that never imports torch. `numpy` earned its way in as predicted:
  codes are `int32` arrays, which is what `align` and `hmm` index dense matrices with in DP
  inner loops and what a Cython or CUDA buffer wants. See `.scratch/dl/ANALYSIS.md` §4 and
  the goal-5 Q&A in `docs/plan/feat-dataseq-merge/TODO.md`.
- **Write the first test suite to the ADR 0003 standard,** including the
  `pytest_report_header` hook that prints the backend matrix. `addopts = "-ra"` is
  already configured in the root `pyproject.toml`; the header hook is the half that
  still has nowhere to live.
  **Partly done, and the remainder re-triggered (2026-08-31).** `dataseq` now has 74 tests,
  the first in this repository. The backend-matrix half is **not** done and should not be:
  ADR 0003 parameterises over backends for dynamic programming, and `dataseq` is a container
  with no DP algorithm and so no backends — under that ADR a lifecycle phase not yet reached
  contributes no parameter at all, so there is nothing for a header to report. The hook still
  has nowhere to live, and its real trigger is **the first `.pyx`**, where a second backend
  first exists. One narrow skip does exist here, in `tests/test_torch_interop.py`: torch is
  not a dependency, so the `DataLoader` integration skips when it is absent. It covers the
  integration only, never the container's own behaviour.

## Trigger: the first `.pyx`

- **Revert `align` and `hmm` to meson-python, and solve the namespace/editable
  interaction.** This is the substance of
  [ADR 0012](../design/adr/0012-align-and-hmm-temporarily-on-hatchling.md): meson-python's
  editable hook claims the whole `pfsmgraph` namespace and shadows its siblings, and two
  meson editables also conflict with each other. Three candidate resolutions are
  recorded there, none chosen — the information needed to choose arrives with the first
  kernel. The verbatim revert recipe lives in each package's `pyproject.toml`; the root
  `dev` group needs `meson-python`, `cython`, and `ninja` restored at the same time.
- **Apply the deferred `_cython.pyx` indexing fix while moving the file** (PRD §11.1).
  Every double-bracket access on the typed 2-D memoryviews `M`, `X`, `Y`, `T` becomes
  comma-form: `M[i - 1][j - 1]` → `M[i - 1, j - 1]`, `T[i][j]` → `T[i, j]`, `M[m][n]` →
  `M[m, n]`. The bracket form materializes an intermediate 1-D view per access. Pure
  performance, no behavioral change, safe to apply blind. Leave the already-correct
  comma-form (`scores[seq_a[i - 1], seq_b[j - 1]]`) and the 1-D memoryviews alone.
  **Do it before the file is copied** — PRD §11 designates it the reference template for
  every future kernel and wavefront pass.
- **Pin the `numba-cuda` lower bound** in `align`'s (and `hmm`'s) `[gpu]` extra once the
  wavefront backend lands; it is currently unpinned.

## Trigger: CI existing

- **Wire `PFSMGRAPH_REQUIRE_BACKENDS` into the GPU job.**
  [ADR 0003](../design/adr/0003-one-parameterized-test-suite-per-algorithm.md) defines it
  — a comma-separated list escalating an absent-hardware skip into a failure — but
  nothing sets it today. Without it, a runner that loses its CUDA device degrades to a
  *green* run whose header nobody reads. Note that its first real use will also be its
  first test: no code path currently exercises it.
- **Attach trusted publishers to the six existing PyPI projects.** Per
  [ADR 0005](../design/adr/0005-namespace-prefix-and-pep-420-layout.md), the correct
  order is claim-by-placeholder first (done), then attach a normal trusted publisher to
  the now-existing project when CI is ready. A pending publisher reserves nothing, so
  there was no point doing this earlier.
- **Run the agent-docs sync check in CI.** `AGENTS.md` and `AGENTS.override.md` are
  generated from `docs/agents/` but committed, so drift between source and artifact is
  silent: a stale artifact still parses, still reads plausibly, and makes agents act on
  outdated instructions while every test stays green. The check that catches it exists
  (`check-agents-md.sh`, surfaced as `/agents-docs-check`) and passes today. **The wrinkle
  is that it lives in the workflow-claude plugin under the untracked `.claude/`, not in
  this repo, and CI cannot invoke a slash command** — so wiring it up means vendoring the
  script, installing the plugin in the job, or reimplementing the regenerate-and-diff in a
  few lines of shell. Decide which when the workflow exists; the entry is here so the
  question is not rediscovered by a confusing review months later.
- **Turn the executed-examples rule in `docs/api/` into a doctest run.**
  [ADR 0013](../design/adr/0013-api-documentation-layout-and-tooling.md) requires every
  code block in `docs/api/` to be executed and its output pasted from the run, and that
  rule is currently a discipline with nothing enforcing it — which is the single largest
  cost of choosing hand-written Markdown over a generator. The examples are already
  written as `>>>` blocks, so `pytest --doctest-glob='*.md'` is most of the way there.
  **Two things must be settled first.** Several blocks show tracebacks, and one in
  `encoder.md` deliberately shows the *escaped* form of a `KeyError` message — because
  `KeyError.__str__` is `repr(args[0])` — so `IGNORE_EXCEPTION_DETAIL` and the other
  doctest option flags have to be chosen rather than defaulted. And the blocks currently
  omit their shared setup, which reads better for a human and does not run: a doctest pass
  needs that setup restored or supplied by a fixture. Neither is hard; both are decisions,
  and making them badly would produce a check that passes while proving nothing.
- **Check documented repo-state *counts* against the tree.** A recurring rot has a
  mechanical half worth automating. Prose in `README.md`, `docs/agents/core.md`,
  `docs/design/PRD.md` and the ADR index makes assertions about the repository that are
  true when written and silently false later. Observed so far, all of them caught by hand
  and none by a test: "63 tests" after the count reached 74; "the twelve initial ADRs"
  after 0013 landed; "Two carry a non-`Accepted` status" after 0010 was promoted; "every
  package is an empty namespace subpackage" for two commits after `dataseq` shipped; and
  the ADR index's reading-order and PRD-coverage notes, which enumerated up to 0012 and
  quietly stopped being exhaustive.
  **The tractable subset is counts and existence claims**, each checkable in a line or
  two: test count from a pytest run; ADR count from the file list; ADR statuses by
  grepping each record's `**Status:**` against its index row; per-package implementation
  status from whether `src/pfsmgraph/<pkg>/` holds anything but `__init__.py`; tag
  existence from `git tag`. The shape to copy is `check-agents-md.sh` — assert, diff,
  exit non-zero — and it inherits that script's blocker verbatim: it lives in the
  workflow-claude plugin under the untracked `.claude/`, so wiring it up means vendoring,
  installing the plugin in the job, or reimplementing in shell.
  **What this does not cover, and must not be claimed to:** semantic claims. "`SymbolTable`
  is the provisional encoder implementation" was false in exactly the same way and no
  count-checker would have found it. Those need the reading sweep filed under "the first
  real release"; the two entries are halves of one problem and neither substitutes for the
  other.

## Trigger: the first real release

- **Settle ADR 0003's sdist/wheel question before the first sdist publishes.** Measured
  during the backend-header branch: sdists ship `tests/` without either half of the ADR
  0003 mechanism, so a packager's run is silent in exactly the way the policy forbids.
  The record now states the measurement and the three candidate remedies; what it cannot
  do is pick one. Publishing an sdist is what makes the choice visible, which is why it
  is filed here rather than left to be noticed.
  **Settled (2026-09-01), and the entry is closed.** The mechanism is repo-local by
  decision: tests keep shipping in the sdist, and ADR 0003 §Resolved now states that
  neither `addopts = "-ra"` nor `pytest_report_header` travels with them, together with
  what that costs. Both self-sufficiency remedies were rejected in the record — excluding
  `tests/` removes the run exactly where a distro packager most wants it, and duplicating
  the mechanism five ways is this ADR's own drift failure committed on the policy instead
  of on the tests. The obligation to revisit is re-filed under the `align` migration,
  because `align` is the first member whose matrix will have a row in it. The Q&A is
  recorded under goal 1 in `docs/plan/chore-release-dataseq-0.1.0/TODO.md`.
- **Set honest version lower bounds.** Every intra-family dependency currently reads
  `>=0.1`, against distributions whose only published version is a `0.0.0` placeholder.
  This is the workspace footgun of
  [ADR 0006](../design/adr/0006-single-repository-as-a-uv-workspace.md): a
  `{ workspace = true }` source satisfies *any* constraint, so a wrong bound never fails
  locally and only breaks a pip user after publish. Set real bounds when `dataseq` and
  `align` first publish, and **revisit on every breaking change** — this one recurs
  forever rather than clearing.
- **Replace the `0.0.0` placeholders within a reasonable window.** PEP 541 treats
  content-free projects as somewhat more reclaimable, and the account email must stay
  reachable. Release order is forced by the dependency graph: `dataseq` → `align` →
  {`hseg`, `hmm`, `dl`}.
- **Do not add dependency declarations to the placeholders** before then. A stub
  declaring `pfsmgraph-dataseq>=0.1` cannot resolve, because no such version exists.
- **Drop the `.dev0` suffix and tag the release, per package.** All five members declare
  `0.1.0.dev0`; the release commit for a package changes only that package's version to
  `0.1.0` and is tagged `pfsmgraph-<pkg>-v<version>` — `pfsmgraph-dataseq-v0.1.0`. Hyphen,
  not slash: git refs are paths, so `pfsmgraph-dataseq/v0.1.0` cannot coexist with a plain
  `pfsmgraph-dataseq` tag, and some tooling mishandles the nesting. `git tag --list
  'pfsmgraph-dataseq-*'` then gives one package's release history. Tagging is manual — no
  command in use here creates a per-package tag (see `docs/agents/claude.md`). Keeping
  `.dev0` until that commit is deliberate: a bare `0.1.0` on an incomplete package means an
  accidental `uv build` + publish burns `0.1.0` on PyPI permanently, since versions are
  immutable and deleting a release does not free the number.
- **Sweep the prose claims about repository state, semantically.** The release is when
  `README.md`, the PRD, and the ADR index are first read by people with no other source of
  truth, so it is the deadline for a class of rot that has recurred on every branch so far:
  a sentence describing the state of the repo, true when written, falsified by later work,
  and caught only when someone happens to reread it. The front page carried "No algorithms
  are implemented yet — every package is an empty namespace subpackage" for two commits
  after the container landed.
  **This is the half no script can do.** Its companion under "CI existing" covers counts
  and existence claims; what remains is meaning — a surface described as provisional after
  it was settled, an ADR called `Proposed` after promotion, a reviewer instruction whose
  reference example has gone stale and now teaches the reviewer to file the correct state
  as a defect (`docs/agents/codex.md` did exactly this), a future-tense promise stranded
  after it was kept ("at which point 0010 is promoted"), and a dated header made internally
  false by updating a number inside it.
  **Read for claims, not for prose quality**, over `README.md`, `docs/agents/*.md`,
  `docs/design/PRD.md`, `docs/design/adr/README.md` and each ADR's `Status`, and
  `docs/api/`. The question for each sentence is only "was this true when written, and is
  it true now" — the two differ, and where they do the fix usually preserves the history
  rather than erasing it, as 0010's index footnote does.
  **Why the release rather than sooner:** the cost of a stale internal note is an agent
  briefly misled, and it is corrected on contact; the cost of a stale README at release is
  paid by readers who cannot tell. If a second repo-hygiene item appears before then, this
  is better promoted to its own revision via `/open-revision` than left waiting.

## Trigger: the `align` migration

- **Make `tokalign`'s `decode` total.** It raises `KeyError` on every reserved code,
  because `_idx_to_sym` is built from the gap index up and never populates `PAD`, `UNK`,
  `BOS`, `EOS` or `MSK`. `dataseq` decodes *totally* over `range(size)` for a specific
  reason — a padded batch is the array most likely to be decoded — so the two halves of
  the family currently disagree at exactly the seam `align` sits on. Recorded as a defect
  rather than a divergence in `.scratch/align-poc/COMPARISON.md` §3 and in
  `docs/agents/codex.md`.

  Deliberately out of scope of `refactor/reserved-block-renumber` (2026-09-01): making
  `decode` total is a behaviour change, not a renumbering, and it needs a decision the
  renumbering did not — what the reserved codes decode *to*. `dataseq` answers that with
  `RESERVED_SYMBOLS`; whether `tokalign` should mirror those strings or use its own is
  the open part.

- **Revisit whether the ADR 0003 mechanism should travel in the sdist.** Settled
  2026-09-01 as repo-local ([ADR 0003](../design/adr/0003-one-parameterized-test-suite-per-algorithm.md)
  §Resolved), on the grounds that `dataseq` registers no backend, so a suite run from its
  sdist has an empty matrix to not-print and loses nothing by not printing it. `align` is
  where that stops being true: the first DP kernel to reach ADR 0002 phase 1 puts a real
  row in the matrix, and from then on an sdist run can pass green while a backend that is
  implemented but not importable is simply absent — the case the root `_backends.py`
  turns into a hard failure. Re-open the choice then, with both rejected remedies on the
  table — exclude `tests/`, or duplicate the mechanism per member — alongside whatever
  `align`'s shape suggests. Nothing is owed before that.

## Trigger: `align` acquiring a backend-selection API

- **Write the runtime backend-selection ADR** (next free number). What the public API
  does when a caller explicitly requests an unavailable backend — raise, or fall back to
  the next-fastest — is user-facing and deliberately out of scope for
  [ADR 0003](../design/adr/0003-one-parameterized-test-suite-per-algorithm.md), which
  governs the test suite only.
  [ADR 0011](../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md) has
  taken a house position on the general shape of the question — strictness, on the
  grounds that silently absorbing a problem produces work that merely looks fine — but
  it has not been applied here.

## Trigger: a vocabulary outliving the process that built it

- **Vocabulary persistence (`save`/`load`).** A `SymbolTable` is built from a corpus and
  is immutable thereafter, so today it is rebuilt wherever it is needed. That holds only
  while every consumer sees the same corpus: an expressible train/test split, or `align`
  scoring sequences encoded in another process, needs the *same* table rather than an
  equivalent one — and first-appearance ordering makes "equivalent" depend on iteration
  order, so rebuilding is not a safe substitute.

  Deferred rather than dropped because it carries an undecided sub-question that would
  otherwise have been settled as a side effect of the encoder API: **the escaping rule**.
  Symbols here are multi-character and arbitrary, so any line- or delimiter-oriented
  format has to say what happens to a symbol containing the delimiter — the problem
  `.scratch/hmm-lush/COMPARISON.md` §2.3 records the Lush original solving by fiat. JSON
  sidesteps escaping but commits the format; that trade is the decision, and it deserves
  to be made deliberately.

## Trigger: a corpus large enough for code locality to matter

- **`SymbolTable.from_frequencies()`, offered but never default.** Frequency ordering
  puts common symbols at low codes, which is what the rudimentary `segalign` did
  (`.scratch/py-rudimentary/COMPARISON.md` §2.3) and what makes truncated vocabularies and
  cache locality work. It is deliberately *not* the default and must never become one:
  it makes every code a function of the whole corpus, so adding one file renumbers the
  alphabet and silently invalidates every previously encoded sequence. First-appearance
  ordering is stable under corpus growth, which is why it is the constructor's rule.

  This wants to arrive as a separate classmethod alongside `from_sequences`, so that
  choosing it is visible at the call site rather than being a flag on the ordinary path.

## Trigger: `hseg` design settling

- **Confirm `hseg`'s build backend.** hatchling holds while `hseg` is pure orchestration
  over `align`; switch to meson-python if it turns out to have its own DP recurrence
  (PRD §8). `hseg` has the largest design gap of the five packages, so this is the
  least-determined item here.

## No trigger yet — revisit deliberately

These have no event that will surface them. They need to be looked at on purpose.

- **Whether `hseg`, `hmm`, and `dl` depend on each other** (PRD §8,
  [ADR 0009](../design/adr/0009-dataseq-as-the-base-layer.md) `Open`). The base and the
  common mid-layer are settled; the top of the graph is not. This determines how much
  the atomic-commit benefit of the workspace is actually worth.
- **Whether tests ship in the sdist or wheel** ([ADR 0003](../design/adr/0003-one-parameterized-test-suite-per-algorithm.md)
  `Open`). If they do, distro packagers and anyone running `pytest --pyargs` inherit the
  unavailable-backend policy, which makes a developer-facing decision weakly
  user-visible.
- **How the Claude Code development plugin fits the multi-package family** — one
  family-wide dev plugin, or one per package. Explicitly out of scope for the PRD (§10)
  and never discussed.
