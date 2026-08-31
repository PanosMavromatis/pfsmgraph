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
- **Confirm `dataseq`'s build backend.** Its `pyproject.toml` presumes pure-Python
  (hatchling) on the strength of the `dl`-derived base; switch to meson-python only if a
  compiled inner loop is found ([ADR 0008](../design/adr/0008-per-package-build-backends.md)).
  Provisionally holds (2026-08-31): the `dl` base is pure Python over pandas and torch,
  with no compiled inner loop. Re-check after the Lush and rudimentary imports.
- **Fix `dataseq`'s third-party runtime dependencies.** `dependencies = []` is a
  placeholder — whether `numpy`, `torch`, or neither belongs there is determined by what
  the merge base actually needs.
  Provisionally **neither** (2026-08-31). `torch` leaves with the dataset views and
  `pandas` with ingestion, neither of which belongs in the base layer; and
  `torch.utils.data.Dataset` is duck-typed, so a stock `DataLoader` works against a
  container that never imports torch. `numpy` may earn its way in for the code arrays, but
  nothing in the base requires it. Recorded because "no change needed" is otherwise
  indistinguishable from "not yet looked at". See `.scratch/dl/ANALYSIS.md` §4.
- **Write the first test suite to the ADR 0003 standard,** including the
  `pytest_report_header` hook that prints the backend matrix. `addopts = "-ra"` is
  already configured in the root `pyproject.toml`; the header hook is the half that
  still has nowhere to live.

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

## Trigger: the first real release

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
