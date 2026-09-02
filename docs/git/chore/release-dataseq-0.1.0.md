# chore/release-dataseq-0.1.0

**Created**: 2026-09-01
**Base**: main at 3a66581
**Status**: active

## Purpose

Publish `pfsmgraph-dataseq` 0.1.0 to PyPI, replacing the content-free `0.0.0`
placeholder, and close the last open subgoal of revision `01-dataseq-v0.1.0`. This is
the first artifact of this project that anyone outside the repository can install, and
the first one whose mistakes cannot be taken back: PyPI versions are immutable and
yanking does not free the number.

## Scope

- Settle ADR 0003's sdist/wheel question — three candidate remedies are recorded, none picked
- Semantic sweep of the prose claims about repository state, ahead of first outside readers
- Honest lower bounds on the intra-family dependencies naming `pfsmgraph-dataseq`
- Drop `.dev0` from `pfsmgraph-dataseq` only, build, publish, and tag `pfsmgraph-dataseq-v0.1.0` by hand

## Context

- `docs/plan/TODO.md:86` — the master-plan subgoal this branch executes
- `docs/plan/DEFERRED.md` — `## Trigger: the first real release` carries five obligations, all of which land here rather than after
- [ADR 0003](../../design/adr/0003-one-parameterized-test-suite-per-algorithm.md) — the sdist/wheel question, measured on the previous branch and settled here; now in its `## Resolved` section
- [ADR 0006](../../design/adr/0006-single-repository-as-a-uv-workspace.md) — the workspace footgun that makes a wrong version bound invisible locally
- `docs/agents/claude.md` — `/smart-commit` cannot produce this project's per-package tags under any configuration; the tag is manual

## Notes

- **2026-09-01 — ADR 0003's sdist/wheel question is settled: the mechanism is
  repo-local.** Tests keep shipping in the sdist; neither `addopts = "-ra"` nor
  `pytest_report_header` travels with them, and the record now says so along with what it
  costs. The two self-sufficiency remedies were rejected in the ADR rather than merely
  passed over. The obligation to revisit is re-filed in `DEFERRED.md` under the `align`
  migration, since `align` is the first member whose backend matrix will have a row in it.
  No packaging change was made, which is stated in the record so it does not read as an
  oversight.
- **2026-09-01 — semantic sweep done.** `README.md` was clean; the agent docs took nine
  edits (`codex.md` claimed "no algorithms, no tests" and called the settled `SymbolTable`
  provisional, which the same file tells a reviewer to report); the PRD was annotated in
  §9's existing style rather than tense-corrected, so it stays readable as a June 2026
  snapshot, with §8's two closed questions rewritten because that heading asserts the
  present. `docs/api/dataseq/` had not drifted at all — 44 of 44 examples matched — but
  nothing was checking it, so `tests/test_api_docs.py` now does. Suite 87 -> 91.
- **2026-09-01 — found while re-measuring: the sdist ships `.gitignore` and no README or
  LICENSE file**, though `license = "MIT"` is declared as metadata. The PyPI page would be
  blank on first publish, and `0.1.0` cannot be re-cut to fix it. Added as a subgoal to
  the branch plan's release goal.
- **2026-09-01 — the intra-family bounds on `pfsmgraph-dataseq` reviewed and spelled
  `>=0.1.0`** in all four dependents. Not a behaviour change — `>=0.1` and `>=0.1.0` are
  indistinguishable to a resolver — but a record that the bound has been checked against a
  version that will exist. The `pfsmgraph-align>=0.1` bounds were left alone on purpose, so
  the divergent spelling marks reviewed from unreviewed. Two measurements back it:
  `uv pip compile` on `pfsmgraph-dataseq>=0.1` outside the workspace is *unsatisfiable*
  against real PyPI, and `>=0.1` excludes the local `0.1.0.dev0` even with
  `prereleases=True`, since a `.devN` sorts below the final on ordering rather than on
  pre-release policy. **The lockfile is blind to all of it**: `uv.lock` came out
  byte-identical, because a workspace member's `requires-dist` entry carries no version
  specifier at all. Recorded in `DEFERRED.md` as the reason that entry recurs forever.

### 2026-09-02 — the release artifacts

`pfsmgraph-dataseq` moves to `0.1.0`; the other four members stay at `0.1.0.dev0`.

Two files the version bump does not imply had to land with it. The member shipped **no
README and no LICENSE file**, so its PyPI page would have been blank and its wheel would
have carried no license text despite declaring `license = "MIT"`. Both are now in the
member directory, and the license needed no `license-files` configuration at all —
hatchling's PEP 639 default glob finds a `LICENSE` in the project root unaided.

**The LICENSE cannot be a symlink to the repo-root one, and the failure mode is the
interesting part.** Hatchling writes the symlink verbatim into the sdist and the build
*succeeds*; `uv build` then dies unpacking that sdist to build the wheel from it —
*"symlink destination for ../../LICENSE is outside of the target directory"*. That is a
consumer-side failure, caught here only because `uv build` routes wheel-building through
the sdist. A backend building both from the source tree would have shipped a permanently
broken sdist under an immutable version number. Real copies, then, and the drift between
the five copies is a cost accepted rather than avoided.

The README is member-specific rather than a copy of the root one: the root README is about
the workspace, and every relative link in it is a 404 on PyPI. `[project.urls]` was added
alongside, since a page with a body but no link off it is the same blankness one layer down.

`tests/test_api_docs.py` now discovers `packages/*/README.md` as well as `docs/api/*/*.md`.
The reason is that a member README becomes a PyPI long description under an immutable
version — the one documentation surface where drift cannot be corrected in place, and until
now the only one with no guard on it. It earned the change immediately, failing the draft
twice where examples had been transcribed from `print()` output rather than the `>>>` repr.
Suite 91 → 92.

Verification is a clean-venv install of the wheel, not a listing of its contents: outside
the workspace it pulls only numpy, reports `0.1.0`, reproduces every README example byte for
byte, imports neither torch nor pandas, and leaves `pfsmgraph.__file__` as `None` — the PEP
420 namespace invariant holding in the shipped artifact rather than only in the source tree.
Both artifacts pass `twine check`.

`.gitignore` still ships in the sdist. That is the repo-root file, which hatchling includes
so the exclusion rules travel with the sdist; it is left alone.

### 2026-09-02 — the PEP 561 marker, and how `uv publish` is actually addressed

`src/pfsmgraph/dataseq/py.typed` was added before publishing, with the `Typing :: Typed`
classifier alongside. The source has been fully annotated since the merge — a `Vocabulary`
`Protocol`, `-> list[str]`, `frozenset[str]` throughout — and without the marker a type
checker discards all of it. Measured against the wheel in a clean venv: `vocab.size`
revealed as `Any`, `vocab.decode(...)` as `Any`, and a deliberate `bad: str = vocab.size`
**accepted silently**. With the marker: `int`, `list[str]`, and the bad assignment reported.

The suggestion that prompted this named `packages/pfsmgraph-dataseq/py.typed`, the
distribution root. Built that way the wheel contains no `py.typed` at all — no error, no
warning — because the file belongs to no package and `packages = ["src/pfsmgraph"]` never
sees it. Inside the importable package it ships with no `pyproject.toml` change, hatchling
including every file under that tree rather than only `.py`.

PEP 420 makes the placement load-bearing rather than conventional. No single distribution
owns the `pfsmgraph/` level, which is why no `__init__.py` may sit there; a marker there
would be one member asserting typedness for four it does not ship. Each member marks its own
regular subpackage, and the obligation for the other four is filed in `DEFERRED.md` under
the release trigger — in the release commit, since `py.typed` is wheel content and adding it
later leaves a published version standing as the one whose types do not work.

**On publishing.** `uv publish` has no `--package` flag; it takes file globs defaulting to
`dist/*`, and `dist/` is one directory shared by all five members, so the files are named
explicitly and a project-scoped PyPI token is the backstop that turns a stray glob into a
403 rather than a burnt version. Nothing in this repository or the user configuration sets
the endpoint — verified by `--dry-run` — so uv uses its default
`https://upload.pypi.org/legacy/`. Credentials are likewise unconfigured and must be passed
at the call: uv attempts trusted publishing (OIDC) first, which resolves only inside CI, so
that failure on a development machine is expected rather than a fault.
