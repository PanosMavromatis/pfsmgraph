# exp/meson-python-namespace

**Created**: 2026-09-04
**Base**: main at ef32701
**Status**: active

## Purpose

Resolve the meson-python editable-install namespace shadowing recorded in ADR 0012 and
move `align` and `hmm` back onto meson-python, choosing between that ADR's three
candidate resolutions by evaluating them rather than by argument. The branch is `exp/`
because the outcome is genuinely unknown at branch time: one of the three candidates is
"an upstream fix", which may not be available at all.

## Scope

- Reproduce the shadowing on today's workspace — ADR 0012's evidence was gathered when
  `hmm` was empty scaffolding.
- Repair the drift in both `meson.build` files before evaluating anything, and retarget
  `hmm`'s dormant extension block from `_baum_welch` to whatever the first `.pyx`
  actually is.
- Settle the sequencing question below, then evaluate and choose a candidate.
- Apply the revert recipe, restore `meson-python`/`cython`/`ninja` to the root `dev`
  group, and verify all five members import after `uv sync`.
- Supersede ADR 0012 with a new record, and clear the footnotes it planted.

## Context

- [ADR 0012](../../design/adr/0012-align-and-hmm-temporarily-on-hatchling.md) — the
  deviation, its evidence, the revert recipe, and the three candidates.
- [ADR 0008](../../design/adr/0008-per-package-build-backends.md) — the decision 0012
  qualifies. [ADR 0005](../../design/adr/0005-namespace-prefix-and-pep-420-layout.md) is
  the other half of the collision.
- Master plan `docs/plan/TODO.md`, the goal above the Cython phase.

**Sequencing tension, carried in deliberately.** ADR 0012 rejected solving this now as
premature — "choosing between them without a compiled kernel to evaluate against would be
guessing… The information needed to choose arrives with the first `.pyx`" — yet the
master plan puts this goal *before* the Cython phase. Both orders are defensible and the
branch has to pick one: build a throwaway extension here purely as an evaluation target,
or interleave with the Viterbi `.pyx` and let this branch land the real one.

**Known drift, found before the branch opened.** `packages/pfsmgraph-hmm/meson.build`
lists only `__init__.py` in `install_sources`, and meson does not glob; `hmm` now has
three further modules. A revert without repairing this builds a wheel that imports
nothing. `align` has the same shape but no drift yet, having no code. This is the cost
ADR 0012 predicted under "the `meson.build` files are unexercised".

## Notes

**2026-09-04 — the shadowing is measured, and ADR 0012 needs three corrections.** The
loader claims `{'pfsmgraph'}` structurally (meson-python derives the claim from top-level
installed names, and under PEP 420 that *is* `pfsmgraph`), so candidate 3 is an upstream
design change rather than a bug report. `pfsmgraph.__path__` collapses to a single
synthetic entry inside the loader file — replaced, not extended. And PRD §6.1's "`ninja`
must be on `PATH`" is necessary but not sufficient: the loader bakes an absolute ninja
path at build time, which under uv points into a deleted build-isolation directory, so
`no-build-isolation-package` is required too.

**2026-09-04 — the drift is repaired and guarded.** `hmm`'s `install_sources` named 1 of
4 modules; meson's install plan now carries all four. `tests/test_meson_sources.py`
guards it, mutation-tested against both a dropped module and an unlisted `py.typed` —
the latter being the release-commit case a "every module is listed" guard would miss.
The extension block now targets `_viterbi_cython.pyx`; `_viterbi` alone would collide
with the phase-1 reference the ADR 0003 registry names.

**2026-09-04 — candidate 2 is refuted.** Two meson-python finders chain rather than
conflict: `align` imports fine behind `hmm`'s finder. The boundary is meson-python versus
plain `.pth`, so a combined compiled distribution would still shadow `dataseq`, `hseg`
and `dl`. A fourth candidate — all five members on meson-python — follows from the same
measurement. Likely explanation for the original claim: without `no-build-isolation`,
every import dies `FileNotFoundError`, which looks like everything conflicting with
everything.

**2026-09-04 — the sequencing question is settled: decide and land here, no `.pyx`.**
Every finding above was obtained with the extension blocks dormant, which falsifies ADR
0012's premise for deferring rather than merely leaning against it. The error is
nameable: 0012's Context identifies the mechanism correctly as the *editable-install*
import hook, then its Alternatives reasons two sections later as though the deferred cost
were *compilation* — "the information needed to choose arrives with the first `.pyx`".
Those are separable, since a meson-python member with no compiled code still injects the
finder. A throwaway `.pyx` was rejected: its only unique yield is dev-loop friction data,
which the real `_viterbi_cython.pyx` produces one master-plan goal later anyway.

One coupling this surfaced, carried into the evaluation goal: *when* to land is
downstream of *what* is chosen. Candidate 1 makes every `.py` edit in `hmm` need a
reinstall — strictly worse than today's hatchling + editable, for a benefit that does not
exist until the `.pyx` — while candidate 4 costs nothing extra today. So a candidate-1
win may still argue for deferring the apply step even though the decision does not need
deferring. Candidate 1's own crux is now pinned too: whether `[tool.uv]` mirrors
`--no-editable-package` the way it mirrors `no-build-isolation-package`. If it does not,
that candidate is not "a manual reinstall step in the dev loop" as 0012 describes it but
a non-default `uv sync` every contributor and CI job must remember.

**2026-09-04 — the candidates are measured and the choice is made: all five members on
meson-python (candidate 4).** Both live candidates work. Candidate 1 — `align` and `hmm`
non-editable — leaves no finder in `sys.meta_path` at all, composes the namespace from a
real `site-packages/pfsmgraph/` portion plus three `.pth` entries, passes 271, and
*retires* the baked-`ninja` footgun rather than inheriting it, since a non-editable
install has no loader and never rebuilds. Candidate 4 puts five `MesonpyMetaFinder`s in
the chain and passes 280.

The crux pinned in the previous entry resolved sideways and is worth recording as stated,
because the conclusion survived the premise: `[tool.uv]` does **not** mirror
`--no-editable-package` — uv's own unknown-field error enumerates the accepted keys and it
is absent — and there is no env var for it either, though the all-or-nothing
`--no-editable` has `UV_NO_EDITABLE`. But `[tool.uv.sources]` accepts an `editable` key,
which a *virtual* workspace root honours, so `{ workspace = true, editable = false }` is
persistent under a plain `uv sync` and the "non-default invocation" objection dissolved.
What replaced it is worse. A member-level `[tool.uv.sources]` declaration beats the
root's, and `hmm`, `hseg` and `dl` each declare `pfsmgraph-align = { workspace = true }`;
setting `editable = false` only at the root left `align` editable, and its surviving
finder broke **all five** imports. The setting has to be repeated at every declaration
site, and the next member that copies a one-line source declaration silently breaks the
workspace. Separately, a non-editable member goes stale with no error on a source edit —
plain `uv sync` reports `Checked 26 packages` and serves the old copy — because uv's
default cache key for a local path is its `pyproject.toml`, not its sources;
`[tool.uv] cache-keys = [{ file = "src/**/*.py" }]` repairs it, after which `uv run` alone
rebuilds in ~2.7 s.

So the decision was made on failure loudness, which is the criterion this repository has
already applied to the misplaced `py.typed`, the inert `.gitignore` rule over a tracked
path, and the workspace version footgun. Candidate 4's characteristic mistake — a module
missing from `install_sources` — is a `ModuleNotFoundError` at import and is already
guarded: `tests/test_meson_sources.py` is parameterised over `packages/*/meson.build` and
went 7 → 16 tests by itself when the three new files appeared. Candidate 1's two mistakes
are both silent and one is workspace-wide. The dev loop reinforces rather than decides:
0.15 s with no sync at all against 2.7 s, and the gap widens at the first `.pyx`, where
candidate 1 degrades to a full recompile per edit. That inverts ADR 0012's implicit cost
model, which read non-editable installation as the cheap fallback.

Candidate 3 is closed too, and not merely because upstream cannot be waited on:
meson-python documents the stub-and-finder mechanism without mentioning PEP 420 anywhere
and no issue describes the shadowing, while the nearest published report
(`microsoft/pylance-release#3002`) is the same shape in a different toolchain — evidence
that the interaction is generic to finder-based editable installs rather than a
meson-python defect. There is nothing here to file as a bug; a finder-composition feature
request is worth making on its own merits, not as a candidate.

Two things the superseding ADR must argue rather than assume: this overrides **ADR 0008**'s
per-package build backends, three members acquiring meson-python and a hand-maintained
source list for no compiled code; and `pfsmgraph-dataseq` **0.1.0 is already published from
hatchling**, so its next release ships a meson-built wheel and the four-file release
invariant needs re-verifying against an actual built wheel in a clean venv.

One trap for whoever lands it: `no-build-isolation-package` does **not** invalidate uv's
build cache. Switching a member onto it reused an editable wheel built *with* isolation,
whose loader had a baked path into a `builds-v0` temp dir uv had already deleted — so
`align` alone failed while the other four worked, which reads exactly like a candidate-4
defect and is a stale cache. `uv sync --reinstall` clears it, and a landing check must
start from `rm -rf .venv` to mean anything.

**2026-09-04 — the choice is landed and verified from a clean venv.** All five members now
declare `build-backend = "mesonpy"`; `dataseq`, `hseg` and `dl` gained a `meson.build`
apiece, and the two dead `[tool.hatch.build.targets.wheel]` blocks that were still sitting
in `dataseq` and `dl` are gone. The suite is green at 280.

*The scope was larger than the plan said, in two ways.* The subgoal read "revert recipe
applied to **both** members", which was drafted while candidates 1–2 were still live —
under the chosen candidate it is five, and the three pure members had no recipe to follow
because nobody had expected them to move. And the root `dev` group needs **`numpy`**
alongside `meson-python`/`cython`/`ninja`: it is a *build* requirement of `align`/`hmm`,
and with `no-build-isolation-package` in force `build-system.requires` can no longer
supply it. That is a general consequence of turning isolation off, not a quirk — every
build dependency of every listed member has to be present in the environment.

*Every one of the five build-system comments was wrong, and only three contained the word
"hatchling".* `dataseq`'s said to switch to meson-python *only if* a compiled inner loop
were found; `hseg`'s said "hatchling while hseg is pure orchestration"; `dl`'s said
"Pure-Python (PRD §6)", which is still true and had simply stopped being an explanation.
Sweeping for the stale *word* would have found three of five. Sweeping for the stale
*claim* — what does this file assert about why it is on this backend? — found all five,
plus `align/meson.build` saying ninja returns "when that ADR is reverted" (it returns
because 0012 was superseded, which is a different thing) and the root dev group still
explaining itself in terms of `align`/`hmm` moving *back*. The namespace argument is now
written once, in `dataseq`'s `pyproject.toml`, with the other four pointing at it.

*A mechanical trap, recorded because the next edit of this kind will hit it.* The old
revert recipes contained `build-backend = "mesonpy"` as a **commented** line. A textual
replacement bounded by that string therefore terminated inside the comment and left a
duplicated `requires` / `build-backend` pair below the new block. TOML is last-key-wins, so
the file still parsed and the backend was still correct — it would not have failed loudly,
merely left contradictory text in the very files this session existed to stop
contradicting themselves. `grep -c '^build-backend'` per file caught it. A file that
documents its own replacement is a hazard for any edit anchored on content.

*What the clean verification actually established.* `rm -rf .venv` and a plain `uv sync` —
`--reinstall` would have been the wrong shortcut, since it reuses an environment that
already has `ninja` on disk. uv reported "Prepared 5 packages without build isolation in
2.06s", which answers the one open question the config does not state: turning isolation
off means the five members need `meson-python`/`cython`/`ninja` present *before* they
build, yet those arrive via the `dev` group in the same sync. uv sequences it correctly,
but that ordering is emergent from the resolver rather than declared anywhere, so it had to
be measured. All seven import paths resolve, five finders sit on `sys.meta_path`, and
`pfsmgraph.__path__` remains a single synthetic loader entry — the namespace is still
replaced, and it no longer matters. The dev-loop claim that decided the choice was checked
rather than trusted: a name appended to `hseg/__init__.py` was visible with **no sync at
all**.

*Downstream consequence for the release runbook.* `docs/ops/release.md`'s reproducibility
findings — byte-identical wheel, non-reproducible sdist carrying the repo-root
`.gitignore` — were all measured against a **hatchling** build, and are properties of that
builder rather than of this project. They are neither known false nor known to still hold.
`pfsmgraph-dataseq` 0.1.0 shipped from hatchling, so its next release is the first
meson-built wheel here, and the section has to be re-measured against it. The four-file
invariant needs re-verifying at the same time, `py.typed` most of all: meson does not glob,
so `install_sources` must name it explicitly.
