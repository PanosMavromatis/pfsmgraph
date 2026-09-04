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
