# feat/hmm-search-log

**Created**: 2026-09-17
**Base**: main at 6768d9b
**Status**: active

## Purpose

Give a topology search a progress report on standard output, and record why the one
alternative to that — the original's GUI console — migrated nowhere. The two are halves of
one decision rather than two adjacent chores: text on stdout is the reporting surface
*because* `hmm-trainer-view.lsh` was dropped, a link `docs/plan/DEFERRED.md:577-593` already
asserts from the deferral's side and which nothing yet states from the migration's side.
This branch executes the two master-plan subgoals that own those halves.

## Scope

- Port `update-training-log` / `training-log-line` (`hmm-trainer.lsh:457-477`) as the
  search's own log: one row per accepted move, carrying size, the split (`n ^`) or merge
  (`i v j`) marker, `data-dl`, `model-dl`, `total-dl` and `d`. Drop `test-data-dl`, which
  the original never assigned.
- Decide what the original leaves open: whether rejected moves are reported at all, and
  whether the nested `baum_welch(..., log=)` convergence block is silenced per candidate.
- Test the **format** against the three tracked `_training_log` files, not the values.
- Record `hmm-trainer-view.lsh`'s non-migration where a reader will meet it, saying both
  why dropping it is lossless and what is lost with it.

## Context

- Master plan `docs/plan/TODO.md:219` and `:221` — the two subgoals, under
  `## Subgoals — revision 04-hmm-v0.3.0`.
- `.scratch/hmm-lush/Code/HMMlib/hmm-trainer.lsh:457-477` — `update-training-log` and
  `training-log-line`, the format being ported; `:677-690` — `keep-model`, which is what
  decides a line is written, and only for an *accepted* move whose topology changed.
- The three format oracles, tracked as fixtures:
  `.scratch/hmm-lush/Training/set02a/set02a_200/m001_0001_001.hmm/_training_log`,
  `m001_0005_005.hmm/_training_log` (five lines, the only multi-move one),
  `m008_0001_008.hmm/_training_log`.
- `.scratch/hmm-lush/HMMLIB-ACCOUNT.md` §11 — `hmm-trainer-view.lsh` as "presentation only",
  and as one of the two written statements of the workflow `_search.py` ports. Its "there is
  no headless entry point" claim carries a 2026-09-17 correction.
- `docs/plan/DEFERRED.md`, `## Trigger: a second module needing training progress reporting`
  — the dashboard deferral this branch must not reopen, and the dangling pointer it makes.
- `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_baum_welch.py:350-382` — revision 03's log
  helpers (`_log_start`, `_log_row`, `_log_budget_stop`, `_write`), the shape a search log
  either reuses or deliberately does not.
- `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_search.py:109` — `_search`'s signature, where a
  `log=` would attach, beside the `backend=` / `score_backend=` pair.

## Notes

- The branch is **not** named `feat/hmm-training-log`: revision 03 used that name for
  `baum_welch(..., log=)` and its plan is the durable record at
  `docs/plan/03-hmm-v0.2.0/feat-hmm-training-log/`. Plans resolve by directory name
  anywhere under `docs/plan/`, so a reused name makes every later lookup ambiguous.
- The first column of the original log is the model **size**, not a round counter. It
  tracks the round in `m001_0005_005` only because every accepted move there is a split.
