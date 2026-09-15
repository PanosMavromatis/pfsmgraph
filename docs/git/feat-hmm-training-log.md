# feat/hmm-training-log

**Created**: 2026-09-15
**Base**: main at 0227f65
**Status**: active

## Purpose

Migrate the Lush trainer's training log as standard-output reporting only, so a long
`baum_welch` run stays observable from a Jupyter output cell while it continues. Ports
`update-training-log` and `training-log-line` (`hmm-trainer.lsh:457-477`) and
`save-training-log` (`hmm.lsh:166`). No GUI, no widget, no plotting dependency.

## Scope

- Decide which of `training-log-line`'s seven fields revision 03 can fill on a fixed
  topology, and what stands in for the rest
- Decide where lines go, when one is emitted, and whether they are retained
- Implement the line format and its emission from `baum_welch`, with tests
- Decide what `save-training-log` becomes, given that persistence is deferred
- Document it under `docs/api/hmm/` and update `core.md`

## Context

- Master plan: `docs/plan/TODO.md`, revision 03-hmm-v0.2.0, the training-log subgoal
- `docs/plan/DEFERRED.md`, `## Trigger: a second module needing training progress
  reporting`: standard output only, decided 2026-09-03; `hmm-trainer-view.lsh` migrates
  nowhere
- `training-log-line` prints `size`, a split/merge marker, `data-dl`, `model-dl`,
  `total-dl`, `d` and `test-data-dl`; the marker, `model-dl`, `total-dl` and choosing `d`
  belong to revision 04's topology search and `_mdl.py`
- `baum_welch` already records `description_lengths` per cycle on `BaumWelchResult`, and
  `_data_description_length(params, records, d)` gives the quantised `data-dl`

## Notes
