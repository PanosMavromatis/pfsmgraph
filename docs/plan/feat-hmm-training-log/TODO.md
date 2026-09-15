# feat/hmm-training-log

**Status**: active
**Created**: 2026-09-15
**Subgoal**: Migrate the training log as standard-output reporting only (revision 03-hmm-v0.2.0)

## Goals

- [ ] Decide the log's surface: which of the seven original fields revision 03 fills and what the rest become, how a caller turns it on and where lines go, how often one is emitted, and whether lines are retained on the result
- [ ] Implement the line formatter and its emission from `baum_welch`
  - [ ] The format stays aligned and readable over hundreds of lines, including `inf` and very large or small values
  - [ ] Logging changes no result: training is bit-identical with the log on and off, on every backend
- [ ] Decide what `save-training-log` becomes while model persistence is deferred
- [ ] Document the log under `docs/api/hmm/`, with executed output, and update `core.md`
