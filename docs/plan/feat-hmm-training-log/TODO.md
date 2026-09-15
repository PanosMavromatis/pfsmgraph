# feat/hmm-training-log

**Status**: active
**Created**: 2026-09-15
**Subgoal**: Migrate the training log as standard-output reporting only (revision 03-hmm-v0.2.0)

## Goals

- [x] Decide the log's surface: which of the seven original fields revision 03 fills and what the rest become, how a caller turns it on and where lines go, how often one is emitted, and whether lines are retained on the result
  > **Note:** the original log is **topology-search history, not EM progress**. `update-training-log` has one caller, `keep-model` (`hmm-trainer.lsh:682`), which runs it only when `topology-dirty` is set, so a line is appended per *accepted* split or merge; `run-converge` writes nothing. The three tracked `_training_log` files agree: one line each for `m001_0001_001` and `m008_0001_008`, five for `m001_0005_005` (sizes 1 to 5, every step a split). The master plan's "hundreds of lines in one cell" assumed a per-cycle report the original never had.
  > **Note:** `test-data-dl` is dead in the original: initialised to `0.0` (`:74`), never assigned again, with `test-data-seq` commented out (`:30`). All seven logged lines print `0`.
  > **Q:** Given that, what does revision 03 deliver: new per-convergence-check progress lines with the faithful per-topology line moved to revision 04, a faithful seven-field port printed once per call, or the whole log descoped to revision 04?
  > **A:** Split it. Revision 03 emits per-convergence-check progress lines to standard output; `training-log-line`'s per-topology record (size, split/merge marker, data/model/total dl, `d`) moves to revision 04, where every field exists, and `test-data-dl` is dropped as dead.
  > **Q:** How does a caller turn the log on, and where do lines go: a `log=` text stream, a `verbose=` bool printing to `sys.stdout`, or a `progress=` callback receiving each line?
  > **A:** A keyword-only `log: TextIO | None = None` on `baum_welch`: `None` is silent, `log=sys.stdout` prints to the cell, each line written and flushed. Tests pass an `io.StringIO`; a caller wanting a file opens one and passes it, which bears directly on goal 3.
  > **Note:** a sample run (random 5-state start with `noise_width=0.5`, seed 0, the tracked corpus as one record) took 5.2 s for 70 cycles, one convergence check about every 0.7 s. From cycle 40 the bits move only from 1636.95 to 1636.82, so the per-check change column is what shows a run approaching its stop.
  > **Q:** How often is a line emitted, and what does it carry: a header, cycle 0, each convergence check and a closing line; every cycle; or checks only?
  > **A:** A header, a cycle-0 line with the starting bits, one line per convergence check (every `batch_cycles`), and a closing line naming how the run stopped (converged, or `max_cycles` reached). Columns: cycle, bits, change since the previous check, and quiet checks out of `patience`. No elapsed time, since ADR 0013's executed-output rule cannot pin a wall-clock value.
  > **Q:** Are the lines also retained on `BaumWelchResult`?
  > **A:** No. The result is unchanged; every printed number is derivable from `description_lengths`, `cycles` and `converged`, and a caller wanting the text passes an `io.StringIO`. The per-topology history Lush kept on the model is revision 04's to design, with persistence.
  > **Done:** surface settled as `baum_welch(..., log: TextIO | None = None)`, writing and flushing a header, a cycle-0 line, one line per convergence check (cycle, bits, change, quiet/patience) and a closing stop line; nothing retained on the result. The original seven-field per-topology line moves to revision 04, `test-data-dl` is dropped as dead.
- [x] Implement the line formatter and its emission from `baum_welch`
  > **Q:** Which number format: fixed-point bits with scientific change, fixed point for both, or scientific for both?
  > **A:** Bits in fixed point to 6 decimals, the column sized from cycle 0's value, which is the widest it can be since EM never raises the description length; change in `%.6e`, a constant 13 characters, so a change far below `change_bits` still shows its magnitude rather than `-0.000000`. `inf` right-aligns in either field with no special case.
  > **Note:** on the reference, a check's forward pass equals that cycle's E-step total bit for bit, 0 mismatches in 2935 cycles over 20 random multi-record runs, so a check row prints exactly `description_lengths[k]` and the tests compare whole lines as strings rather than parsing floats. On `torch` only the cycle-0 row, which is torch's own E-step, can differ from the reference.
  > **Note:** a `max_cycles` stop between checks writes one extra row with no unchanged count (none was taken); a stop on a check adds only the stop line; `max_cycles=0` writes the header, cycle 0 and the stop line. The cycle-0 row is written after the impossibility check, so an impossible corpus raises having printed nothing.
  - [x] The format stays aligned and readable over hundreds of lines, including `inf` and very large or small values
    > **Note:** pinned by calling `_log_start` and `_log_row` directly with a 12,345,678-bit start, a change of `-1.1e-7`, `±inf` and a five-digit cycle, asserting every field ends at its header's column; the saddle run's rows are asserted line for line, unchanged counts included.
  - [x] Logging changes no result: training is bit-identical with the log on and off, on every backend
    > **Note:** `test_the_progress_log_changes_no_result_on_any_backend` compares params, `description_lengths`, cycles, `converged` and degenerate states with `==` on `python` and `torch`, converged and `max_cycles`-stopped. Torch on a CUDA device is not in that fixture's table; the log reads only host floats after the E-step, so `device=` has no path into it.
- [ ] Decide what `save-training-log` becomes while model persistence is deferred
- [ ] Document the log under `docs/api/hmm/`, with executed output, and update `core.md`
