# feat/hmm-search-log

**Status**: active
**Created**: 2026-09-17
**Subgoal**: Two — report search progress on standard output, and record `hmm-trainer-view.lsh`'s non-migration — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Context

The two master-plan subgoals share a branch because they are halves of one decision rather
than two adjacent chores. `docs/plan/DEFERRED.md`'s `## Trigger: a second module needing
training progress reporting` decided on 2026-09-03 that training reports to **standard
output and nothing else**, and gives `hmm-trainer-view.lsh` as the consequence: "This is why
`hmm-trainer-view.lsh` (237 lines) is migrated **nowhere** — see revision `04-hmm-v0.3.0`."
That pointer has had no other end for a revision and a half. Goal 1 builds the stdout
report the deferral promised; goal 2 writes the record the pointer points at. Neither is
large, and splitting them would put the two sides of one argument in two PRs.

**The format is a port; the numbers are not.** `training-log-line`
(`hmm-trainer.lsh:462-477`) fixes the columns and their widths, and the three tracked
`_training_log` files are the oracle for it:

```
m001_0001_001:    1       -       3140.16    58.2207    3198.38    29     0
m001_0005_005:    1       -       3140.16    58.2207    3198.38    29     0
                  2     0 ^       2342.21    91.7754    2433.99    10     0
                  3     0 ^       2044.98    86.1254    2131.1     3      0
                  4     2 ^       1665.24    108.788    1774.03    3      0
                  5     0 ^       1606.87    142.024    1748.9     3      0
m008_0001_008:    8       -       1640.45    439.154    2079.6     4      0
```

Two things a reader should not infer from them. The first column is `:params:size`, the
**model size**, not a round counter — it tracks the round in `m001_0005_005` only because
every accepted move there is a split, and a merge would decrement it while a rejected round
emits nothing at all. And line 1's `d = 29` is Brent's local minimum: our bounded exact scan
returns **13** on that same one-state model, 10.46 bits cheaper (ADR 0023 §5). A test
pinning printed values against these files would pin a defect this project deliberately
fixed, so the oracle is the layout.

`keep-model` (`:677-690`) is what decides a line exists: it writes one only `when
topology-dirty`, so the log records accepted moves that changed the topology and nothing
else. `_search` keeps each round's winner unconditionally, which is the same rule, but it
also *visits* candidates the original never scored, since the original's user ran one trial
at a time by hand.

## Goals

- [ ] Port the training log to standard output as the search's own progress report. `update-training-log` and `training-log-line` (`hmm-trainer.lsh:457-477`) append one line per accepted move, holding the size, the split (`n ^`) or merge (`i v j`) marker, `data-dl`, `model-dl`, `total-dl` and `d`. Drop `test-data-dl`, which the original never assigned (all seven logged lines read `0`). Every one of those quantities is produced by `_mdl.py` and the search loop alone, so this is a port of a format rather than of a computation.
  - [ ] Decide whether **rejected** moves are reported. The original records only accepted ones, and `_search` rejects most of what it scores; a search that reported every candidate would say far more about the run than the original's log ever did, which is either the improvement this goal is for or noise that buries the accepted line.
  - [ ] Decide what happens to the **nested** `baum_welch(..., log=)` block. It already writes a header, a row per convergence check and a stop line, and a search re-converges every candidate — so a search that passes its stream down emits one such block per rejected move. Decide whether the search silences it, forwards it, or makes it selectable, and say why in the docstring rather than only here.
  - [ ] Implement it on `_search`, beside `backend=` / `score_backend=`, reusing or deliberately not reusing `_baum_welch.py:350-382`'s `_log_start` / `_log_row` / `_write` helpers. Flushing per line is already that module's rule and a search is the longer-running caller, so it matters more here.
  - [ ] Test the **format** against the three `_training_log` oracles — column order, the three marker forms (`n ^`, `i v j`, `-`), and that a start row precedes every move — never the values. Add a merge-accepting case, which no oracle has.
  - [ ] Document whatever this exports in `docs/api/hmm/`, with executed blocks per ADR 0013. Note that a log's output is host-independent here only if the numbers in the example are, which is the constraint `docs/benchmarks/` exists to take off these pages.

- [ ] Record that `hmm-trainer-view.lsh` (237 lines) migrated **nowhere**, and why, so the omission reads as a decision rather than an oversight. It is the only file in `Code/HMMlib/` with no destination.
  - [ ] Decide where the record lives. Four places already name the file and none of them *is* the record: `HMMLIB-ACCOUNT.md` §11 (which says "it migrates nowhere" in passing, inside a section whose headline claim carries a 2026-09-17 correction), `DEFERRED.md:580` (which cites this revision for the reason), ADR 0017 and ADR 0023. The test is where a reader asking "what happened to the GUI?" would actually look.
  - [ ] Say **both** things the master plan asks for, since either alone misleads. It is "presentation only — every button calls a trainer method and then `update-view`" (§11), which is what makes dropping it lossless; and it is, beside `Training/hmm-train-new-nw`, one of the two written statements of the workflow `_search.py` ports, which is what is lost with it. Its fifteen buttons are the requirements list ADR 0023 read.
  - [ ] Check the record against the correction §11 already carries. "There is no headless entry point" is false and marked so; a record that leans on §11's framing without noticing would inherit a retracted claim.
