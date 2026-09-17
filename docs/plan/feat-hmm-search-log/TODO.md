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

- [x] Port the training log to standard output as the search's own progress report. `update-training-log` and `training-log-line` (`hmm-trainer.lsh:457-477`) append one line per accepted move, holding the size, the split (`n ^`) or merge (`i v j`) marker, `data-dl`, `model-dl`, `total-dl` and `d`. Drop `test-data-dl`, which the original never assigned (all seven logged lines read `0`). Every one of those quantities is produced by `_mdl.py` and the search loop alone, so this is a port of a format rather than of a computation.
  > **Done:** `_search` takes a `log=` text stream: a header, a row for the starting model,
  > one row per round, and a closing sentence naming the stop rule. Format ported from
  > `training-log-line`, verified byte-for-byte against all seven oracle lines *before*
  > departing from those bytes for a table. 11 tests, mutation-checked against nine
  > defects, two of which exposed genuine gaps. `docs/api/hmm/` is deliberately unwritten
  > -- nothing is exported -- and that surfaced a master-plan gap, now its own subgoal:
  > no item of revision 04 decides whether the search is exported at all.
  > **Q:** Does the search's log report rejected moves, or only the accepted one per round?
  > **A:** Accepted, preceded by a per-round summary naming the candidate count and the
  > runner-up's margin. A near-tie is precisely what ADR 0023's Open item says the criterion
  > cannot decide -- rankings closer than the 58-71 bits one integer step of `d` moves the
  > model half are settled by where EM stopped -- so the margin is the one number the
  > original's log could not have carried, its user having run one trial at a time by hand.
  > **Q:** What happens to the nested `baum_welch(..., log=)` block, which a search would emit
  > once per re-converged candidate?
  > **A:** Silence it: `_search` never passes `log=` down. Its stream is about topology moves
  > and EM's is about convergence inside one candidate, and `baum_welch`'s `log=` is already
  > public for anyone wanting the latter. Say so in the docstring, not only here.
  > **Note:** The three number columns cost no second forward pass, but not the obvious way.
  > `TrialResult.data_bits` is the **unrounded** corpus length, which `d` never touched, so
  > printing it beside `total - data_bits` would put the whole quantization gap into the model
  > column -- 1.5 to 29 bits on the tracked models, against a model half that is 58 bits on
  > line 1. Take the model half from `_model_description_length(params, d)`, which reads no
  > corpus at all, and derive the data half as `total - model`: exact, and consistent with the
  > printed total by construction.
  > **Q:** `_search` is private and `hmm.__all__` is unchanged at ten names, so there is no
  > public surface to document. What should the `docs/api/hmm/` subgoal produce?
  > **A:** Defer the page and close the pointer. Mark it `[-]` with the reason, and amend
  > `baum_welch.md`'s one dangling sentence so it stops pointing at nothing. The export
  > decision should be made on its own merits, not as a side effect of a docs subgoal.
  > **Q:** Should the master plan record a decision about whether revision 04 exports the
  > topology search at all?
  > **A:** Yes -- add a revision-04 subgoal. No existing subgoal makes it public, so 0.3.0
  > as planned ships "topology search by state merge and split" with a ten-name `__all__`.
  > Either answer is fine; what is missing is that it was never asked.
  > **Q:** Does the interleaved `round n: ...` summary line read well beside the move rows?
  > **A:** No -- two line shapes for one event makes the output awkward to scan. Each round
  > emits exactly **one** row instead, with the summary folded in as columns: `Round  Size
  > Move  Data DL  Model DL  Total DL  d  Candidates  Runner-Up  Comments`. The `stopped:`
  > line stays a bare sentence at the very end, where the visual disruption is the point --
  > it marks that the process has terminated.
  > **Note:** This is where the port leaves the original's *bytes* behind, and the departure
  > was measured before it was taken. All seven oracle lines round-trip byte-for-byte through
  > the Lush field widths (`%3d`, the 13-character marker, three `%-9g`, one `%-5g`); a table
  > carrying four more columns cannot also be that. What the oracles still constrain -- and
  > so what the format test asserts -- is the **column order** and the three marker forms
  > `n ^`, `i v j` and `-`, which is what the subgoal asked for in the first place.
  - [x] Decide whether **rejected** moves are reported. The original records only accepted ones, and `_search` rejects most of what it scores; a search that reported every candidate would say far more about the run than the original's log ever did, which is either the improvement this goal is for or noise that buries the accepted line.
  - [x] Decide what happens to the **nested** `baum_welch(..., log=)` block. It already writes a header, a row per convergence check and a stop line, and a search re-converges every candidate — so a search that passes its stream down emits one such block per rejected move. Decide whether the search silences it, forwards it, or makes it selectable, and say why in the docstring rather than only here.
  - [x] Implement it on `_search`, beside `backend=` / `score_backend=`, reusing or deliberately not reusing `_baum_welch.py:350-382`'s `_log_start` / `_log_row` / `_write` helpers. Flushing per line is already that module's rule and a search is the longer-running caller, so it matters more here.
    > **Note:** Only `_write` is reused from `_baum_welch`, and the split is principled.
    > `_write` encodes a *policy* -- flush every line -- which both logs share and which
    > matters more here, a round costing seconds where an EM cycle costs milliseconds.
    > `_log_start` and `_log_row` encode a *layout* with no column in common with this one,
    > so reusing them would mean parameterising until nothing of either was left.
    > **Note:** The `Round` column is `index + 1`, deliberately out of step with
    > `SearchResult.rounds[r].index == r`. The column counts the walk's rows, of which 0 is
    > the starting model; the field stays 0-based because it is also the round's seed key,
    > `spawn_key + (1, r)` (ADR 0023 §2), and renumbering it would silently re-seed every
    > search. `_log_move_row`'s docstring carries this.
    > **Note:** The `Runner-Up` column earned itself on its first real run. Over four
    > 200-symbol records of `set02a_200`, every round printed a margin of `+0`: the two
    > split trials of the one-state start converge to **bit-identical parameters**, from
    > candidates that genuinely differ before EM (max transition delta 1.3e-3, the 1% seed
    > working as ADR 0022 specifies) and after different cycle counts, 50 and 60. So the
    > criterion did not fail to separate them -- there was nothing to separate, and the
    > round's choice fell to `sorted`'s stability. Measured at `split_min_cycles=30` on a
    > corpus slice, so it is no claim about the full corpus at ADR 0023 §6's 200-cycle
    > floor; it is a claim that the column reports something the original's log could not.
  - [x] Test the **format** against the three `_training_log` oracles — column order, the three marker forms (`n ^`, `i v j`, `-`), and that a start row precedes every move — never the values. Add a merge-accepting case, which no oracle has.
    > **Note:** 11 tests, `test_search.py` 37 -> 48, suite 3088 -> 3099. The oracle is used
    > twice over: `_oracle_rows` parses all seven lines, and
    > `test_the_oracle_identifies_its_own_column_order` asserts `data-dl + model-dl ==
    > total-dl` on each, which is what makes the column *order* recoverable from the files
    > rather than only from `training-log-line`'s source. No printed value is pinned.
    > **Note:** The oracle's tolerance is measured, not guessed: `%g` rounds each field to
    > six significant digits independently, so the halves miss their own printed total by
    > up to **3.4e-6** relative and `rel=1e-6` fails on four of the seven lines. Our own
    > rows need no tolerance at all and get none -- the source values are known, so the
    > assertion compares the emitted text to `f"{expected:g}"` exactly. The first draft
    > used `rel=1e-6` on our rows and failed, two tests below a docstring warning about
    > precisely that.
    > **Note:** Mutation-checked against **nine** defects, all killed -- but two survived
    > the first pass and are the reason two tests exist. Swapping `Data DL` and `Model DL`
    > inside `_log_move_row` passed everything, because the start row and the move row are
    > laid out by two separate calls and only the first was pinned. And negating the
    > runner-up margin passed, because the only scripted round had one candidate and so
    > printed `-` rather than a number: **a sign is only pinned by a case that has one.**
    > **Note:** The merge row is scripted, not searched for. Every logged move in all three
    > oracles is a split, asserted as `{"-", "0 ^", "2 ^"}`, so whether this corpus would
    > ever accept a merge is a fact about the corpus where this is a test about the format.
    > **Note:** Scripting a round *and* passing `log=` needs real `HMMParams` in each
    > `Move.result`. The existing `_Script` puts a bare `object()` there, which is fine
    > while the walk only compares totals but would raise under a log, since `_halves`
    > calls `_model_description_length`. The new tests build real results instead.
  - [-] Document whatever this exports in `docs/api/hmm/`, with executed blocks per ADR 0013. Note that a log's output is host-independent here only if the numbers in the example are, which is the constraint `docs/benchmarks/` exists to take off these pages.
    > **Deferred:** there is nothing exported to document. `_search` is underscore-prefixed
    > and `pfsmgraph.hmm.__all__` is the same ten names it has carried since
    > `viterbi_batch`, so the README's own rule decides it -- "anything underscore-prefixed
    > ... is private, out of contract, and may change without notice" -- and its table has
    > no eleventh row to write. ADR 0013 scopes `docs/api/` to public surfaces. The page is
    > owed if and when the search is exported, which the new master-plan subgoal settles.
    > **Done:** the one thing that *was* owed is closed. `baum_welch.md` ended its `log=`
    > section pointing at a training log that "belongs with topology search" -- a forward
    > reference with no other end, the same defect goal 2 of this branch exists to fix for
    > `hmm-trainer-view.lsh`. It now says the search's log exists, why this page is the
    > wrong place for it, and that the two streams cannot interleave. Doc tests still 11.
    > **Note:** two staleness findings handed to the audit subgoal rather than fixed here.
    > `docs/api/hmm/README.md` names six private modules (`_params`, `_viterbi`,
    > `_baum_welch`, `_forward_backward`, `_backends`, `_numeric`) where there are now
    > **fifteen** -- `_mdl`, `_topology`, `_trials`, `_search` and the four kernels are
    > missing. Its public-surface *table* is correct; only the prose around it is not.
  - [x] Sync `docs/agents/core.md`, **last**, in one commit. Three counts are already stale
        as of the format tests -- `2962` twice and `3088` twice, now 2973 and 3099 -- and the
        docs subgoal above moves them again, since `tests/test_api_docs.py` executes every
        block a new page adds. The prose owes an update too: core.md's `baum_welch` paragraph
        says its `log=` is "not the Lush training log, **which is per-topology history for
        revision 04**", a forward reference this branch discharges.

- [ ] Record that `hmm-trainer-view.lsh` (237 lines) migrated **nowhere**, and why, so the omission reads as a decision rather than an oversight. It is the only file in `Code/HMMlib/` with no destination.
  - [ ] Decide where the record lives. Four places already name the file and none of them *is* the record: `HMMLIB-ACCOUNT.md` §11 (which says "it migrates nowhere" in passing, inside a section whose headline claim carries a 2026-09-17 correction), `DEFERRED.md:580` (which cites this revision for the reason), ADR 0017 and ADR 0023. The test is where a reader asking "what happened to the GUI?" would actually look.
  - [ ] Say **both** things the master plan asks for, since either alone misleads. It is "presentation only — every button calls a trainer method and then `update-view`" (§11), which is what makes dropping it lossless; and it is, beside `Training/hmm-train-new-nw`, one of the two written statements of the workflow `_search.py` ports, which is what is lost with it. Its fifteen buttons are the requirements list ADR 0023 read.
  - [ ] Check the record against the correction §11 already carries. "There is no headless entry point" is false and marked so; a record that leans on §11's framing without noticing would inherit a retracted claim.
