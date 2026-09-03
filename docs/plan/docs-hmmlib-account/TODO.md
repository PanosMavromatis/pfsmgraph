# docs/hmmlib-account

**Status**: active
**Created**: 2026-09-03
**Subgoal**: Read `Code/HMMlib/` in its own terms and write `.scratch/hmm-lush/HMMLIB-ACCOUNT.md`, following `ACCOUNT.md`'s conventions — measurements against the two tracked specimen corpora, and **provenance unknown** for behaviours the code admits but may never have exercised. Check the three falsifiers and revise the plan if any holds (revision `02-hmm-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

**The falsifier check is not a formality.** Revision 02's boundaries, and 03's and 04's
behind them, were drawn from a structural survey rather than from the source. Goal 2 is the
first time anything in this repository can contradict them, so a held falsifier is the
method working. Record the negative verdicts with the same care as a positive one: "checked,
does not hold" is a finding, while silence is indistinguishable from not having looked.

## Goals

- [ ] Read the four `HMMlib` files and write `HMMLIB-ACCOUNT.md`
  - [ ] Read `hmm.lsh` (319) and `hmm-param.lsh` (386) — the model and its parameters — before the trainer, so the trainer is read against a model that is already understood
  - [ ] Read `hmm-trainer.lsh` (1102), noting which of its regions belong to which revision: `21-126` scaffolding, `126-188` forward, `188-257` Viterbi, `257-346` M-step, `738-1073` topology search
  - [ ] Read `hmm-trainer-view.lsh` (237) far enough to say whether it is presentation only, and therefore whether it migrates at all
  - [ ] Follow `HMMlib`'s calls into `Code/Utility/` only as far as they go; the migration itself is subgoal 3, not this branch
  - [ ] Write the account to `ACCOUNT.md`'s conventions — **Sources** block with line counts and dates, structure before behaviour, an appendix collecting every measurement, and **provenance unknown** where the code admits a behaviour that may never have run

- [ ] Check the three falsifiers the master plan names, and record each verdict
  - [ ] **Does Viterbi read the forward variables?** The 02/03 split assumes `update-viterbi-path` (`hmm-trainer.lsh:188-257`) computes δ independently of the α that `update-data-p` (`126-188`) builds. If it reads α, the boundary moves and Viterbi drags the forward pass into 02 with it
  - [ ] **Is the stationary-distribution solve what it looks like?** `hmm-param.lsh:82` and `hmm.lsh:244` build a matrix from `int-delta` and call `LU-solve`; that reads as `(I - Pᵀ)π = 0`, inferred from two lines of context
  - [ ] **Is `hmm-trainer.lsh:21-126` separable?** The scaffolding is assumed shareable across 02 and 03. If the constructor demands the training apparatus, 02 gets no trainer at all and Viterbi becomes a free function over a model — which may be the better design regardless
  - [ ] Revise `docs/plan/TODO.md` for any falsifier that holds, and amend the affected `planned/` draft if the consequence reaches 03 or 04

- [ ] Record what the account changes for the subgoals downstream of it
  - [ ] Note anything subgoal 2 (the public surface) now has evidence for that it previously had only a survey of
  - [ ] Note anything subgoal 3 (the `Utility` migration) should expect, without doing it here
