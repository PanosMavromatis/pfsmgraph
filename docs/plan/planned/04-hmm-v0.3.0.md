**Status**: planned — drafted 2026-09-03 on `docs/hmm-migration-plan`, not yet opened
**Splice**: everything from `## Subgoals` down is what `/open-revision` places into
[`docs/plan/TODO.md`](../TODO.md); this preamble stays behind and this file is removed.
It lives in `planned/`, not in `docs/plan/04-hmm-v0.3.0/`, because `/open-revision` refuses a
label whose directory already exists.

**Only metadata is above the heading, and deliberately so.** The falsifiers sit *below*
`## Subgoals` because the splice boundary would otherwise leave them here to be deleted
with this file — which is what happened when revision 02 was opened, and had to be
repaired by hand. Do not tidy them back up here: everything that must survive the
splice belongs under the heading.

## Subgoals — revision 04-hmm-v0.3.0

Topology search: the model learns its own shape by merging and splitting states. This is
335 lines of the Lush trainer (`hmm-trainer.lsh:738-1073`) and the part with no analogue
anywhere else in the family.

Two things make it a separate revision rather than a feature of the last one. First, the
**scoring criterion is minimum description length**, and the MDL machinery lives in
`Code/Utility/util.lsh`, not in the trainer — `int-code-length` (a universal code for
integers), `comb-code-length` (a log-binomial), and `calculate-entropy`. These encode
research decisions about what a model *costs*, and they are the reason a merge is ever
preferred to a better fit. Second, every accepted move **changes the size of every
parameter array**, so this revision is about matrix copying and resizing under a search
that mostly rejects — the trainer must be able to try a move, score it, and put the model
back unchanged, which `keep-model`/`reset-model` (`677-705`) exist to do.

Settled on the planning branch: `_mdl.py` is private to `pfsmgraph.hmm`. If `hseg` later
scores segmentations by description length, `DEFERRED.md`'s
`## Trigger: hseg needing description lengths` promotes it to a shared home; inventing a
sixth distribution before a second consumer exists is not warranted.

**Drafted before the source was read.** What would falsify the shape below:

- **The MDL criterion not being the whole story.** `int-code-length` and
  `comb-code-length` are called only from `update-model-dl` (`hmm-trainer.lsh:402-430`),
  which reads as the model cost in a description-length score. If `suggest-move`
  (`952-1073`) applies further criteria — a likelihood-ratio gate, a trial budget — the
  scoring subgoal widens.
- **`*min-split-trials*` and `*split-trials-per-state-p*` (`hmm-trainer.lsh:17-18`)**
  suggest split is *stochastic and retried*, not deterministic. If so, reproducibility
  becomes a first-class concern of this revision and `rand-p-vector`'s seeding is part of
  the public contract rather than an implementation detail.

- [ ] Implement `_mdl.py`: `int_code_length`, `comb_code_length`, `entropy`, and the model description length. Document *which* universal code for integers the original uses — the choice is a research decision, and reproducing the search behaviour depends on it.
- [ ] Implement state split (`try-split`, `suggest-split`, `hmm-trainer.lsh:738-843`), including the trial budget the `*min-split-trials*` and `*split-trials-per-state-p*` parameters imply, and settle the reproducibility contract if the search is stochastic.
- [ ] Implement state merge (`try-merge`, `suggest-merge`, `843-952`).
- [ ] Implement the search driver (`suggest-move`, `952-1073`): how a move is proposed, scored against the total description length, and accepted or rolled back.
- [ ] Settle the resize strategy — reallocate versus over-allocate-and-slice — and make the rollback path cheap, since the search rejects far more moves than it accepts. Measure before choosing; this is the one revision where the data structure, not the recurrence, is the cost.
- [ ] Report search progress on **standard output**, extending revision 03's training log with the move that was tried, its description length, and whether it was accepted. This is the run a user most wants to watch and the one most likely to prompt a dashboard; the decision is that it does not get one — see `DEFERRED.md`, `## Trigger: a second module needing training progress reporting`.
- [ ] Record that `hmm-trainer-view.lsh` (237 lines) migrated **nowhere**, and why, so the omission reads as a decision rather than an oversight. It is the only file in `Code/HMMlib/` with no destination.
- [ ] Release `pfsmgraph-hmm` 0.3.0.
