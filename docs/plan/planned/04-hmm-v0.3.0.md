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

**The criterion itself is an open question, and this revision does not answer it.**
The Lush implementation scores with a **two-part** code — model cost plus data cost — and
this revision reproduces it, because reproducing is what a migration means. Whether a
**refined one-part** code (NML / stochastic complexity) is the criterion this project
should end up with is registered at [PRD §8](../../design/PRD.md), *"Which description
length scores the topology search"*, along with why it cannot be answered here: exact NML
for HMM classes is intractable, and the tractable route — factorised NML over the
multinomial case — is a research decision rather than a porting one. The only obligation
it places on this revision is structural: **keep the criterion a seam.** If scoring is
inlined into the search driver, answering the question later means rewriting the search;
if it is a boundary in `_mdl.py`, it means substituting a function.

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

- [ ] Implement `_mdl.py`: `int_code_length`, `comb_code_length`, `entropy`, and the model description length — a **two-part** code, matching the original. Document *which* universal code for integers it uses; the choice is a research decision, and reproducing the search behaviour depends on it. **Give the total score a single named entry point** that the search driver calls and nothing else computes, so the two-part/refined question registered at [PRD §8](../../design/PRD.md) can later be answered by substitution rather than by rewriting the search.
- [ ] Implement state split (`try-split`, `suggest-split`, `hmm-trainer.lsh:738-843`), including the trial budget the `*min-split-trials*` and `*split-trials-per-state-p*` parameters imply, and settle the reproducibility contract if the search is stochastic.
- [ ] Implement state merge (`try-merge`, `suggest-merge`, `843-952`).
- [ ] Implement the search driver (`suggest-move`, `952-1073`): how a move is proposed, scored against the total description length, and accepted or rolled back. It must obtain that score by **calling** `_mdl.py`'s entry point, never by assembling it from the pieces — the driver decides *whether* a move wins, not *what winning costs*.
- [ ] Settle the **parameter representation**, of which the resize strategy is only one branch. There are three options, not two: reallocate on every accepted move; over-allocate and slice; or hold **no dense array at all** and keep an edge list, in which resizing is not the problem being solved. [ADR 0015](../../design/adr/0015-arc-emission-mealy-formulation.md) leaves this open deliberately — it fixes the model's semantics and not its storage — and it is also why the third option exists at all: under arc-emission the emission tensor is `(S, S, A)` and so quadratic in the number of states, measured at **62,500** entries against a Moore model's 1,250 for a 50-state model over `set11a_dInt`'s 25-symbol alphabet, and mostly empty for any sparse topology. Whichever is chosen, make the rollback path cheap, since the search rejects far more moves than it accepts. Measure before choosing; this is the one revision where the data structure, not the recurrence, is the cost.
- [ ] Report search progress on **standard output**, extending revision 03's training log with the move that was tried, its description length, and whether it was accepted. This is the run a user most wants to watch and the one most likely to prompt a dashboard; the decision is that it does not get one — see `DEFERRED.md`, `## Trigger: a second module needing training progress reporting`.
- [ ] Record that `hmm-trainer-view.lsh` (237 lines) migrated **nowhere**, and why, so the omission reads as a decision rather than an oversight. It is the only file in `Code/HMMlib/` with no destination.
- [ ] Release `pfsmgraph-hmm` 0.3.0.
