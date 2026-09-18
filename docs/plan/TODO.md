# Master plan — pfsmgraph

**Status**: active

The two-tier plan convention for this repository:

- **Master plan** — this file, `docs/plan/TODO.md`, lives on `main`. It defines
  **revisions** (milestones) and their **subgoals**; each subgoal spawns a branch.
- **Branch plan** — `docs/plan/<type>-<slug>/TODO.md`, created by `/new-branch`, worked
  by `/hitl-step`, stamped `merged` by `/smart-merge`. It **survives on `main`** as the
  durable record of how a subgoal was executed, and is filed into its revision's
  directory by `/file-plans`.
- **PR body** — the distilled description plus a pointer to the branch plan directory.
  Not a verbatim archive.

`TODO.md` rather than `DO.md` is deliberate: branches inherit the master plan's model,
so every subgoal here runs under `/hitl-step` with its Q&A logged inline. The `dataseq`
merge reconciles three existing implementations and settles a public API that four other
packages depend on — the questions are genuinely open, and the answers are worth keeping.

**Related documents.** Decided-but-not-yet-actionable work is _not_ listed here; it lives
in [`DEFERRED.md`](DEFERRED.md), indexed by the trigger that unblocks it. Open design
questions live in [PRD §8](../design/PRD.md) and the `Open` sections of the
[ADRs](../design/adr/README.md). This file tracks work that is active now, plus the
revisions already drafted and waiting to be opened.

## Planned revisions

**None are planned today** — `planned/` is empty as of 2026-09-16, when the last of the
three drafts was spliced in. The convention stands for the next revision drafted ahead of
its opening, and is recorded here rather than in the empty directory, which git does not
track: one file per revision under `planned/`, carrying the same `## Subgoals` section
`/close-revision` archives, so opening one is a splice rather than an authoring job. A
draft sits in `planned/` rather than in `docs/plan/<label>/` because `/open-revision`
refuses a label whose directory already exists, on the sound assumption that only
`/file-plans` creates one; a draft filed there would make every planned revision look
already-opened. The detail lives in those files and not here, which is the point: this
file stays short enough to read on every session, and a revision's subgoals enter it only
while that revision is in progress.

**A draft is reconciled at opening, not copied.** Each of the three was written before the
Lush source was read, and each opening found assumptions that had moved — so the splice
carries a record of what changed and why, beside the draft's original reasoning rather
than in place of it. That record is the reason a stale draft is still worth keeping.

The `hmm` migration is **three** releases rather than one. The Lush trainer is 1,102 lines
spanning three problems that fail differently — a decode, a fixed-topology estimator, and
a search over model shapes — and each raises its own questions about parallelism and data
structures. Conflating them would put the project's first `.pyx`, its first EM loop and its
first resizing search in one revision, where a failure in any of them would be diagnosed
against all three.

The first two of the three are **closed** (see below), and `pfsmgraph-hmm` is released at
0.2.0; the third, `04-hmm-v0.3.0`, is **open** as of 2026-09-16 and its subgoals follow
this section. All three have now been opened, so nothing remains to move up.

All three were drafted from a structural survey of `.scratch/hmm-lush/Code/HMMlib/` —
definition maps and call-site counts — **before the source was read**. Each names the
findings that would falsify its boundaries, and revision 02's first subgoal is the reading
that checks them.

## Subgoals — revision 04-hmm-v0.3.0

Topology search: the model learns its own shape by merging and splitting states. This is
335 lines of the Lush trainer (`hmm-trainer.lsh:738-1073`) plus the 228 lines of parameter
surgery that back them (`hmm-param.lsh:142-369`), and the part with no analogue anywhere
else in the family. Opened 2026-09-16, the day `pfsmgraph-hmm` 0.2.0 was released, from the
draft written on `docs/hmm-migration-plan` (2026-09-03). That draft was written **before
the Lush source was read**; the reading happened during revisions 02 and 03 and is recorded
in `.scratch/hmm-lush/HMMLIB-ACCOUNT.md`. The section below is reconciled against it, and
the corrections are listed so a reader of the draft can tell what changed and why.

Two things make it a separate revision rather than a feature of the last one. First, the
**scoring criterion is minimum description length**, and the MDL machinery lives in
`Code/Utility/util.lsh`, not in the trainer — `int-code-length` (a universal code for
integers), `comb-code-length` (a log-binomial), and `calculate-entropy`. These encode
research decisions about what a model *costs*, and they are the reason a merge is ever
preferred to a better fit. Second, every accepted move **changes the size of every
parameter array**, so this revision is about matrix copying and resizing under a search
that mostly rejects — the trainer must be able to try a move, score it, and put the model
back unchanged, which `keep-model`/`reset-model` (`677-705`) exist to do.

**Settled, and not to be relitigated:**

- **`_mdl.py` is private to `pfsmgraph.hmm`.** If `hseg` later scores segmentations by
  description length, a shared home is reconsidered then; inventing a sixth distribution
  before a second consumer exists is not warranted. *(The draft cited a
  `## Trigger: hseg needing description lengths` heading in
  [`DEFERRED.md`](DEFERRED.md). No such heading exists — the file has no occurrence of
  "description length" at all, so the promotion path was never actually written down.
  Write that trigger as part of this revision's first subgoal, or restate the deferral
  under `## Trigger: hseg design settling`, which is where it would surface. Written 2026-09-16 on `feat/hmm-mdl` as `## Trigger: a second module scoring by description length`: named for the event rather than filed under `hseg`, since `dl` or `align` could be the second consumer, and scoped to the primitives rather than the whole module.)*
- **The criterion stays the original's two-part code, and this revision does not answer
  whether it should.** Whether a refined one-part code (NML / stochastic complexity) is
  what this project should end up with is registered at
  [PRD §8](../design/PRD.md), *"Which description length scores the topology search"*:
  exact NML for HMM classes is intractable, and the tractable route — factorised NML over
  the multinomial case — is a research decision rather than a porting one. The only
  obligation it places here is structural: **keep the criterion a seam.** If scoring is
  inlined into the search driver, answering the question later means rewriting the search;
  if it is a boundary in `_mdl.py`, it means substituting a function.
- **No alignment-seeded initialisation.** The search starts from a single state and grows
  by splitting, as the original does (`model-starting-size` defaults to **1**,
  `HMMLIB-ACCOUNT.md` §11). Seeding the topology from a multiple alignment is **its own
  later revision**, gated on `align` being able to produce one and expected at `hmm`
  0.4.0 — see [`DEFERRED.md`](DEFERRED.md), `## Trigger: align able to produce a multiple
  alignment`. It follows this search rather than replacing it, because the seed is a
  better *starting point for* merge/split. So nothing here imports `pfsmgraph-align`, and
  `hmm`'s metadata still declares no edge to it under
  [ADR 0019](../design/adr/0019-declared-dependencies-follow-imports.md).
- **Rollback is free, and the draft's open branch on this is closed.** The draft weighed a
  single mutable class (which "owes this subgoal an explicit save-point") against an
  immutable parameter object (which "gets rollback for free by construction").
  [ADR 0017](../design/adr/0017-frozen-parameter-object-for-hmm.md) settled it as the
  second: `HMMParams` is a frozen value, so trying a move means building a candidate and
  keeping or dropping the reference. Lush's `hmm`/`hmm-param` working-copy split — which
  existed precisely to make `keep-model`/`reset-model` cheap (`HMMLIB-ACCOUNT.md` §5) —
  is therefore **not** inherited, and this revision must not reintroduce it as a
  performance argument without measuring first.
- **No `1e100` sentinel.** `update-total-dl` (`430-433`) maps the Lush `-1` log-zero
  sentinel to `1e100` so an impossible model sorts last rather than best. Here `bits(0)`
  is `+inf`, which already sorts last and absorbs under addition, so the mapping
  dissolves — the same dissolution revision 02 applied to `safe->--log` and revision 03
  inherited. A ported `1e100` would be a magic number that is also *comparable*, which is
  strictly worse than the infinity it replaces.
- **The filing of `docs-hmm-migration-plan` is decided: it stays put.** `DEFERRED.md`'s
  `## Trigger: revision 03 or 04 opening` asked whether that plan moves, splits across the
  revisions it touches, or stays in `docs/plan/02-hmm-v0.1.0/` with its scope note. It
  stays: a branch plan is filed where it was *worked*, not where its consequences land,
  and it has no master-plan backlink to redirect. Close that entry. (Its other item,
  unifying `log2` across the Viterbi backends, was settled on 2026-09-14 and is already
  closed.)

**Falsifiers, reconciled at opening.** The draft named two; one fired, one is resolved,
and the reading turned up a third correction larger than either:

- **There is no search driver to port.** This is the draft's biggest misreading, and it
  affects the subgoal list rather than a detail. `HMMLIB-ACCOUNT.md` §11: the library
  offers `suggest-split`, `suggest-merge` and `suggest-move`, which **score candidates**,
  but "nothing in the tree loops over them automatically. A training run was a person
  watching the description length and pressing buttons." Both training scripts under
  `Training/` end by constructing a `HMMtrainerWindow` and waiting on it. So
  `suggest-move` (`952-1073`) is a scoring method, not a driver, and — in the account's
  own words — "anything in revision 04 that reads as 'the search strategy' will be a
  design decision being made for the first time, not a translation." The subgoals below
  therefore **separate the two**: porting the scored primitives is a migration and is
  reviewable against the original; the loop that calls them is new work and must be
  reviewed as a design.
  *(Corrected 2026-09-17, `feat/hmm-search-loop`: there **is** a driver to port, and §11
  was wrong to deny it. `Training/hmm-train-new-nw` and `hmm-train-load-nw` run
  `(repeat N (suggest-move) (keep-model))` headless, as does the view's `Continue for `
  button. `suggest-move` is still a scoring method, and the separation stands; what
  changes is that the loop is reviewed as a port whose acceptance test, stopping rule and
  trajectory selection are named departures, since the original keeps every best
  neighbour unconditionally and stops only at N.)*
- **The MDL criterion not being the whole story** — does not hold, and the pieces are
  already named. `HMMLIB-ACCOUNT.md` §8 reads `update-model-dl` (`402-427`) as
  `int-code-length(n-states-r) + int-code-length(d) + (1+n-states-r)·comb-code-length(d,
  1+n-states)`, plus `n-non-0-transitions × comb-code-length(d, 1+n-symbols)`, and
  `update-total-dl` (`430-433`) as the sum of the data and model halves — so the "single
  named entry point" the draft asks for is a shape the original already has. It also
  answers the draft's "document *which* universal code for integers": `int-code-length` is
  **Rissanen's universal prior for the integers**, `log₂ 2.865` plus the iterated
  logarithm while the term exceeds 1, and `comb-code-length(sum, m)` is
  `log₂(sum+m) + log₂ C(sum+m−1, m−1)`. What the falsifier asked about `suggest-move`
  applying *further* criteria is subsumed by the finding above: there is no automatic
  search to apply them in.
- **Split is stochastic and retried — confirmed, but the reproducibility cost is already
  paid.** `HMMLIB-ACCOUNT.md` §5: `split-state` (`hmm-param.lsh:142-215`) "randomizes
  every outbound emission fibre with `rand-p-vector … 0.1`", so the draft was right that
  `*min-split-trials*` and `*split-trials-per-state-p*` imply trials rather than a
  deterministic move. The consequence it feared does not follow, because revision 02 built
  `rand_p_vector(size, noise_width, rng)` with a **required** `numpy.random.Generator`,
  no default and no module state — reproducibility is structural here rather than merely
  available. What is still owed is the *contract*: whose generator the search takes, and
  where it enters the public API. Note the `0.1` noise width is also what breaks the
  symmetry that would otherwise stall EM, since `baum_welch`'s stopping rule can halt on
  an exactly uniform saddle.
- **A third Lush defect, undecided and this revision's to settle.**
  `HMMLIB-ACCOUNT.md` §5 records that `hmm-param.lsh:172` and `:262` seed the new initial
  distribution from the **stationary** one — `(new-init-state-p i (state-p ...))` — where
  the surrounding code is copying `init-state-p`, and `split-state` contradicts itself
  four lines later at `179-180` by using `(init-state-p state-n)`. The effect is that
  after **any** split or merge, every state not directly involved has its initial
  probability overwritten by its stationary probability, and the vector is not
  renormalized afterwards. Provenance unknown: consistent across both methods (weak
  evidence for intent), internally inconsistent within one of them (stronger evidence for
  a copy-paste slip). This is the same shape as the δ-seeding defect revision 02 fixed
  rather than reproduced, and unlike that one it **does** reach downstream — it perturbs
  the model every accepted move scores. Decide it explicitly in the split and merge
  subgoals, and record the divergence.

- [x] Write `_mdl.py`, and note that **revision 03 already built half of it**. Present today in `_baum_welch.py`: `_quantize(p, d)` (the original's `update-approx-*`, `round-using` half up then an ascending renormalisation) and `_data_description_length(params, records, d)` (`update-data-dl`, `346-399`), matched against the three tracked models' logged `data-dl` to 0.009 bits. `entropy` is in `_numeric.py` and **must not be re-implemented** — the draft listed it as new work. What is owed: `int_code_length` (Rissanen's universal prior), `comb_code_length` (`log₂(sum+m) + log₂ C(sum+m−1, m−1)`), the **model** description length (`update-model-dl`, `402-427`), and the **total** as the single named entry point the search calls and nothing else computes, so the PRD §8 question can later be answered by substitution rather than by rewriting the search. Two decisions come with it: whether the data half and `_quantize` **move** out of `_baum_welch.py` into `_mdl.py` (the seam argues yes; the existing tests argue for care), and how `d` is chosen (`suggest-d`), which is not a free parameter — §8 notes a transition counts as non-zero **after** quantization, so `d` is simultaneously the rounding grid and the code's argument, and it is rounding at `1/d` that drives small transitions to exactly zero and makes a sparse topology cheaper to describe than a dense one. The three tracked models' stored `d` is the oracle. Write the missing `hseg` promotion trigger in `DEFERRED.md` here too.
  > **Branch:** feat/hmm-mdl
  > **Done:** `_mdl.py` holds the whole criterion — both code lengths, the data half (moved in from `_baum_welch.py`), the model half, the total as a bare-float entry point, and `_suggest_d` — each reproducing a value the original saved: `data-dl` to 0.009 bits, `model-dl` to print precision, `_total_dl` to 0.013, stored `d` exactly. Left for later rather than decided: Brent's local minimum and subtracting totals, both now in the loop-design subgoal, and a shared home for the primitives, in `DEFERRED.md` — PR #43
- [x] Settle the **parameter representation**, of which the resize strategy is only one branch. There are three options, not two: reallocate on every accepted move; over-allocate and slice; or hold **no dense array at all** and keep an edge list, in which resizing is not the problem being solved. [ADR 0015](../design/adr/0015-arc-emission-mealy-formulation.md) leaves this open deliberately — it fixes the model's semantics and not its storage — and it is also why the third option exists at all: under arc-emission the emission tensor is `(S, S, A)` and so quadratic in the number of states, measured at **62,500** entries against a Moore model's 1,250 for a 50-state model over `set11a_dInt`'s 25-symbol alphabet, and mostly empty for any sparse topology. Add to that the `6·S²` reserved-fibre entries `HMMParams` requires to be exactly zero, which are dead by construction. Measure before choosing; this is the one revision where the data structure, not the recurrence, is the cost. Rollback is no longer part of this question — ADR 0017 answered it — so what remains is allocation cost under a search that rejects far more moves than it accepts.
  > **Branch:** exp/hmm-param-representation
  > **Done:** Storage stays as `HMMParams` is -- dense, frozen, rebuilt on every move -- because building a candidate is about 0.005% of a move's cost (0.17 ms at S=5, 1.2 ms at S=50, against 6.5-21 s to score it). Over-allocate and slice is ruled out by construction, since `_frozen` copies its inputs. The edge list's saving belongs to kernels iterating live arcs (8.7-12.1% per step on the trained models, skip-zero measured bit-exact), deferred in `DEFERRED.md`. Recorded as Resolved entries in ADRs 0015 and 0017; split and merge build fresh arrays and return a new `HMMParams` -- PR #44
- [x] Implement state split (`split-state`, `hmm-param.lsh:142-215`, and its trainer wrappers `try-split`/`suggest-split`, `hmm-trainer.lsh:738-843`): add one state at index `old-size`, halve the initial and inbound transition mass between the original and its new partner, copy the original's inbound emissions to the partner, and randomize every outbound emission fibre at noise width `0.1`. Settle the reproducibility contract the trial budget (`*min-split-trials*`, `*split-trials-per-state-p*`) makes load-bearing — whose `Generator`, and where it enters the public API. **Decide the `hmm-param.lsh:172` initial-distribution defect here** and record the decision; revision 02's oracles cannot exhibit it, since splitting is exactly what they never do. Note that halving initial probabilities without halving a topology is also what decouples `init_p == 0` from "cannot emit `begin`", which is the degenerate δ-seeding case revision 02 recorded as masked by the learned topology and unmasked by this operation.
  > **Branch:** feat/hmm-split-state
  > **Note:** `try-split`/`suggest-split` moved to "Port the scored primitives" (decided 2026-09-16 on `feat/hmm-split-state`, goal 6). `try-split` re-converges and chooses `d`, and both of those are the search-loop subgoal's decisions, so this subgoal delivers the split function alone. The split itself departs from the original on two points, recorded in [ADR 0022](../design/adr/0022-state-split-initialisation.md): `:172` is fixed, and symmetry is broken on inbound arcs rather than by redrawing outbound fibres.
  > **Done:** `_topology._split_state(params, state, *, rng)` implements ADR 0022, with 645 tests in `test_topology.py`. `:172` is fixed by carrying `init_state_p`. Symmetry is broken by a per-predecessor inbound perturbation (`w = 0.01`) and a 1% emission seed, both measured against the original's redraw in `split_symmetry_breaking.py`. The split takes a required `rng`, with a draw count that depends only on `S`. The zero-init `begin` decode case is constructed and tested. `try-split`/`suggest-split` moved to the scored-primitives subgoal — PR #45
- [x] Implement state merge (`merge-states`, `hmm-param.lsh:218-369`, with `try-merge`/`suggest-merge`, `hmm-trainer.lsh:843-952`): remove the higher-indexed of the two, remap indices through an `old-state-numbers` table, sum inbound transition mass, and take **stationary-probability-weighted** averages of the outbound transitions and emissions (`299-313`, `state-p` values renormalized to sum to one). Two consequences already recorded elsewhere land here and are worth expecting rather than discovering: `safe_divide` finally acquires consumers — the original uses `safe-/` throughout so a merge of two unreachable states yields zeros rather than dividing by zero — and a merge can produce a **reducible** chain, whose stationary solve `stationary_distribution` raises `ValueError` on. That failure surfaces on a `cached_property` access under ADR 0017, and since the merge *itself* consumes `state_p`, decide whether a move that makes the model's own scoring quantity undefined is rejected at proposal time or allowed to raise.
  > **Note:** By the rule decided for the split (`feat/hmm-split-state`, goal 6), `try-merge`/`suggest-merge` belong to "Port the scored primitives", and this subgoal delivers the merge function. `:262` repeats the split's `:172` defect, and ADR 0022 §1 already decides the fix: a merge sums the pair's `init` and every other state keeps its own.
  > **Branch:** feat/hmm-merge-states
  > **Done:** `_topology._merge_states(params, first, second)` carries `init` (the `:262` fix), weights by exact stationary mass, refuses two transient states, and is an exact lumping of the stationary chain, with 741 tests including a split-then-merge round trip. A merge cannot make a chain with one closed class reducible, so nothing is rejected at proposal time. It also makes `stationary_distribution` decide reducibility structurally, through `closed_classes`, which fixed 18 of 254 random reducible chains being solved silently, and write exact zeros on transient states — PR #46
- [x] Port the scored primitives as a migration: `suggest-split`, `suggest-merge` and `suggest-move` (`952-1073`) score candidate moves and return them ranked. This is reviewable line-by-line against the original. It must obtain a score by **calling** `_mdl.py`'s total entry point, never by assembling it from the pieces — the primitive decides *whether* a move wins, not *what winning costs*. A merge step tries all state pairs and so is quadratic in model size; record the measured cost, because it is what the deferred alignment seed later attacks at its input rather than in its inner loop.
  > **Note:** This subgoal now also owns `try-split` (`hmm-trainer.lsh:749-770`) and `try-merge` (`858-881`), the per-trial wrappers the split and merge subgoals originally named (decided 2026-09-16, `feat/hmm-split-state` goal 6). Two ADR 0022 obligations land here. Re-converging a split needs a minimum EM budget before `run-converge` may stop (§4); its size is the search loop's to set. Split trials report unrounded data description length beside the total (§5). The draw contract under its **Resolved** gives each trial its own `Generator`, tied to (round, state, trial).
  > **Note:** **Merge weights are a decision for this subgoal** (recorded 2026-09-17 on `feat/hmm-merge-states`, goal 4). `_merge_states` weights the pair's outbound arcs by stationary mass, as `merge-states` does, taking exact zeros on transient states. It refuses a pair of two transient states with `ValueError`. A transient state that is still occupied, such as a start state reached only through `init` (plausible under per-record training), is therefore merged with weight 0, and what it learned about its outbound arcs is discarded. Weights from expected occupancy would keep it, but they need records, which a candidate builder does not take. So `try-merge` should decide whether to reweight from E-step counts, or to filter transient pairs with the same recurrent mask.
  > **Branch:** feat/hmm-scored-primitives
  > **Done:** `_trials.py` ports `try-split`/`try-merge` as `_try_split`/`_try_merge`, scored by calling `_total_description_length` with the unrounded data length beside it, and `suggest-*` as rankings of every candidate. A stable sort keeps the original's winner and tie-breaks, impossible candidates rank last at `+inf`, and split generators are keyed by `(state, trial)`. `baum_welch` gains a public `min_cycles` floor (ADR 0022 §4). Merge weights stay stationary mass, since occupancy agrees to 0.003 on every tracked model. A merge round costs its pair count (1.3–1.7 s per pair up to `S = 16`), and `_suggest_d`'s numpy forward pass is more than half of each trial, a saving the loop's `d` decision should weigh — PR #47
- [x] Design the automatic search loop, as **a port of the original's headless loop with named departures**, and say which in the plan, the docstrings and the ADR. *(Corrected 2026-09-17: this read "mark it as new work… The original has no such loop"; `Training/hmm-train-new-nw` is that loop, `(repeat N (suggest-move) (keep-model))`, keeping every best neighbour unconditionally and stopping only at N — `HMMLIB-ACCOUNT.md` §11 is corrected.)* There is still no numeric oracle for a trajectory: what to decide is how a move is proposed, when a candidate is re-converged with `baum_welch` before scoring, the acceptance rule against the total description length, the rollback path, and the stopping criterion. Start from a single-state model, as `model-starting-size` does. `hmm-trainer-view.lsh`'s fifteen buttons, beside the `-nw` scripts, state the intended workflow and should be read as a requirements list rather than as a UI. Beware of presenting a departure as faithful; a reviewer who believes it is will check it against the wrong thing. **Decide how `d` is chosen, too.** `_suggest_d` (`feat/hmm-mdl`) reproduces the original's Brent search, which stops in a local minimum on the one-state model every search starts from: `d = 29`, where `d = 13` is 10.46 bits cheaper. So reproducing the original and finding the best `d` are two different choices, and this is where that gets decided rather than in the scoring branch. A global integer scan over the original's `[1, 10000]` costs up to 10000 evaluations per call against Brent's 18–26, and it fails the pinning test in `test_mdl.py`, which is how that choice should arrive. Note that a global scan would also make the optimizer's `1e100` sentinel unnecessary, since it only compares scores and never subtracts them. **And do not subtract totals without handling impossible models first.** `_total_description_length` returns `+inf` for an impossible model, which is right when totals are compared, but an acceptance rule written as a margin — accept when `incumbent - candidate` exceeds some number of bits — computes `inf - inf = nan` when both are impossible, and whether `nan` then accepts or rejects depends on how the comparison happens to be phrased: `diff > k` rejects, `not diff <= k` accepts. This is the trap `_suggest_d` hit inside Brent (`feat/hmm-mdl`, goal 5). Compare totals directly, or test for `inf` before taking the difference.
  > **Branch:** feat/hmm-search-loop
  > **Done:** `_search.py` ports `hmm-train-new-nw`'s loop under ADR 0023: a walk that keeps the best model visited, stops on patience, a required `max_rounds` or a dead end, and keys seeds by role. Trials choose `d` by a bounded exact scan (`d = 13` on the one-state start, against Brent's 29), give splits a 200-cycle floor, and score on a chosen `score_backend`; acceptance has no margin, since a candidate's total steps by 58–71 bits with its integer `d`. The first three rounds reproduce the Lush log's moves. PR #48
- [x] Settle how the topology search uses the compiled backends, and accept [ADR 0024](../design/adr/0024-search-compiled-work.md), drafted 2026-09-17 at Proposed. It comes **before the `docs/api/` audit** because it may add a public keyword to `baum_welch`, and before the training log because that goal designs around `baum_welch(..., log=)`. Three questions were asked after PR #48: does the search layer need compiling (no, ADR 0024 §1); how Cython relates to the parallel phases in a full search (they are alternatives chosen per call by name, not layers, and all but `torch` are bit-identical; §2 and §4); and should parallelizing the search be recorded for later (yes, along trials; §3). The profile that answered the first found the round's real cost: `baum_welch`'s convergence check runs the numpy forward pass whatever `backend=` says (`_baum_welch.py:302`, `_finish` at `:320`), which ADR 0023 §7's `score_backend=` could not reach. This **qualifies the release goal's expectation** from `HMMLIB-ACCOUNT.md` §12, that leaving the driver interpreted "costs little": true of the driver, which was running the one uncompiled forward pass.
  > **Note:** Its nine acceptance criteria and the round profile moved verbatim to the branch plan `feat-hmm-convergence-backend` when the branch opened (2026-09-17), which executes them as its goals: decide ADR 0024's Open item, (a) a separate `score_backend=` on `baum_welch` or (b) the phase derived from `backend=`, and accept the record; implement and test it with results `==` across the bit-identical phases; measure the round after the change and whether `torch` leaves the serial path; document the choice in `docs/api/hmm/baum_welch.md`; and record the `DEFERRED.md` trigger, the `HMMLIB-ACCOUNT.md` §12 note, the pointer from ADR 0023 §7 and the `core.md` sync. The profile also stands in ADR 0024's Evidence: one round took 12.7 s, 10.6 s of it in the numpy forward pass of the convergence check.
  > **Branch:** feat/hmm-convergence-backend
  > **Done:** ADR 0024 is Accepted on (a): `baum_welch` takes a `score_backend=` of its own, checked before any work, and both convergence checks run on it, so the last forward pass no backend could name now takes one. A measured round fell from 9.38 s to 1.70 s, and the Cython E-step is two thirds of what remains. 98 tests pin it, with a spy per equality test because every phase returns the same bits, mutation-checked against four defects. A torch search was measured *not* to leave the serial path: 66 candidates, same `d`, same EM cycles, bit-identical totals, at 350-400 times the cost on the CPU. A new `docs/benchmarks/` area holds both runs' logs; `DEFERRED.md` gains the parallel-trials and `device=` triggers; `HMMLIB-ACCOUNT.md` §12 and ADR 0023 §7 are qualified where they are read — PR #49
- [x] Report search progress on **standard output** by porting the original training log, which is this revision's rather than revision 03's — a division the draft predicted and revision 03 honoured. `update-training-log` and `training-log-line` (`hmm-trainer.lsh:457-477`) append one line per *accepted* move from `keep-model` (`677-690`), holding the size, the split (`n ^`) or merge (`i v j`) marker, `data-dl`, `model-dl`, `total-dl` and `d`, every one of which only `_mdl.py` and the search loop produce. Drop `test-data-dl`, which the original never assigned (all seven logged lines read `0`); the three tracked `_training_log` files are the format's oracle. Decide whether rejected moves are reported too, since the original records only accepted ones and the search rejects most. `baum_welch(..., log=)` already writes a header, a row per convergence check and a stop line to a text stream, so a search nests those under its moves unless it silences them — decide which, and note that a search re-converging every candidate would otherwise emit one such block per rejected move. Persisting the log beside a saved model stays `DEFERRED.md`'s model-persistence entry. This is the run a user most wants to watch and the one most likely to prompt a dashboard; the decision is that it does not get one — see `DEFERRED.md`, `## Trigger: a second module needing training progress reporting`.
  > **Branch:** feat/hmm-search-log
  > **Done:** `_search` takes a `log=` text stream: a header, a row for the starting model, one row per round, and a closing sentence naming the stop rule. `update-training-log`'s format ported less `test-data-dl`, verified byte-for-byte against all seven oracle lines *before* departing from those bytes for a table with four columns the original could not have had (`Round`, `Candidates`, `Runner-Up`, `Comments`). Rejected moves get no row, but each round reports its candidate count and the runner-up's margin, which is the one number ADR 0023's Open section says the criterion cannot supply. The nested `baum_welch(..., log=)` block is silenced. 11 tests pin the format and never a value, mutation-checked against nine defects, two of which exposed real gaps. `docs/api/hmm/` gets no page, since nothing is exported — which is what surfaced the export subgoal below — and `baum_welch.md`'s dangling pointer is closed instead. Suite 3099 — PR #50
- [x] Record that `hmm-trainer-view.lsh` (237 lines) migrated **nowhere**, and why, so the omission reads as a decision rather than an oversight. It is the only file in `Code/HMMlib/` with no destination. Two things make the record worth more than a line: it is "presentation only — every button calls a trainer method and then `update-view`" (§11), which is what makes dropping it lossless, and it is, beside `Training/hmm-train-new-nw`, one of the two written statements of the workflow the search loop above ports. Say both.
  > **Branch:** feat/hmm-search-log
  > **Done:** `HMMLIB-ACCOUNT.md` gains §16, "What migrated nowhere", beside the source table that lists the view at 237 lines. It says both required things: dropping the file loses no capability, since every button is a lambda calling one trainer method and then `update-view`; and what is lost is a statement of intent, the more complete of the two the library has, since the scripts state the loop while the buttons state everything a person could do between iterations of it. §11 is repaired on the way past — retitled from "There is no headless entry point", a claim its own body retracts, and its button list corrected from thirteen to fifteen. The two missing were `Keep d` / `Reset d`, which complete the commit-and-rollback pattern ADR 0017 cites this file for: that record argues from "two buttons" where there are four. ADR 0023's evidence line is qualified where the fix made it stale — PR #50
- [ ] Decide whether revision 04 **exports** the topology search, and record the decision either way. Found 2026-09-17 on `feat/hmm-search-log`, while asking where the training log should be documented: **no subgoal of this revision makes it public.** As planned, 0.3.0 ships what `core.md` and the PRD call "topology search by state merge and split" with `pfsmgraph.hmm.__all__` unchanged at the ten names it has carried since `viterbi_batch`, and `_search`, `_trials`, `_topology` and `_mdl` all private and out of contract. That may well be right — [ADR 0023](../design/adr/0023-topology-search-loop.md)'s Open section leaves the near-optimum ranking unresolved and PRD §8 leaves the criterion itself an open research question, so freezing a public signature now has a real cost, and the seam exists precisely so the criterion can be swapped. But it was never asked, and two documents lean the other way: the audit subgoal below says the API check covers "whatever the search loop and training log export", and `docs/api/hmm/baum_welch.md` points at a training log that "belongs with topology search". **Decide before the audit**, since the audit checks `docs/api/` against what 0.3.0 actually ships, and before the release, since `__all__` is frozen by a published version. If the answer is to export, this subgoal owns the names, `docs/api/hmm/search.md` with executed blocks per ADR 0013, and the README's public-surface table; if it is not, it owns the sentence that says so and the reason, so the next revision does not have to rediscover the question.
  > **Branch:** feat/hmm-search-export
- [ ] Review the whole `hmm` codebase and audit `docs/api/` against what 0.3.0 ships. Revision 03's audit (`docs/hmm-api-audit`, PR #39) covered the documentation alone. This one reads the code as well, because revision 04 is the first to add modules that no differential oracle checks: the split departs from the original on purpose (ADR 0022), and the search loop departs from its original on purpose. **Code:** read every module under `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/` as a whole. Look for private helpers left without a consumer, which revision 04 is expected to *remove* from that list for `safe_divide`, so check that it did. Look for docstrings that still describe a subgoal's intermediate state, such as `_topology.py`'s "state merge next". Check that every ADR, `HMMLIB-ACCOUNT.md` and Lush line reference still points at what it names, and that no module presents new work as a port. Check the draw-order and `Generator` contracts end to end, from the public search call down to `_split_state`. Confirm the ADR 0003 backend header is unchanged from 0.2.0, and do not assume it. Check that the test counts, module count and ADR count in `core.md` match `uv run pytest` and the tree. **Docs:** check every name in `pfsmgraph.hmm.__all__` against its page, together with every parameter, return field and raised error of each public call. The search loop and training log export **nothing**: [ADR 0025](../design/adr/0025-topology-search-not-exported.md) keeps `__all__` at ten names at 0.3.0, so this check is against the same ten 0.2.0 shipped, and revision 04's four new modules are audited as private rather than as documented surface. Also check the `hmm/README.md` index, contracts and example against a package that now searches topology as well as trains; `docs/api/README.md`; and `packages/pfsmgraph-hmm/README.md`, which becomes the PyPI long description under an immutable version. [ADR 0013](../design/adr/0013-api-documentation-layout-and-tooling.md) still governs, so every code block is executed with its output pasted, and `tests/test_api_docs.py` stays the guard.
  > **Note:** placed second to last on purpose, as revision 03's audit was: run after every feature subgoal, it reviews the surface that actually ships rather than one the merge, scoring and search subgoals would re-invalidate. Added 2026-09-16, after the state split merged (PR #45).
- [ ] Release `pfsmgraph-hmm` 0.3.0. It follows 0.2.0's CI path — `just release-ci`, since `pfsmgraph-hmm` is in `ci_built_packages` and `just release` refuses it — and it is the first release to exercise that path from a version the tree bumped at opening rather than at the release commit. Expect **no** new ADR 0002 lifecycle phases: `HMMLIB-ACCOUNT.md` §12 records that not one topology-search method was compiled in the original, on the reasoning that "search *drives* compiled work — a split trial calls `run-converge`, which is compiled — so leaving the driver interpreted costs little". If that holds, 04 is the first `hmm` revision that adds no backend row, the `dp-compile` gate never arms, and the backend header is unchanged from 0.2.0's. Verify it rather than assuming it, and check the four-file release invariant plus `license-files` are still intact.

## Closed revisions

Extracted by `/close-revision` once finished. The subgoals, their `> **Done:**` records,
and the branch plans that executed them all live in the revision's directory.

- **Revision 01-dataseq-v0.1.0** — closed. See `docs/plan/01-dataseq-v0.1.0/_TODO.md`.
- **Revision 02-hmm-v0.1.0** — closed. See `docs/plan/02-hmm-v0.1.0/_TODO.md`.
- **Revision 03-hmm-v0.2.0** — closed. See `docs/plan/03-hmm-v0.2.0/_TODO.md`.
