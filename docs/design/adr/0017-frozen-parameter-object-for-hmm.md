# 0017. `pfsmgraph.hmm` parameters are a frozen value, not a mutable model object

- **Status:** Accepted
- **Date:** 2026-09-03
- **Source:** none in the PRD — postdates it, like 0012–0016. Decided under revision
  `02-hmm-v0.1.0`'s public-surface subgoal.

## Context

The Lush original organises the HMM as three classes: `hmm` (`hmm.lsh`), the committed
model; `hmm-param` (`hmm-param.lsh`), a near-duplicate of its parameter slots plus a
back-reference; and `hmm-trainer`, which mutates the working copy and commits it with
`keep-model` or discards it with `reset-model`. A migration that does not decide the
question inherits that shape by default, which is the outcome this record exists to
prevent.

The decision cannot be made on what 0.1.0 uses, because 0.1.0 uses almost none of it.
There is no trainer object in this release — `HMMLIB-ACCOUNT.md` §15 establishes that the
Lush trainer's constructor is not separable from the training apparatus, so "a decode-only
use of this library is not expressible in its own terms" and revision 02 builds no trainer
at all. A working copy with nothing to be a working copy *for* is not a design; it is
inherited furniture. The choice has to be made against what revisions 03 (Baum-Welch on a
fixed topology) and 04 (topology search by state merge and split) will actually demand,
and recorded now, because both of those revisions are written on the assumption that this
question was settled here.

Revision 04 is where the pressure is real. Every accepted merge or split **changes the
size of every parameter array**, and the search rejects far more moves than it accepts, so
the cost that matters is the cost of *trying* a move and putting the model back. Lush's
answer to that was precisely the `hmm`/`hmm-param` split. Whether that answer is a good
one, or an artefact of the environment it was written in, is the question.

## Decision

**`pfsmgraph.hmm` represents model parameters as a single frozen value.** One type holds
the three arrays the arc-emission formulation of [ADR 0015](0015-arc-emission-mealy-formulation.md)
requires — `init_state_p` of shape `(S,)`, `transition_p` of shape `(S, S)`, and
`output_p` of shape `(S, S, A)`, indexed by source state, destination state and symbol —
and holds nothing else that can change.

Four properties follow from that sentence and are part of the decision rather than
consequences of it:

**The buffers are read-only, not merely the bindings.** A frozen dataclass prevents
rebinding an attribute; it does nothing whatever about writing through the array reference
it holds. Each array is marked `writeable = False` at construction. Without that the type
is immutable in name only, and — the reason this is stated in the Decision — the omission
raises nothing, produces no warning, and is invisible until some caller mutates a
parameter set another caller is still holding.

**Derived quantities are computed, never stored.** The stationary distribution `state_p`
(the row-replaced solve of `(Pᵀ - I)π = 0` described in `HMMLIB-ACCOUNT.md` §4), the
per-state entropies, and the model entropy are cached properties of the parameters, not
slots alongside them. In the original they are slots, refreshed by a manual
`update-entropy` call after any change.

**The vocabulary the symbol axis was sized against is held with the arrays.** `output_p`'s
third axis is indexed by symbol code, and a code means nothing without the table that
assigned it: two different 25-symbol vocabularies produce parameter sets of identical
shape and incompatible meaning, and no check on the codes alone can tell them apart. The
type therefore holds a `Vocabulary` — `dataseq`'s Protocol rather than the concrete
`SymbolTable`, matching how `SequenceDataset` types its own parameter — while `n_symbols`
stays derived from `output_p.shape[2]` rather than stored beside it. This does not weaken
the sentence above: a `SymbolTable` is frozen by construction with no method that adds a
symbol, so it is not something that can change. It is a parameter of the model in the
sense that matters, not bookkeeping about a run — without it the emission tensor does not
denote anything.

**Algorithms take parameters; parameters do not own algorithms.** Viterbi receives a
parameter value rather than being a method that reaches into `self` for it, and revision
04's `split_state` and `merge_states` return a new value rather than mutating one in
place. The public entry point takes a `dataseq` `SequenceRecord` and returns a result
value; the array-level kernel that each [ADR 0003](0003-one-parameterized-test-suite-per-algorithm.md)
backend implements takes plain arrays, so a backend never touches a dataclass. Lush's
remaining slots — `name`, `counter`, `d`, `training_log` — are bookkeeping about a
training run, not parameters of a model, and do not join this type; where they are needed
they belong to whatever owns the run.

## Consequences

### Positive

**Rollback is dropping a reference, and its cost does not depend on the accept rate.**
Revision 04 tries a move, scores it, and either keeps the new value or continues from the
old one it never overwrote. There is no save-point to take, no restore path to test, and
no state in which a half-applied move can be observed.

**A whole class of staleness bug becomes unrepresentable.** In the original, `state-p`,
`state-entropies` and `entropy` are stored beside the parameters they are derived from and
kept correct by discipline: `init-random` "fills … then recomputes the entropies", and
every other mutation path owes the same call. A path that mutates and forgets leaves a
model whose entropy describes parameters it no longer has, and nothing detects it. Derived
quantities computed from an immutable value cannot disagree with it.

**The duplication that the split forced does not arise.** `update-entropy` exists twice in
the original — `hmm.lsh:228-262` and `hmm-param.lsh:66-100`, 35 lines identical but for
one symbol, the stationary solve included — because the model and its working copy each
need it. With one type there is one stationary solve. This matters beyond line count:
`HMMLIB-ACCOUNT.md` §13 records the forward recurrence appearing three times and the
stationary solve twice as the migration's most likely hiding place for a translation
error, "similar enough to skim and different enough to matter."

**It is the shape that survives not knowing the search strategy.** See Evidence: no
accept/reject ratio was ever measured, because the original's search was driven by hand.
An architecture whose cost is insensitive to that ratio does not have to be revisited when
revision 04 discovers what the ratio is.

### Negative / costs

**`writeable = False` is a discipline this record creates.** It must be applied at every
construction site, including the ones revisions 03 and 04 add. It fails silently when
omitted, which is why it is in the Decision above; a test that asserts a constructed
parameter set rejects writes is the cheap guard and should exist.

**Every EM iteration allocates a parameter set.** Revision 03's M-step produces a new value
rather than re-estimating in place. This is judged negligible — an allocation of
`(S,S,A)` against a forward-backward sweep over the entire corpus — but it is a real cost
and the judgement is recorded here so that a later measurement can contradict it rather
than having to reconstruct the reasoning. Immutability at the public type says nothing
about accumulator buffers *inside* an M-step, which stay mutable and local.

**Revision 03's `torch` backend has to bridge.** `nn.Parameter` is mutable and carries
`.grad`, so the autograd backend builds transient parameters from the frozen value and
returns a new frozen value at the end of the step. Revision 03's plan already assigns that
friction to its own `torch`-backend subgoal rather than to the base type, and this record
does not move it.

**A large model cannot be updated in part.** Nothing in revisions 02–04 wants to, but a
future caller that wants to adjust one transition without rebuilding the arrays will find
this type unhelpful, and would need a builder alongside it.

## Alternatives considered

**One mutable class holding the arrays** — the shape closest to a PyTorch `nn.Module`, and
the smallest surface for a release that has no trainer. Rejected on what it defers rather
than on what it costs now: revision 04 would owe an explicit save-point and restore path,
and the failure mode of forgetting to restore after a rejected move is silent corruption
of the model rather than an exception. It also carries the staleness discipline for derived
quantities forward unchanged. The convenience is real and arrives in 0.1.0; the cost
arrives in 0.3.0, where it is paid by the revision least able to afford another source of
silent error.

**Reproducing Lush's `hmm` / `hmm-param` split** — faithful to the source, which is
normally this migration's default. Rejected because its motivation does not survive the
port and its cost does. The motivation is a user-interface affordance: `hmm-param` exists
so `hmm-trainer-view.lsh` can offer `Keep model` and `Reset model` buttons against a
speculative split, and that file — 237 lines of `WindowObject`, presentation only —
migrates nowhere (`HMMLIB-ACCOUNT.md` §11). The cost is the 35 duplicated lines above,
and the §5 defect that both surgery methods seed the new initial distribution from the
stationary one — with `split-state` contradicting itself four lines later — lives inside
exactly that duplicated surface. This is [ADR 0010](0010-dataseq-composition-merging-three-implementations.md)'s
rule applied to `hmm`: the imports are evidence about what has been tried, not a source of
decisions.

**A frozen parameter value wrapped in a mutable model object**, so callers hold a stable
handle whose contents can be swapped. Rejected as premature: it is a straightforward
addition on top of this decision if a caller ever wants identity to outlive a parameter
change, and nothing in revisions 02–04 does.

## Evidence

**Observed** in the imported source, and checkable against it:

- `hmm-param.lsh:153-158` allocates a complete fresh parameter set — `new-state-p`,
  `new-state-entropies`, `new-init-state-p`, `new-transition-p`, `new-output-p` — and
  `210-215` rebinds every slot to it. `merge-states` does the same at `238-243`. **The
  mutable working copy does not mutate in place on the path it exists to serve.** It
  cannot: every accepted move changes array shape, and a shape change is a reallocation
  whatever the surrounding design. What `split-state` performs is construction of a new
  parameter set followed by rebinding — which is this ADR's decision, minus the freezing.
- `HMMLIB-ACCOUNT.md` §12: `hmm-param.lsh` compiles all six of its methods, `split-state`
  and `merge-states` among them, while every topology-search *driver* method is left
  interpreted. So the allocate-and-rebind path above is in the hot set, not an incidental
  corner where allocation was never a concern.
- `HMMLIB-ACCOUNT.md` §11: **topology search in the original was driven by hand.** The
  library scores candidates through `suggest-split`, `suggest-merge` and `suggest-move`,
  but nothing in the tree loops over them; a training run was a person watching the
  description length and pressing buttons.

**Measured:** 35 verbatim-duplicated lines between `hmm.lsh:228-262` and
`hmm-param.lsh:66-100`, differing in one symbol (`alphabet-size` versus
`:model:alphabet-size`), the stationary linear solve included.

**Reasoned, not observed:** that an architecture insensitive to the accept/reject ratio is
the right choice under ignorance of it. The ratio is genuinely unknown — §11 above means
no ratio was ever measured, so revision 04's expectation that "the search rejects far more
moves than it accepts" is a forward design assumption about a strategy this project writes
for the first time, not a translated fact. Immutable rollback costs the same at one
acceptance in five as at one in five hundred; a save-point's cost and its
forgotten-restore risk both grow exactly as trials become cheaper and more numerous.

## Open

**Whether this type is also what revision 04 stores.** That revision has to choose among
three parameter representations — reallocate per accepted move, over-allocate and slice,
or hold no dense array at all and keep an edge list — and is instructed to measure before
choosing, since the arc-emission tensor is quadratic in the state count (62,500 entries at
`S=50`, `A=25`, against a state-emission model's 1,250) and mostly empty for a sparse
topology. This record fixes that parameters are a frozen value and that moves return a new
one; it does not fix what the value holds internally. [ADR 0015](0015-arc-emission-mealy-formulation.md)
leaves storage open for the same reason, and an edge-list representation is as freezable as
a dense one.
