# 0015. The HMM is arc-emission (Mealy): symbols are emitted on transitions

- **Status:** Accepted
- **Date:** 2026-09-03
- **Source:** — postdates the PRD; no §9 counterpart.

## Context

Nothing in the PRD or in ADRs 0001–0014 says which HMM formulation `pfsmgraph-hmm`
implements. The packaging records treat dynamic programming abstractly, and the
algorithm records ([0002](0002-three-phase-algorithm-lifecycle.md),
[0003](0003-one-parameterized-test-suite-per-algorithm.md)) constrain how a kernel is
built and tested rather than what it computes. The question went unasked because both
answers are called "an HMM".

They are not interchangeable. In the **state-emission (Moore)** formulation — the one in
Rabiner's tutorial, in every textbook, and in every library API — a symbol is emitted
while *occupying* a state, and the emission parameters form a `B[state, symbol]` matrix.
In the **arc-emission (Mealy, transducer)** formulation, a symbol is emitted while
*crossing* a transition, and the parameters are indexed by source state, destination
state and symbol together. The two describe the same string distributions under stated
conditions, but they differ in parameter count, in the shape of every array a caller
touches, and in what can be hoisted out of an inner loop.

Two independent lines of evidence say this family is arc-emission, and they were produced
without reference to each other:

- **Reading the source.** `.scratch/hmm-lush/HMMLIB-ACCOUNT.md` §1–§2 (2026-09-03) finds
  `output-p` declared `-idx3-` and allocated `(float-matrix size size alphabet-size)`,
  with every read in all four files spelled `(output-p state-i state-j symbol-k)`.
- **A design conversation predating the reading.**
  [`arc-emission-hmm-handoff.md`](../arc-emission-hmm-handoff.md) §1 states the
  arc-emission commitment as intent, with its lineage — the IBM/information-theoretic
  tradition (Jelinek; Bahl, Jelinek & Mercer) rather than the Rabiner mainstream that
  crystallised later.

The decision has to be recorded **now**, before revision `02-hmm-v0.1.0`'s second
subgoal, because that subgoal settles the public surface of `pfsmgraph.hmm` 0.1.0 and the
shape of the emission array *is* part of that surface. Recording it later would mean
either revising a published API or letting the most consequential fact about the model go
unstated in the one document a consumer reads.

It is also already partly shipped. `dataseq`'s `seq-state` carries `symbol-data` of
length *N* alongside state arrays of length *N+1* — a path emitting *N* symbols visits
*N+1* states. `ACCOUNT.md` §6 recorded that `+1` as exact without being able to say why it
was there; it is this decision, seen from the container side.

## Decision

**`pfsmgraph.hmm` models are arc-emission.** A symbol is emitted on the transition
`i → j`, so the emission parameter is `output_p[i, j, symbol]` and not `B[state, symbol]`.
A path over *N* symbols visits *N+1* states, and the trellis is indexed by the gaps
between symbols rather than by the symbols.

This is a decision about **model semantics**, not about storage. Whether the parameters
live in a dense `(S, S, A)` array or in an edge list is deliberately left open below.

The conceptual frame it serves is a Newell & Simon problem space: a *state* is a
situation calling for an action, an *arc* is an operator that emits a token and produces
the successor situation. That maps term-for-term onto a weighted finite-state transducer
— states are situations, arcs are weighted operators, a derivation is a path, and path
weight is sequence probability. Arc-emission is the formulation in which the model and
the intended reading of the model are the same object.

## Consequences

### Positive

- **The domain framing survives into the code.** An operator that emits a token *is* an
  arc with an emission on it. Under Moore the emission would have to be attributed to the
  situation the operator produces, which is a different claim about the music.
- **WFST machinery applies directly**, if it is ever wanted: composition, weight pushing,
  epsilon removal. Under Moore these would each need a translation step first.
- **Alignment gaps are expressible as epsilon-emitting arcs.** This is what makes an
  alignment-derived seed for topology search a coherent construction rather than an
  analogy, and it is the reason `GAP` is already reserved at index 4 by
  [ADR 0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md). See
  [`DEFERRED.md`](../../plan/DEFERRED.md),
  `## Trigger: align able to produce a multiple alignment`.
- **No change to a released package.** `dataseq` 0.1.0's *N*/*N+1* geometry is already the
  shape this requires, so the decision is recorded rather than paid for.
- **A Moore model can still be embedded, cheaply.** Lifting state-emission to
  arc-emission is the easy direction — replicate each state's emission onto its incoming
  arcs — so a state-emission library can serve as a **test oracle** on the reduced case
  where the emission depends only on the destination state, checking the forward-backward
  against `hmmlearn`'s `CategoricalHMM` before it is trusted under the search. This is
  available to revision `03-hmm-v0.2.0` and is **not yet a subgoal of it**; see
  [`arc-emission-hmm-handoff.md`](../arc-emission-hmm-handoff.md) §3.

### Negative / costs

- **Emission parameters are quadratic in the number of states, not linear.** Measured on
  the tracked specimen corpus `set11a_dInt` (alphabet 25): a 50-state model carries
  **62,500** emission parameters where a Moore model of the same size carries **1,250**.
  Dense storage stops being the obvious default much earlier than it would otherwise.
- **The emission factor cannot be hoisted out of the inner loop.** It depends on both
  endpoints of the transition, so the standard optimisation — precomputing one emission
  column per timestep and reusing it across source states — is unavailable. Both the
  forward recurrence and Viterbi carry `transition_p[k, j] * output_p[k, j, symbol]` as a
  single indivisible term.
- **Every textbook and every library disagrees.** Rabiner 1989, `hmmlearn`, `pomegranate`
  are all state-emission, so the literature has to be translated on the fly and no library
  can be used as a drop-in replacement for any part of this. `docs/api/hmm/` will have to
  say this in its first paragraph, because a reader arriving with the standard model in
  mind will misread every array shape.
- **More parameters for the same distribution means more data, or stronger priors.** The
  arc↔state correspondence is distributional, not parameter-count-preserving.
- **Model-selection results are not transferable — under a two-part code.** A description
  length computed for a state-emission HMM is not a baseline for one of ours, because
  parameter count is exactly what such a criterion sees. This bites revision
  `04-hmm-v0.3.0`, whose entire search is scored by description length. Note the
  qualification: under a *refined* (NML) code the penalty counts distinguishable
  distributions rather than raw parameters, so the discrepancy might narrow or vanish.
  Which code this project uses is open — [PRD §8](../PRD.md), *"Which description length
  scores the topology search"* — and the interaction between that question and this
  record is one of the reasons it is worth answering.
- **It is effectively irreversible after the first release.** The formulation is visible
  in every array shape a caller touches and in every persisted model.

## Alternatives considered

- **State-emission (Moore).** Rejected on three counts. It would make the migration a
  redesign rather than a port, since the imported implementation is arc-emission
  throughout. It would lose the operator/situation reading that motivates the model. And
  converting arc → state is the *hard* direction: arcs entering one state with different
  emissions force that state to be split, which fights the topology search that is this
  project's novel content rather than composing with it.
- **Supporting both behind an abstraction.** Rejected. The two formulations index
  differently in the innermost loop, which is precisely where
  [ADR 0002](0002-three-phase-algorithm-lifecycle.md) drives the code toward compiled
  phases and [ADR 0001](0001-encode-at-the-boundary.md) requires integer-only kernels with
  no Python objects. An abstraction there costs in the one place the family has decided
  not to spend, and buys a formulation nothing else in the project wants.
- **Deferring the decision to revision 03, when Baum-Welch lands.** Rejected: revision 02
  settles the public surface, and the emission array's shape is part of it. Deferring
  would mean shipping 0.1.0 with the question open in the one document that has to answer
  it.
- **Adopting `hmmlearn` or `pomegranate` and working around the formulation.** Rejected.
  Neither can represent arc-emission without fighting its model definition, and neither
  offers the merge/split topology search that is the reason this package exists — so the
  workaround would be paid for nothing.

## Evidence

**Observed** in the imported source, and reproducible from it:

- `output-p` is declared `((-idx3- (-float-)) output-p)` at `hmm.lsh:40` and allocated
  `(float-matrix size size alphabet-size)` at `hmm.lsh:86`. Every read across all four
  files is three-index. (`HMMLIB-ACCOUNT.md` §1.)
- `delta` and `psi` are `(1+ data-seq-size) × size` (`hmm-trainer.lsh:197-198`), and the
  symbol consumed entering trellis position `i` is `(d-seq (1- position-i))` — the *N+1*
  geometry from the trainer side. (`HMMLIB-ACCOUNT.md` §2.)
- `dataseq`'s `seq-state` holds `symbol-data` of length `size` against `path-states` of
  length `size + 1`. (`ACCOUNT.md` §6, recorded before the reason was known.)
- A search of `Code/HMMlib/` and `Code/Utility/` for `epsilon|silent|null-|empty-symbol`
  returns **zero hits** (2026-09-03). The imported implementation has no silent
  transitions, so the epsilon machinery the seed construction needs is new work and not
  part of any migration subgoal.

**Measured:** 62,500 against 1,250 emission parameters, as above.

**Reasoned, not observed:** the arc ↔ state equivalence conditions, the claim that the
correspondence is not parameter-count-preserving, and the direction-of-difficulty
argument. These rest on the literature recorded in
[`references.md`](../references.md) under *Probabilistic automata and HMM equivalence*,
whose bibliographic details are verified and whose contents are **not** independently
checked. Revision 04 should confirm the parameter-count claim against Dupont, Denis &
Esposito before an MDL score is published on its authority.

## Open

- **Dense array versus edge list.** This record fixes the semantics and not the storage.
  A dense `(S, S, A)` tensor is what the imported code uses and is mostly empty for any
  sparse topology; an edge-list representation is the natural fit for a model whose shape
  changes under search. Left to revision `04-hmm-v0.3.0`, whose resize subgoal is where
  the cost is actually paid — noting that this ADR adds a third option to the two that
  subgoal currently names, and it is the option in which resizing is not the problem.
- **Whether the reduced case is public.** An arc-emission model whose emission depends
  only on the destination state is a state-emission model. That reduction is wanted as a
  test oracle in revision 03; whether it is also exposed as a constructor for callers who
  want a conventional HMM is not decided here.
- **Whether [ADR 0002](0002-three-phase-algorithm-lifecycle.md)'s phase 2 suits this
  family of kernels.** Arc-emission pushes toward scatter/gather over arcs rather than
  loops over a dense matrix, which is a poor fit for single-threaded Cython and a good one
  for batched tensor ops. Bracketed with the anti-diagonal question already deferred to
  revision 02's phase-3 subgoal, since both are the same doubt about that ADR's claimed
  universality.
