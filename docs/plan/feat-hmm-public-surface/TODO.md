# feat/hmm-public-surface

**Status**: active
**Created**: 2026-09-03
**Subgoal**: Settle the public surface of `pfsmgraph.hmm` 0.1.0 and where it meets `dataseq` (revision `02-hmm-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [x] Decide the class architecture
  > **Done:** Frozen parameter value, recorded as
  > [ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md). Lush's
  > `hmm`/`hmm-param` split is not inherited — its surgery methods reallocate rather than
  > mutate in place (`hmm-param.lsh:153-158`, `210-215`), so the mutability never bought
  > what it appeared to.
  > **Q:** Which class architecture should `pfsmgraph.hmm` adopt — a frozen parameter
  > object, one mutable class, or Lush's `hmm`/`hmm-param` split reproduced?
  > **A:** The frozen parameter object. `HMMParams` frozen over `init_state_p`,
  > `transition_p` and `output_p` with `writeable=False` buffers; `state_p` and the
  > entropies as cached properties; Viterbi takes it; revision 04's topology moves return
  > a new one, so rollback is dropping a reference. Lush's bookkeeping slots (`name`,
  > `counter`, `d`, `training_log`) do not join it.
  - [x] Weigh Lush's `hmm`/`hmm-param`/`hmm-trainer` split (GUI-undo motivation, measured duplication cost per `HMMLIB-ACCOUNT.md` §5) against an immutable/frozen parameter object or other alternative
    > **Done:** The split loses on its own terms. Its only performance argument — a
    > mutable working copy avoiding allocation — is **not exercised**: `split-state`
    > allocates a complete fresh parameter set (`hmm-param.lsh:153-158`) and rebinds every
    > slot (`210-215`), `merge-states` likewise (`238-243`), because every accepted move
    > changes array *shape* and cannot resize in place. That is construction of a new
    > value minus the freezing, and per §12 it is the compiled/hot path, not a corner.
    > Its actual motivation is a UI affordance being deleted (§11: the `Keep model` /
    > `Reset model` buttons of `hmm-trainer-view.lsh`, which migrates nowhere), and its
    > measured cost is 35 verbatim-duplicated lines including the stationary solve
    > (§5, §13) — the same surface the §5 initial-distribution defect lives in.
  - [x] Check revision 04's topology-search accept/reject ratio expectations before committing, per this session's discussion
    > **Done:** There is no measured ratio, and that is the finding. §11: topology search
    > in the original "was driven by hand" — nothing loops over `suggest-move`, a person
    > watched the description length and pressed buttons — so revision 04's "rejects far
    > more moves than it accepts" (`planned/04-hmm-v0.3.0.md`:62) is a forward expectation
    > about a strategy this project writes for the first time, not a translation.
    > This strengthens the choice rather than qualifying it: with the ratio unknown, the
    > architecture insensitive to it wins. Immutable rollback costs the same at 1-in-5 as
    > at 1-in-500, whereas a save-point's cost *and* its forgotten-restore risk both grow
    > exactly as trials get cheaper and more numerous.
  - [x] Record the decision as an ADR (order of ADR 0010/0015)
    > **Done:** [ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md),
    > Accepted 2026-09-03, plus its three `adr/README.md` edits (index row, reading-order
    > bullet, coverage paragraph).

- [x] Settle what a caller constructs and what Viterbi is a method on
  > **Q:** Should the frozen parameter value retain the `Vocabulary` it was sized against?
  > **A:** Retain it. Two different 25-symbol tables produce identically-shaped,
  > incompatibly-meaning tensors and no range check can tell them apart; holding the
  > vocabulary is the only thing that makes the mismatch detectable. It is consistent with
  > [ADR 0017](../../design/adr/0017-frozen-parameter-object-for-hmm.md) because a
  > `SymbolTable` is frozen and cannot change, and it is a parameter of the model rather
  > than run bookkeeping — the emission axis means nothing without it.
  > **Q:** What does the public Viterbi entry point consume?
  > **A:** Two layers. Public `viterbi(params, record)` takes one `SequenceRecord` and
  > returns a `ViterbiPath`; a private array-level kernel takes the three arrays plus a
  > codes array and is what each ADR 0003 backend implements. The split is what keeps the
  > Cython and Numba phases mechanical — a backend never touches a dataclass.
  - [x] No trainer object exists in 0.1.0 (falsifier-3 finding, `HMMLIB-ACCOUNT.md` §15) — confirm this shapes the construction API
    > **Done:** Confirmed, and §7 makes it stronger than §15 alone did. The decode "reads
    > no forward variable": `update-viterbi-path` binds only the three parameter arrays,
    > the sequence and `state-entropies`, and allocates δ and ψ itself; `alpha*` is a
    > local of `update-data-p` and unreachable. The two are scheduled together by
    > `update-data` "for readability", but "the coupling is scheduling, not data". So
    > Viterbi sitting on `hmm-trainer` is an artefact of where the corpus lived, and it
    > becomes a **free function over parameters and one sequence**. Two riders: the
    > entropy annotation is separable (it needs the stationary solve, the δ/ψ recurrence
    > does not) and is free under 0017's cached properties; and Lush writes its results
    > back into the corpus object (`:data-seq:path-states`, `:data-seq:path-entropy`) —
    > the same coupling as the alphabet read, so **the decode returns a result rather
    > than mutating the dataset**.
  - [x] Settle `SymbolTable` consumption: explicit constructor argument, not a file-coupled read
    > **Done:** Taken explicitly, and typed as the `Vocabulary` **Protocol** rather than
    > the concrete `SymbolTable` — `SequenceDataset.__init__` already types its own
    > parameter that way, so the structural type is the established pattern rather than a
    > new one. Retained on `HMMParams` per the Q&A above. `hmm` needs exactly one thing
    > from it, `size`, to fix the `A` axis of `output_p`; everything crossing the boundary
    > after that is already integers. Lush's file-coupled read of `_alphabet_size` /
    > `_alphabet` out of the `.sds` directory (§4) is not reproduced.
  - [x] Settle whether Viterbi 0.1.0 consumes a single `dataseq` record directly, deferring `pad_collate` to revision 03
    > **Done:** A single `SequenceRecord`, and the deferral is structural rather than a
    > scheduling choice: a record never holds padding (`_record.py` — length is always the
    > true length), so a single-sequence decode has no mask to consult and `pad_collate`
    > has nothing to contribute. Padding exists only in a collated batch, which is
    > revision 03's batched trainer. Noted in passing: `SequenceRecord` already stores its
    > `codes` array read-only, so ADR 0017's `writeable = False` discipline is `dataseq`'s
    > existing practice rather than a new invention.

- [x] Apply *encode at the boundary* (ADR 0001)
  > **Done:** `pfsmgraph.hmm` sits **entirely below** the boundary. The boundary for this
  > package is upstream, in `dataseq` — `SymbolTable.encode` and
  > `SequenceDataset.from_symbols` are where symbols become codes, and `hmm` is only ever
  > handed the result. That is the condition [ADR 0001](../../design/adr/0001-encode-at-the-boundary.md)
  > says makes the compiled and GPU phases mechanical, so ADR 0002 phases 2–4 have no
  > string handling to port.
  - [x] Name the exact entry and exit points where strings are still permitted
    > **Done:** Two of each, and no more.
    >
    > **Entry.** (1) `HMMParams` construction takes a `Vocabulary`. It *contains* strings,
    > but `hmm` calls only `size` on it — to fix the `A` axis of `output_p` — and holds it
    > for identity comparison; no symbol string is read. (2) `SequenceRecord.label`
    > (`str | None`) arrives with the record. It is never encoded and never indexes
    > anything; it is an opaque identifier.
    >
    > **There is deliberately no entry point that accepts symbols as strings.** ADR 0001
    > names that exact temptation as the erosion mechanism — "a helper that accepts a
    > string *just this once* for convenience is how the boundary erodes" — so there is no
    > `viterbi(params, ["D3", "F3"])` overload. A caller holding strings encodes through
    > `dataseq` first.
    >
    > **Exit.** (1) `ViterbiPath.label`, the passthrough of the above. (2) Nothing else.
    > In particular `ViterbiPath.states` is **not** decoded, and that is not an omission:
    > ADR 0001's "results decode back to strings at exit" is about *symbols*, and a state
    > path lives in a different index space with no string representation in this release.
    > The clause has no referent here rather than being waived.
    >
    > **What keeps it true is goal 2's two-layer split**, which turns out to do more work
    > than backend mechanics. ADR 0001 lists "unenforceable by the type system" among its
    > costs; here the kernel signature `_viterbi(init_p, transition_p, output_p, codes)`
    > enforces it, because the label physically cannot reach the kernel — the public
    > wrapper attaches it to the result instead of passing it down.
    >
    > **Consequence for persistence, recorded because it is not obvious:** 0.1.0 ships no
    > model save/load, and could not without firing `DEFERRED.md`'s
    > `## Trigger: a vocabulary outliving the process that built it`. ADR 0001's own cost
    > clause is the reason — the mapping must travel with any persisted artifact, and
    > `SymbolTable` has no serialization yet (the escaping rule is undecided).
    >
    > **Surfaced for the implementation subgoal, not decided here:** whether `A` is
    > `vocab.size` (reserved block included, so the emission tensor carries fibres for
    > `PAD`/`UNK`/… that must never be emitted) or only the user symbols. It changes the
    > `(S, S, A)` shape, so it belongs with the kernel rather than with the boundary.
