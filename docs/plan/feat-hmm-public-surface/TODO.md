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

- [ ] Settle what a caller constructs and what Viterbi is a method on
  - [ ] No trainer object exists in 0.1.0 (falsifier-3 finding, `HMMLIB-ACCOUNT.md` §15) — confirm this shapes the construction API
  - [ ] Settle `SymbolTable` consumption: explicit constructor argument, not a file-coupled read
  - [ ] Settle whether Viterbi 0.1.0 consumes a single `dataseq` record directly, deferring `pad_collate` to revision 03

- [ ] Apply *encode at the boundary* (ADR 0001)
  - [ ] Name the exact entry and exit points where strings are still permitted
