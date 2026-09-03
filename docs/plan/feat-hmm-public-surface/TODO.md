# feat/hmm-public-surface

**Status**: active
**Created**: 2026-09-03
**Subgoal**: Settle the public surface of `pfsmgraph.hmm` 0.1.0 and where it meets `dataseq` (revision `02-hmm-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

## Goals

- [ ] Decide the class architecture
  - [ ] Weigh Lush's `hmm`/`hmm-param`/`hmm-trainer` split (GUI-undo motivation, measured duplication cost per `HMMLIB-ACCOUNT.md` §5) against an immutable/frozen parameter object or other alternative
  - [ ] Check revision 04's topology-search accept/reject ratio expectations before committing, per this session's discussion
  - [ ] Record the decision as an ADR (order of ADR 0010/0015)

- [ ] Settle what a caller constructs and what Viterbi is a method on
  - [ ] No trainer object exists in 0.1.0 (falsifier-3 finding, `HMMLIB-ACCOUNT.md` §15) — confirm this shapes the construction API
  - [ ] Settle `SymbolTable` consumption: explicit constructor argument, not a file-coupled read
  - [ ] Settle whether Viterbi 0.1.0 consumes a single `dataseq` record directly, deferring `pad_collate` to revision 03

- [ ] Apply *encode at the boundary* (ADR 0001)
  - [ ] Name the exact entry and exit points where strings are still permitted
