# docs/hmm-api

**Status**: active
**Created**: 2026-09-13
**Subgoal**: Write the `/docs/api/` documents that pertain to this release —
`docs/plan/TODO.md`, revision 02-hmm-v0.1.0

## Goals

- [ ] Settle what `docs/api/hmm/` contains before writing any of it
  - [ ] Decide the page split (by analogy with `dataseq/`'s `README.md` / `container.md` / `encoder.md`) and which contract each page owns
  - [ ] Decide how the unselectable backends, the `gpu` / `cpu-parallel` extras and the `numpy<2.5` cap are presented to a consumer
- [ ] Write the pages under ADR 0013's rule
  - [ ] Every code block executed, its output pasted from the run, tracebacks included
  - [ ] `tests/test_api_docs.py` green over the new pages
- [ ] Update the index and reconcile with the source
  - [ ] `docs/api/README.md`'s table lists `pfsmgraph-hmm` as documented
  - [ ] Each page checked against the docstrings of the names it documents; disagreements fixed on whichever side is wrong
