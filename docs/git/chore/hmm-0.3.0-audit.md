# chore/hmm-0.3.0-audit

**Created**: 2026-09-18
**Base**: main at b105bc9
**Status**: active

## Purpose

Revision 04's penultimate subgoal: read the whole `hmm` codebase and audit
`docs/api/` against what 0.3.0 actually ships. Revision 03's audit (PR #39)
covered the documentation alone; this one reads the code as well, because
revision 04 is the first revision to add modules that no differential oracle
checks — `_topology.py`'s split departs from the Lush original on purpose
(ADR 0022) and `_search.py`'s loop departs from its own original on purpose
(ADR 0023). Where revision 02 and 03 could ask "does it match?", here the only
available question is "does it say what it does?".

It runs before the release because both halves check things an immutable
version freezes: `__all__`, and `packages/pfsmgraph-hmm/README.md`, which
becomes the PyPI long description.

## Scope

- Read every module under `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/` as a
  whole, rather than as the sum of the branches that wrote them.
- Private helpers left without a consumer — revision 04 was expected to remove
  `safe_divide` from that list; check that it did.
- Docstrings still describing a subgoal's intermediate state, such as
  `_topology.py`'s "state merge next".
- Every ADR, `HMMLIB-ACCOUNT.md` and Lush line reference still pointing at what
  it names, and no module presenting new work as a port.
- The draw-order and `Generator` contracts end to end, from the public search
  call down to `_split_state`.
- Counts verified rather than assumed: test count, module count and ADR count
  in `core.md`, and the ADR 0003 backend header unchanged from 0.2.0.
- `docs/api/hmm/` against the ten names in `__all__` — every parameter, return
  field and raised error — plus the `hmm/README.md` index, `docs/api/README.md`
  and `packages/pfsmgraph-hmm/README.md`.

## Context

- Master plan subgoal: `docs/plan/TODO.md`, revision 04, line 228.
- [ADR 0025](../../design/adr/0025-topology-search-not-exported.md) settles the
  surface this audit checks: `__all__` stays at ten names at 0.3.0, so the docs
  half is an audit against the *same* ten 0.2.0 shipped, and revision 04's four
  new modules — `_search`, `_trials`, `_topology`, `_mdl` — are audited as
  private rather than as documented surface.
- [ADR 0013](../../design/adr/0013-api-documentation-layout-and-tooling.md)
  still governs: every code block is executed with its output pasted, and
  `tests/test_api_docs.py` stays the guard.
- Revision 03's audit: PR #39, `docs/hmm-api-audit`.
- ADR 0025 §7 obliges 0.3.0's release notes to state that the headline
  capability has no supported entry point; the root `README.md` says topology
  search "is a later revision", true today and false once 0.3.0 ships. Both
  belong to the release subgoal, not this one — noted here so they are not
  rediscovered.

## Notes
