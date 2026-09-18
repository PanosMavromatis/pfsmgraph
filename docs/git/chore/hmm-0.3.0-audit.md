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

- **2026-09-18, goal 1 — the module read.** Six stale docstrings found and fixed. One was
  substantive rather than merely stale: `_topology.py` promised that "the public surface for
  topology search arrives with the search loop", which ADR 0025 overturned four days later —
  and no module cited 0025 at all, including the four it governs. All four now do, each
  naming the cost 0025 accepts for it. Also two off-by-one `util.lsh` line numbers in
  `_mdl.py` (`:124`/`:125` for `CGOLD`/`ZEPS`, actually 125/126), and `_search.py`'s Markdown
  ADR link converted to reST, where it had been rendering as literal text under `help()`.
- `safe_divide`'s loose end is closed at thirteen call sites, and the same claim was stale in
  `core.md` too. Two new orphans opened in its place — `_trials._suggest_split` and
  `_suggest_merge`, which only tests call, because `_suggest_move` must rank merges and
  splits in one stable sort and so cannot compose their two rankings. Left alone
  deliberately: resolving them is a code change, not an audit.
- **2026-09-18, goal 2 — the counts.** Four of five claims in `core.md` verified correct
  against the tree: the suite at 3099 (74 / 2973 / 52, derived per directory), the ADR 0003
  header byte-identical to 0.2.0's, fifteen Python modules and two Cython kernels, and
  twenty-five ADRs. `uv run pytest` is green at 3099 in 378 s with **no skips**, since this
  host carries both a CUDA device and torch — so the 229 numba-cuda tests and
  `test_torch_interop.py` all executed, which a GPU-less run would not have done.
- The one wrong count was `tests/test_search.py`, claimed at 37 and holding 48. It grew on
  `feat/hmm-convergence-backend` and again on `feat/hmm-search-log` with neither branch
  updating the figure, and **both totals stayed correct throughout** — the tests were counted,
  merely attributed to the wrong file, so no aggregate check could have seen it. Fixed, with a
  parenthetical recording the failure mode rather than just the number.
- Noted and deliberately not acted on: `docs/agents/codex.md` still describes `hmm` as "three
  modules and 167 tests", under a heading dated 2026-09-04. The date makes it honest rather
  than false, so it is not drift in the sense this audit fixes — but it points a Codex
  reviewer at revision 02's surface, with Baum-Welch, the MDL criterion and the whole search
  invisible. Refreshing it is a `codex.md` job, outside this branch's `docs/api/` scope.
