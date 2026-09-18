# chore/hmm-0.3.0-audit

**Status**: active
**Created**: 2026-09-18
**Subgoal**: Review the whole `hmm` codebase and audit `docs/api/` against what 0.3.0 ships (revision 04)

## Goals

- [ ] Read every `hmm` module as a whole, and record what the read finds
  - [ ] Private helpers left without a consumer — confirm revision 04 removed `safe_divide` from that list
  - [ ] Docstrings still describing a subgoal's intermediate state, such as `_topology.py`'s "state merge next"
  - [ ] Every ADR, `HMMLIB-ACCOUNT.md` and Lush line reference points at what it names; no module presents new work as a port
  - [ ] The draw-order and `Generator` contracts end to end, from the public search call down to `_split_state`
- [ ] Verify the counts and the backend header against the tree rather than against the prose
  - [ ] `uv run pytest` — the suite count, and the ADR 0003 header unchanged from 0.2.0
  - [ ] Module count and ADR count in `core.md`
- [ ] Audit `docs/api/` against the ten names 0.3.0 ships
  - [ ] Every `__all__` name against its page: parameters, return fields, raised errors
  - [ ] `docs/api/hmm/README.md` — index, contracts and example, against a package that now searches topology as well as trains
  - [ ] `docs/api/README.md`
  - [ ] `packages/pfsmgraph-hmm/README.md`, which becomes the PyPI long description under an immutable version
  - [ ] Every code block executed with its output pasted; `tests/test_api_docs.py` green
