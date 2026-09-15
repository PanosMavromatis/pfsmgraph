# docs/hmm-api-audit

**Created**: 2026-09-15
**Base**: main at 356cc50
**Status**: active

## Purpose

Read `docs/api/` as a whole against what `pfsmgraph-hmm` 0.2.0 will ship, before that
release. Each revision-03 subgoal documented what it touched, so every page is accurate
where it was edited but none has been read against the finished surface. The package now
trains as well as decodes, batches both, and selects among five training backends, and
the index pages still describe a decoder. The most consequential surface is
`packages/pfsmgraph-hmm/README.md`, which becomes the PyPI long description under an
immutable version and so cannot be corrected after publishing.

## Scope

- Every name in `pfsmgraph.hmm.__all__`, and each public call's parameters, return
  fields and raised errors, checked against its page under `docs/api/hmm/`.
- `docs/api/hmm/README.md`: index, contracts and example. Its contracts section speaks
  only of the decode and parameters, with nothing on training, batching or `device=`.
- `docs/api/README.md`, which never mentions `baum_welch` or backends, and
  `packages/pfsmgraph-hmm/README.md`.
- The `dataseq` pages revision 03 leans on, such as `pad_collate` feeding the batched
  trainer.
- Out of scope: the release itself, which is the next master-plan subgoal.

## Context

- Master-plan subgoal: `docs/plan/TODO.md`, revision `03-hmm-v0.2.0`, the `docs/api/`
  audit, with a note placing it after every feature subgoal so it audits what ships.
- [ADR 0013](../../design/adr/0013-api-documentation-layout-and-tooling.md) governs:
  hand-written Markdown, docstrings normative for signatures and `docs/api/` for
  contracts, every code block executed with its output pasted.
- `tests/test_api_docs.py` executes those blocks and reads `packages/*/README.md` too.
- Backend examples must print the same on every machine, since which backends run
  differs by host; `backends.md` and `baum_welch.md` already avoid executing `cuda`.

## Notes

