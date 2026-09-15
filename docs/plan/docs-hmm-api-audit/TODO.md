# docs/hmm-api-audit

**Status**: active
**Created**: 2026-09-15
**Subgoal**: Audit `docs/api/` for gaps before the release (revision 03-hmm-v0.2.0)

## Goals

- [ ] Check the public surface against its pages
  - [ ] Every name in `pfsmgraph.hmm.__all__` is documented on a page under `docs/api/hmm/`
  - [ ] Each public call's signature, parameters and defaults match its docstring and page
  - [ ] Each return field and each raised error is documented, with executed examples where the page shows one
- [ ] Bring `docs/api/hmm/README.md` up to a package that trains
  - [ ] The index lists every page
  - [ ] The contracts section covers training, batching, `backend=` and `device=`, not only the decode and parameters
  - [ ] The example still runs and still represents the package
- [ ] Audit the two top-level surfaces
  - [ ] `docs/api/README.md` mentions `baum_welch` and backends
  - [ ] `packages/pfsmgraph-hmm/README.md` describes what 0.2.0 ships, since it becomes an immutable PyPI long description
- [ ] Audit the `dataseq` pages revision 03 leans on
  - [ ] `pad_collate` and the container pages agree with how the batched trainer and decode use them
  - [ ] `uv run pytest tests/test_api_docs.py` is green
