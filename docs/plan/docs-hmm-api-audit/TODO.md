# docs/hmm-api-audit

**Status**: active
**Created**: 2026-09-15
**Subgoal**: Audit `docs/api/` for gaps before the release (revision 03-hmm-v0.2.0)

## Goals

- [~] Check the public surface against its pages
  > **Q:** `BackendName` appears in the published signatures of `viterbi`, `viterbi_batch` and `baum_welch` but cannot be imported from `pfsmgraph.hmm`. Export it, define it on the page, or leave it?
  > **A:** Define it on the page: a paragraph in `backends.md` saying it is a `Literal` of the five names, for annotations only, and that a caller passes a plain string. No API change.
  > **Commit:** deferred after subgoal 1 (plan-only); keep going.
  - [x] Every name in `pfsmgraph.hmm.__all__` is documented on a page under `docs/api/hmm/`
    > **Note:** all ten have a section: `HMMParams` in `params.md`; `viterbi`, `viterbi_batch`, `ViterbiPath` and `ImpossibleSequenceError` in `viterbi.md`; `backends`, `BackendStatus` and `BackendUnavailableError` in `backends.md`; `baum_welch` and `BaumWelchResult` in `baum_welch.md`.
  - [x] Each public call's signature, parameters and defaults match its docstring and page
    > **Done:** `backends.md` now defines `BackendName` beside `backend=`, including that the `Literal` accepts `"torch"` for a call that has no such backend. `baum_welch`'s `:param backend:` names all five. In `baum_welch.md`, "validation happens before any cycle runs" was not true of the impossible-record check, which fires after the first E-step and before any re-estimation; it now says so, and it lists the stopping-rule and `max_cycles` errors separately. `viterbi.md`'s `ImpossibleSequenceError` section named only `viterbi` as the raiser, and now names all three. `tests/test_api_docs.py` passes, 11 of 11.
    > **Note:** every signature block on the pages matches `inspect.signature` exactly, dataclass fields and defaults included. They all spell `backend: BackendName`, though, and `BackendName` is a `Literal` in `_backends.__all__` that `pfsmgraph.hmm` does not export, so a reader can neither import nor look it up.
    > **Note:** `baum_welch`'s docstring is stale where its page is not: `:param backend:` names only `"python"` and `"torch"`, but `"cython"`, `"cpu_parallel"` and `"cuda"` have trained since `feat/hmm-forward-phases`. Docstrings are normative for signatures under ADR 0013, so the docstring is the defect here.
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
