# docs/hmm-api-audit

**Status**: active
**Created**: 2026-09-15
**Subgoal**: Audit `docs/api/` for gaps before the release (revision 03-hmm-v0.2.0)

## Goals

- [x] Check the public surface against its pages
  > **Q:** `BackendName` appears in the published signatures of `viterbi`, `viterbi_batch` and `baum_welch` but cannot be imported from `pfsmgraph.hmm`. Export it, define it on the page, or leave it?
  > **A:** Define it on the page: a paragraph in `backends.md` saying it is a `Literal` of the five names, for annotations only, and that a caller passes a plain string. No API change.
  > **Commit:** deferred after subgoal 1 (plan-only); keep going.
  - [x] Every name in `pfsmgraph.hmm.__all__` is documented on a page under `docs/api/hmm/`
    > **Note:** all ten have a section: `HMMParams` in `params.md`; `viterbi`, `viterbi_batch`, `ViterbiPath` and `ImpossibleSequenceError` in `viterbi.md`; `backends`, `BackendStatus` and `BackendUnavailableError` in `backends.md`; `baum_welch` and `BaumWelchResult` in `baum_welch.md`.
  - [x] Each public call's signature, parameters and defaults match its docstring and page
    > **Done:** `backends.md` now defines `BackendName` beside `backend=`, including that the `Literal` accepts `"torch"` for a call that has no such backend. `baum_welch`'s `:param backend:` names all five. In `baum_welch.md`, "validation happens before any cycle runs" was not true of the impossible-record check, which fires after the first E-step and before any re-estimation; it now says so, and it lists the stopping-rule and `max_cycles` errors separately. `viterbi.md`'s `ImpossibleSequenceError` section named only `viterbi` as the raiser, and now names all three. `tests/test_api_docs.py` passes, 11 of 11.
    > **Note:** every signature block on the pages matches `inspect.signature` exactly, dataclass fields and defaults included. They all spell `backend: BackendName`, though, and `BackendName` is a `Literal` in `_backends.__all__` that `pfsmgraph.hmm` does not export, so a reader can neither import nor look it up.
    > **Note:** `baum_welch`'s docstring is stale where its page is not: `:param backend:` names only `"python"` and `"torch"`, but `"cython"`, `"cpu_parallel"` and `"cuda"` have trained since `feat/hmm-forward-phases`. Docstrings are normative for signatures under ADR 0013, so the docstring is the defect here.
  - [x] Each return field and each raised error is documented, with executed examples where the page shows one
    > **Q:** `ViterbiPath` has a field table, but `BaumWelchResult`'s fields appear only in examples and `BackendStatus` gets one sentence. Add tables for both?
    > **A:** Write both.
    > **Done:** both dataclasses now have field tables in `ViterbiPath`'s layout. `degenerate_states` is documented as ascending, since `_re_estimate` builds it with `np.flatnonzero`. Every `raise` in `_viterbi.py`, `_baum_welch.py`, `_backends.py` and `_baum_welch_torch.py` that a public call can reach is on a page; subgoal 2's fixes supplied the two that were missing, `max_cycles` and the stopping rule.
- [x] Bring `docs/api/hmm/README.md` up to a package that trains
  > **Note:** this goal ran unattended at the user's request. Its questions were answered with the recommendation and are logged here so they can be revisited.
  > **Q:** The introduction said training was "on `main` for 0.2.0". Describe 0.2.0's contents plainly, although it is not released yet, or keep the hedge?
  > **A (recommendation, adopted):** plainly. The release is the next master-plan subgoal, and this page should be right on release day. "On `main`" would go stale at exactly that moment, while "0.2.0 adds" is only a few days early.
  > **Q:** Replace the decode-only example with a training one, or extend it?
  > **A (recommendation, adopted):** extend it. The decode example carries the three differences from a textbook HMM, which no training example shows as directly. A `### Training it` block on the same model trains on `baum_welch.md`'s corpus, then decodes it with `viterbi_batch`, so all three calls appear.
  - [x] The index lists every page
    > **Note:** all four pages were already listed. The Decode entry was missing `viterbi_batch` and now names it. The Backends section below was a worse defect than the index: it still said `baum_welch` "has two" backends, and now says five, and that the compiled four agree to the bit.
  - [x] The contracts section covers training, batching, `backend=` and `device=`, not only the decode and parameters
    > **Done:** contracts 5 and 7 now name `viterbi_batch` and `baum_welch`. New contracts cover what the package now does: 8 training keeps the topology, 9 counts are kept per record, 10 batching bounds memory and changes no result, 11 a backend is chosen and never inferred, with the bit-exact and `torch` tolerance promises, and 12 `device=`. Related records gained `forward_backward/FORMALIZATION.md`.
    > **Note:** contract 10's training half claims more than one test shows. `test_training_is_bit_identical_at_every_batch_size` runs only the reference. The compiled phases are covered by `test_every_kernel_matches_the_reference_row_by_row_on_a_ragged_batch`, which gives equal per-record counts, and those counts are summed in record order. So the claim for the phases rests on two tests combined, not on a direct one.
  - [x] The example still runs and still represents the package
    > **Done:** the decode example is unchanged and still passes. The new training block trains in 50 cycles, from 13.4368 bits to 10.4047, and decodes with the trained model. It asserts that the starting model is untouched. `tests/test_api_docs.py` passes, 11 of 11.
- [x] Audit the two top-level surfaces
  > **Note:** worked unattended at the user's request, like the previous goal. Questions were answered with the recommendation.
  > **Q:** The package README opened "Version 0.1.0 carries …". Name 0.2.0 in its place?
  > **A (recommendation, adopted):** name no version in the prose. The wheel's metadata already states the version, and a number in immutable text is one more thing to keep in step at every release. The introduction lists what the package contains and calls topology search forthcoming, which stays true of the version it ships with.
  > **Q:** Fix `pyproject.toml`'s `description` as well, though the plan names only the README?
  > **A (recommendation, adopted):** yes. It is the one-line summary PyPI shows above the README, it is just as immutable once published, and it said "Baum-Welch training … forthcoming".
  - [x] `docs/api/README.md` mentions `baum_welch` and backends
    > **Done:** the distribution table's `Status` column became `Covers`, naming each documented member's surfaces, including `baum_welch`, `backend=` and `backends`. Two paragraphs were added. One names `tests/test_api_docs.py` as the mechanism behind the executed-examples rule, which the page had stated as a habit. The other says why backend examples are chosen to print the same output on every machine, and that the bit-exact promise is per machine.
  - [x] `packages/pfsmgraph-hmm/README.md` describes what 0.2.0 ships, since it becomes an immutable PyPI long description
    > **Done:** the introduction lists parameters, single and batched decode, training and backends. A new `## Training` section runs the same training-then-`viterbi_batch` example as the hmm index and states that `batch_size` does not change a result. `## Backends` now covers every call, `torch` included; its executed block lists `backends("baum_welch")`; the extras list adds `torch` and marks `gpu` as not on macOS. `tests/test_api_docs.py` passes, 11 of 11, and it reads this README too.
    > **Note:** the README says nothing about what kind of wheel ships. 0.1.0 was published as a pure `py3-none-any` wheel (`docs/ops/release.md`). If 0.2.0 is pure again, `backend="cython"` raises `BackendUnavailableError` for every pip user, and `backends.md`'s "a platform wheel, or a source build" becomes the remedy nobody can take from PyPI. The release subgoal owns that decision. Whatever it decides, `backends.md` should be checked against it, since that page can still be corrected after publishing and the README cannot.
- [ ] Audit the `dataseq` pages revision 03 leans on
  - [ ] `pad_collate` and the container pages agree with how the batched trainer and decode use them
  - [ ] `uv run pytest tests/test_api_docs.py` is green
