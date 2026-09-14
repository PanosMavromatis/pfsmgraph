# feat/hmm-backend-seam

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Write the runtime backend-selection ADR and build the seam it decides — `docs/plan/TODO.md`, revision 03-hmm-v0.2.0

## Goals

- [x] Write ADR 0021 on runtime backend selection
  > **Q:** How should a caller choose the backend: a per-call keyword only, a keyword plus a module-level default (`set_backend` / `use_backend` context), or backend objects?
  > **A:** A per-call keyword only, `viterbi(params, record, *, backend=...)`, and the same keyword on the training entry points when they become public. No module state: the choice is visible at every call site, which follows `encode(on_unknown=...)` and `rand_p_vector`'s required generator, and ADR 0003 tests pass the fixture value straight through.
  > **Q:** What should the public backend names be?
  > **A:** The registry's names as strings, `"python" | "cython" | "cpu_parallel" | "cuda"`, typed as a `Literal` and validated before any work. They are the same vocabulary as the session header and `PFSMGRAPH_REQUIRE_BACKENDS`; `torch` becomes a fifth value.
  - [x] Settle the public shape: a per-call `backend=` keyword, a module-level default, or both; and the backend names
  > **Q:** What should `backend=` default to: `"python"`, `"cython"`, or an `"auto"` value resolving to the best available?
  > **A:** `"python"`. The phase-1 reference is present in every install and every wheel and is ADR 0002's oracle, so acceleration stays opt-in (ADR 0004) and a call never changes behaviour because an extra or a device appeared. `"auto"` was declined because "best" depends on `S`, `N` and the host (phase 3 is 853x slower than phase 2 at `S = 5` on one host and faster at `S = 64` on another), and because it would let a result's backend, and after `torch` its last ulps, depend on the environment.
  > **Q:** What happens when a caller explicitly requests a backend that is unavailable here?
  > **A:** Raise before any work, with a `BackendUnavailableError` (an `ImportError` subclass) naming the reason and the remedy (the extra to install, or "no CUDA device detected"). No fallback and no opt-in fallback keyword: ADR 0011's strictness applies, and "silently ran on the CPU path" is ADR 0003's own example of work that merely looks fine. A name that is not a backend at all is a plain `ValueError`.
  > **Note:** at runtime a missing `cython` extension is a legitimate absence, not the broken working copy ADR 0003 makes a hard test failure: `meson.options`' `compiled=false` already builds a pure wheel, and a platform with no binary wheel installs one. The ADR has to keep the two policies apart. The test suite still escalates an unbuilt extension; the API reports it as unavailable.
  - [x] Settle the default backend and what an unavailable requested backend does (raise vs fall back), against ADR 0011's strictness position
  > **Q:** Where should the backend registry live, so the runtime and the repo-root test header read one source: in `hmm` with the root keeping only the test policy, at the root with a mirror in `hmm`, or in `dataseq` as a family registry?
  > **A:** In `hmm`. A private, shipped `pfsmgraph/hmm/_backends.py` holds the table, keyed by algorithm and then backend name: the kernel module, what an absence needs (nothing, the compiled extension, numba, a CUDA device) and the extra that supplies it. The runtime resolves it lazily. The repo-root `_backends.py` keeps only ADR 0003's policy (header, skip versus escalation, `PFSMGRAPH_REQUIRE_BACKENDS`) and reads `hmm`'s table; `align` will ship its own table, and the root will read both, so no distribution imports another's registry. A mirror was declined as the drift ADR 0003 exists to prevent, and a `dataseq` registry because the base layer has no DP and would need a release for every new row.
  > **Note:** today's registry is one algorithm, a row per Viterbi phase. `forward_backward` is registered with `dp-compile` but absent from `BACKENDS`, so the header cannot show it; keying by algorithm fixes that. `dp-compile.toml`'s `[backends] registry = "_backends.py"` must move with the table, or phase skills will add rows where nothing reads them.
  - [x] Settle where the registry lives, so the runtime and the repo-root test header read one source
  > **Q:** Reviewing the draft, its Open section left public enumeration undecided. Should it be settled, and in what shape: a status per backend, available names only, or keyed by the callable?
  > **A:** Settled, as a status per backend: `pfsmgraph.hmm.backends("viterbi")` returns frozen `BackendStatus(name, available, reason)` rows in phase order. Unavailable backends are included, with the same reason `BackendUnavailableError` carries. The key is the public call's name, never a private kernel's. It probes once per process and caches the result. The repo-root header is built from it plus the private "absence attributable to" field, so the header and the API cannot disagree. Recorded as ADR 0021 §4.
  > **Note:** the draft's second Open item, batching and device placement, stays open. It belongs to the master plan's "Batch the trainer over sequences" subgoal, which now carries a note saying so, including whether `viterbi` gains a batched form there, since that subgoal is worded for the trainer only.
  > **Done:** [ADR 0021](../../design/adr/0021-runtime-backend-selection.md) records all four decisions: a per-call `backend=` string, a default of `"python"`, `BackendUnavailableError` with no fallback, the table in `hmm` with the test policy at the root, and public `backends(name)`. It is indexed in `adr/README.md`, and ADR 0003's Resolved section points at it.
- [ ] Build the seam in `pfsmgraph-hmm`
  - [ ] `viterbi` dispatches over all four phases through it
  - [ ] Export `backends(name)`, `BackendStatus` and `BackendUnavailableError` from `pfsmgraph.hmm`, with one per-process probe cache shared by `backends()` and `backend=` resolution
  - [ ] Register forward-backward and EM kernels in the table at phase 1 only; they stay private, so no public `backends()` key until a training call is public
- [ ] Parameterise the ADR 0003 suites over the seam
  - [ ] Shared Viterbi tests run once per available backend through the public API
  - [ ] Fold `test_viterbi.py`'s labelled non-shared section into them
  - [ ] Rebuild the repo-root `_backends.py` on `backends()` plus the private absence field (no table of its own); header rows per public call; rewrite `tests/test_backends.py`; move `dp-compile.toml`'s `[backends] registry`
  - [ ] Test `backends()` itself: phase order, unavailable rows included, `reason` identical to `BackendUnavailableError`'s message, one probe per process
- [ ] Restore the accelerator extras (`cpu-parallel`, `gpu` with the `numpy<2.5` cap)
- [ ] Update the docs: `docs/api/hmm/`, `core.md`, the ADR index, and the `DEFERRED.md` entries
  - [ ] Document `backends()` and `backend=` in `docs/api/hmm/` with host-independent executed examples (e.g. the `python` row, or an unknown-name `ValueError`), since the ADR 0013 verifier compares pasted output and availability varies by machine
