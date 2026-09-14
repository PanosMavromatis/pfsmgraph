# 0021. A backend is chosen per call, defaults to `python`, and is never substituted

- **Status:** Accepted
- **Date:** 2026-09-14
- **Source:** none in the PRD — postdates it. Settles the question
  [ADR 0003](0003-one-parameterized-test-suite-per-algorithm.md)'s Open section defers,
  under `docs/plan/DEFERRED.md`'s `## Trigger: a backend-selection API` (titled `align acquiring a
  backend-selection API` until this record fired it).

## Context

[ADR 0002](0002-three-phase-algorithm-lifecycle.md), as amended by
[ADR 0016](0016-numba-cpu-parallel-phase.md), gives every dynamic-programming kernel four
backends: pure Python/numpy, Cython, Numba CPU-parallel and Numba CUDA. `pfsmgraph-hmm`'s
Viterbi decode has all four since 2026-09-13, and all four are bit-exact with one another on
a given host. None of them is reachable except the first: `viterbi(params, record)` has no
way to name a backend, so 0.1.0 shipped three kernels no caller can run and withheld the
`cpu-parallel` and `gpu` extras that would install their dependencies for nothing.

The same gap blocks the test suite. ADR 0003 requires one suite per algorithm with the
backend as a fixture parameter, *and* tests written against the public API only. With no
backend in the public API the two cannot both hold, so today a green session header —
`backends: python ✓ · cython ✓ · cpu_parallel ✓ · cuda ✓` — means each kernel imports,
not that the suite ran against it. Equivalence is checked only in `test_viterbi.py`'s
labelled non-shared section.

The trigger that filed this record names `align`, because it was written expecting `align`
to reach a compiled phase first. `hmm` reached all four phases first, and revision 03 is
about to add training kernels and an optional `torch` backend on top of them, so the
decision is taken here.

Four questions are open:

1. **How a caller names a backend** — per call, through module-level state, or through
   backend objects.
2. **What is used when the caller names none, and what happens when the named one cannot
   run here** — raise, or fall back.
3. **Where the table of backends lives**, given that the runtime needs it and the
   repository's test policy already has one, at the repo root, that no shipped artifact
   contains.
4. **Whether a caller can see which backends run here** without asking for each in turn.

Two facts constrain every answer.

- **The fastest backend is not a property of the environment.** Measured at `N = 400`:
  phase 3 is 853x slower than phase 2 at `S = 5` on the host where it landed, and faster than
  phase 2 at `S = 64` on a 4-vCPU Xeon; phase 4 on an NVIDIA L4 is 205x slower than Cython
  at `S = 5` and 9x faster at `S = 160`. "Best available" depends on the model's size and on
  the machine.
- **A backend can be legitimately absent from an install for reasons the test suite would
  call a broken working copy.** `meson.options`' `compiled=false` builds a pure wheel, as
  0.1.0 was, and a platform with no binary wheel installs one. So a consumer can lack the
  Cython extension with nothing wrong, while in this repository an unbuilt extension is the
  stale-build failure ADR 0003 makes a hard error.

## Decision

### 1. A backend is a per-call keyword, named by string

Every public call that runs a DP kernel takes a keyword-only `backend`:

```python
viterbi(params, record)                      # the reference
viterbi(params, record, backend="cython")
```

Names are strings, typed as a `Literal`: `"python"`, `"cython"`, `"cpu_parallel"`,
`"cuda"`. They are the names the session header and `PFSMGRAPH_REQUIRE_BACKENDS` already
use, so one vocabulary covers the API, the header and CI. A new backend is a new value —
`"torch"` will be the fifth. *(Amended 2026-09-14 on `feat/hmm-torch-backend`: it is, as a
backend of `baum_welch` only. It is not a lifecycle phase, so `BACKEND_NAMES` lists it after
the four phases. Its results agree with the reference within
[ADR 0020](0020-scaled-probability-domain-forward-backward.md)'s tolerance rather than
exactly, and `BackendStatus` does not carry that tolerance;
`docs/api/hmm/baum_welch.md` states it.)*

There is **no module-level default, setter or context manager.** The backend a result came
from is visible at the call that produced it. That follows two positions this family has
already taken: `encode(..., on_unknown=...)` states its policy per call rather than per
table, and `rand_p_vector` takes a required `Generator` so reproducibility needs no hidden
state ([ADR 0017](0017-frozen-parameter-object-for-hmm.md) makes parameters a re-derivable
value, which a process-global backend switch would quietly undercut once a backend is
allowed a tolerance). A caller who wants one choice everywhere can bind it with
`functools.partial`.

The keyword is validated **before any work**, in this order:

- A name that is no backend at all — `"cuda12"`, `"numba"` — raises `ValueError` listing the
  names.
- A known name whose lifecycle phase this algorithm has not reached — `"cython"` for
  forward-backward while only phase 1 exists — also raises `ValueError`, naming the backends
  that algorithm has. It fails the same way on every machine, so it is a mistake in the
  call, not a fact about the environment; this is ADR 0003's "contributes no parameter"
  category, seen from the API.
- A backend that exists but cannot run here raises `BackendUnavailableError` (below).

### 2. The default is `"python"`, and an unavailable backend raises

**The default is `"python"`**, the phase-1 reference. It is present in every install and
every wheel, and it is ADR 0002's oracle. Acceleration stays opt-in, as
[ADR 0004](0004-gpu-backends-and-optional-dependency-strategy.md) requires, and a call never
changes behaviour because an extra was installed or a device appeared.

**An explicitly requested backend that cannot run here raises
`BackendUnavailableError`**, a subclass of `ImportError`, before any work. Its message names
what is missing and what supplies it: the `cpu-parallel` or `gpu` extra, a platform wheel
or a source build with a compiler, or a CUDA device. **Nothing falls back**, with or without
a warning, and there is no keyword to opt into falling back.

This is [ADR 0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md)'s strictness
applied to a second question, and ADR 0003's Open section had already named the failure:
"silently ran on the CPU path" is work that merely looks fine. A caller who wants a fallback
writes one, with `except BackendUnavailableError`, and has then chosen it.

Subclassing `ImportError` says what the error is — a backend's dependency could not be
imported or its device found — and lets code that already guards an optional import catch
it. It is a distinct type for the same reason `ImpossibleSequenceError` is: the case is an
ordinary outcome for a program that probes backends, not a malformed call.

**Resolution is lazy.** `import pfsmgraph.hmm` imports no backend module. A kernel module
is imported on the first call that names it, so an install without numba never touches
numba, and a program that only decodes on `"python"` never pays for a CUDA probe. The one
call that probes backends it was not asked to run is the enumeration in §4, and probing is
what it is for.

### 3. The table lives in `hmm`; the test policy stays at the root

**The backend table is a private module of the distribution that owns the kernels**:
`pfsmgraph/hmm/_backends.py`, shipped in the wheel. It is keyed by algorithm, then by
backend name, and each row records three things: the kernel module; what an absence of
that backend is attributable to — nothing, the compiled extension, numba, or a CUDA
device; and the extra that supplies it, if any. The public calls resolve through it, and
`BackendUnavailableError`'s remedy text is derived from it.

**The repo-root `_backends.py` keeps ADR 0003's policy and nothing else**: the session
header, the skip-or-escalate rule and `PFSMGRAPH_REQUIRE_BACKENDS`. It holds no table. It
takes availability from the private `_status(algorithm)` behind the public enumeration in
§4, which accepts private algorithms too, and reads the private "absence attributable to"
field to decide skip versus escalation, so the header and a user's `backends()` call run one
probe and cannot disagree. *(Amended 2026-09-14 while building it: the first wording named
`backends()` itself, which refuses the private `forward_backward` the header reports.)* A future
`pfsmgraph.align` is read beside it; no distribution imports another's table.

The row's "absence attributable to" field is what lets one table serve two policies:

| Absence attributable to | Public API (an install) | Test suite (this checkout) |
|---|---|---|
| nothing (`python`) | cannot occur | hard failure — broken working copy |
| the compiled extension (`cython`) | `BackendUnavailableError` | hard failure — missing or stale build |
| numba (`cpu_parallel`) | `BackendUnavailableError` | reported skip |
| a CUDA device (`cuda`) | `BackendUnavailableError` | reported skip; `PFSMGRAPH_REQUIRE_BACKENDS` escalates |
| torch (`torch`) | `BackendUnavailableError` | reported skip; `PFSMGRAPH_REQUIRE_BACKENDS` escalates |

The second row is the one that differs, and it differs correctly: an install may be a pure
wheel, a checkout may not.

### 4. Enumeration is public, and reports why a backend is missing

```python
>>> from pfsmgraph import hmm
>>> hmm.backends("viterbi")
(BackendStatus(name='python', available=True, reason=None),
 BackendStatus(name='cython', available=True, reason=None),
 BackendStatus(name='cpu_parallel', available=False,
     reason="numba is not installed: pip install 'pfsmgraph-hmm[cpu-parallel]'"),
 BackendStatus(name='cuda', available=False, reason='no CUDA device detected'))
```

`backends(name)` returns a tuple of frozen `BackendStatus(name, available, reason)`, one per
backend that call has, **in lifecycle-phase order**. Unavailable backends are included, not
filtered out, and their `reason` is the same text `BackendUnavailableError` carries, so
asking in advance and asking by calling give one answer. A backend whose phase has not been
reached is absent, as it is from the test matrix.

- **The key is the public call's name**, `"viterbi"` — not the kernel's. Kernel modules are
  private and revision 04 is free to rename or split them. A training entry point is keyed by
  its own name when it becomes public, whatever kernels it runs, and an unknown name raises
  `ValueError` listing the known ones. *(Amended 2026-09-14: `baum_welch`, whose rows are
  per-record E-steps rather than forward-backward kernels, since the torch E-step builds no β
  and so cannot share `_forward_backward`'s signature.)*
- **It probes, once per process.** Checking `cuda` imports numba-cuda and asks for a device,
  which is the cost the lazy rule in §2 avoids on ordinary calls. The result is cached, so a
  later `backend=` call that names an unavailable backend raises from the cache without
  probing again. A device that appears mid-process is not noticed; that trade favours one
  consistent answer per process over a live one.
- **Availability is not recommendation.** The tuple is ordered by phase, not by speed, and
  no field ranks backends, for the reasons the `"auto"` alternative below is rejected.

It exists because the refusal to fall back makes it necessary: a caller who must choose
explicitly should be able to see the choices, with reasons, without writing a
`try`/`except` per backend.

## Consequences

### Positive

- **ADR 0003's two requirements become jointly satisfiable.** A shared test takes a
  `backend` fixture and passes it to the public call, so every case runs against every
  available backend, and `test_viterbi.py`'s non-shared equivalence section can fold into
  the shared suite. The session header then means what it appears to.
- **The withheld extras have something to promise.** `cpu-parallel` and `gpu` return in
  the version that ships this seam, and the pure-wheel restriction ends with it.
- **The header grows the forward-backward row it was missing.** `forward_backward` has
  been registered with `dp-compile` since 2026-09-14 but was absent from the test-only
  table, which was keyed by phase for Viterbi alone.
- **The header and the API cannot disagree about availability**, because the header is
  built from `backends()`. A user and the test run see the same reason for the same absence.
- **Results are reproducible from the call site.** With the default fixed and no
  substitution, the same call on two machines runs the same backend or fails; it never runs
  a different one.

### Negative / costs

- **The fast path is opt-in, and a caller who never reads the docs never finds it.** Users
  who install `pfsmgraph-hmm[gpu]` still get the reference decode until they write
  `backend="cuda"`. That is the price of never changing behaviour with the environment.
- **Portable code that wants "use it if you have it" chooses for itself**, from `backends()`
  or with a `try`/`except`. No keyword does it for them.
- **`BackendStatus` and its `reason` text are public.** The reason is prose meant for a person,
  and code that matches on it will break when a remedy is reworded; `available` is the field
  to branch on. The call names accepted by `backends()` are public too.
- **The table is now shipped surface**, private but in the wheel. Renaming a row after a
  release is a change a consumer's `backend=` string can observe; the names are public even
  though the table is not.
- **`dp-compile.toml`'s `[backends] registry` must move to the new table.** Otherwise a phase
  skill adds rows to a file that no longer holds any.
- **Every public DP entry point carries the keyword**, including ones with only phase 1, so
  a call with `backend="cython"` fails with `ValueError` until that phase lands. The failure
  is uniform and explained, which is the point, but it is a keyword most calls cannot yet use.

## Alternatives considered

- **A module-level default (`set_backend`, or a `use_backend` context manager) beside the
  keyword.** Rejected: convenient in a notebook, but the backend a call used is no longer
  visible at the call, and a context variable set in one cell silently governs another.
- **Backend objects** (`backend=backends.cython`, or a `Decoder(backend)` holding the
  choice). Rejected: more public types for no information a string lacks, and the rows
  would become API that is costlier to change.
- **A default of `"auto"`, resolving to the best available.** Rejected on the measurements
  in Context: "best" depends on `S`, `N` and the host, so any rule is a guess that differs
  by machine, and once `torch` is a backend a result's last few ulps would depend on what
  happened to be installed.
- **A default of `"cython"`.** Rejected: absent from a pure wheel, so the default call would
  fail, or need the fallback this record refuses.
- **Warn and fall back to the next phase down.** Rejected: a warning is shown once per
  location and routinely filtered, so a batch job can run for hours on the wrong backend.
  ADR 0003 names this failure.
- **Raise by default, with an `on_unavailable="fallback"` keyword.** Rejected: more surface
  for a case `except BackendUnavailableError` already expresses, and unlike
  `on_unknown="unk"`, which substitutes a documented symbol, a fallback substitutes a
  different computation.
- **Keep the table at the repo root and mirror it in `hmm`.** Rejected: two lists of the
  same backends that can disagree, which is the drift ADR 0003 exists to prevent, one level
  up.
- **A family-wide table in `dataseq`.** Rejected: the base layer has no dynamic programming
  ([ADR 0009](0009-dataseq-as-the-base-layer.md)), and every new backend row in `hmm` or
  `align` would wait on a `dataseq` release.
- **No public enumeration**, leaving `BackendUnavailableError` as the only way to learn what
  runs here. Rejected: with no fallback, a caller has to choose, and discovering the choices
  one exception at a time is the cost that refusal would otherwise impose.
- **Enumeration of available names only** (`available_backends("viterbi") ->
  ("python", "cython")`). Rejected: no new type, but *why* a backend is missing — the part a
  user acts on — would again be discoverable only by calling it.
- **Enumeration keyed by the callable** (`backends(hmm.viterbi)`). Rejected: no string to
  mistype, but it ties the key to a function object, and a training entry point composed of
  several kernels still needs one name for its backend set.

## Open

- **Batching and device placement.** Phase 4's speed case is a batch, and neither a batch
  call nor a device index exists yet. Scheduled in `docs/plan/TODO.md`, revision
  03-hmm-v0.2.0's "Batch the trainer over sequences" subgoal, which decides what a batch call
  takes beyond `backend` and whether `viterbi` gains a batched form there too; the phases 2-4
  subgoal for the forward recurrence consumes the answer. This record fixes the keyword, not
  what a batch adds to it.
