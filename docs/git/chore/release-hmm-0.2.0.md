# chore/release-hmm-0.2.0

**Created**: 2026-09-15
**Base**: main at 9850b43
**Status**: active

## Purpose

Release `pfsmgraph-hmm` 0.2.0: Baum-Welch training, batched Viterbi decoding, and runtime
backend selection over four lifecycle phases plus torch. It is the first version whose
public API reaches a compiled kernel, so it cannot ship 0.1.0's pure `py3-none-any` wheel
without every pip user seeing `cython ✗`. The branch settles the wheel shape, adapts the
release tooling to it, verifies what a consumer installs, and publishes.

## Scope

- Decide the wheel shape and where it is built (cibuildwheel in CI, locally, or pure again)
- Write the repository's first `.github/workflows/`: cibuildwheel across the matrix, a
  tag-gated publish job, and a plain test job
- Drop `-Dcompiled=false`, restore `Programming Language :: Cython`, teach `preflight` platform tags
- Review 0.2.0's immutable metadata: version, extras bounds, `dataseq` lower bound, README
- Verify each built wheel in a clean venv, including the compiled kernels' bit-exactness
- Publish and tag `pfsmgraph-hmm-v0.2.0` (user-run steps marked **User**)

## Context

- Master plan `docs/plan/TODO.md`, revision 03-hmm-v0.2.0, the release subgoal and its two notes
- Precedent: `docs/plan/02-hmm-v0.1.0/chore-release-hmm-0.1.0/TODO.md` (PR #25)
- `docs/plan/DEFERRED.md`: "a backend-selection API" (the pure-wheel ending) and "CI existing"
  (trusted publishers, `PFSMGRAPH_REQUIRE_BACKENDS`)
- `docs/ops/release.md`, the root `justfile`, `packages/pfsmgraph-hmm/meson.build` / `meson.options`
- ADR 0018 (meson-python), ADR 0020 (no fused multiply-add), ADR 0021 (backend selection)

## Notes

**2026-09-16 — the wheel shape is settled, and the deciding fact was the host rather than a
preference.** 0.2.0 ships twenty platform wheels plus the sdist: linux x86_64 and aarch64,
macOS arm64, Windows x86_64, on cp310–cp314, built by a GitHub Actions cibuildwheel workflow
and published by a tagged run through PyPI Trusted Publishing.

The plan opened by proposing a local cibuildwheel measurement, and that is not runnable
here. `cibuildwheel --platform linux` builds inside a manylinux container and this host has
no container engine — no `docker`, no `podman`. The container-free fallback is narrower than
it first appears: `auditwheel repair` needs no Docker, but it can only claim the host's own
floor, and this is Ubuntu 24.04 on glibc 2.39, so a locally repaired wheel would be tagged
`manylinux_2_39_x86_64` — accepted by PyPI and installed by almost nothing, against a common
floor of `manylinux_2_28`. And it is a Linux x86_64 box, so macOS and Windows wheels are
unbuildable here whether or not a container engine appears. The matrix was therefore never
independent of where the build runs, and the third subgoal was taken first.

**The "CI existing" trigger fires here, and two of its three items land.**
`PFSMGRAPH_REQUIRE_BACKENDS` is wired in, though not in the shape its own entry describes:
no GitHub-hosted runner has a GPU, so the GPU job that entry names cannot exist. Set to
`cython` in the wheel-test step it does something the entry did not anticipate and this
release needs more — it escalates a missing extension from a silent skip to a failure, per
wheel, per platform, which is exactly how a wheel built without its `.so` would otherwise
pass green. A plain `uv run pytest` job on push and PR lands too, for a reason specific to
this project: ADR 0016 and ADR 0020 make bit-exactness across backends *on one host* the
contract, and this suite has already found host-dependent behaviour twice — libm's `log2`
against numpy's on AVX-512, and phase-3 fork/join costs an order of magnitude apart between
two machines. A second host is new information, not redundancy. The agents-docs sync check
stays deferred: wiring it up first needs a choice between vendoring `build-agents-md.sh`,
reimplementing it, and installing the plugin in the job, and that choice does not improve
for being made on a release branch.

**2026-09-16 — the workflows are written, and one of them carries a line that must not
survive the merge.** `.github/workflows/release.yml` builds the twenty wheels with
cibuildwheel and the sdist with `pipx run build`, and publishes through Trusted Publishing
only when the ref is a `pfsmgraph-hmm-v*` tag; `.github/workflows/test.yml` runs the suite on
push and pull request with `PFSMGRAPH_REQUIRE_BACKENDS=cython,cpu_parallel,torch`.

The per-wheel test is deliberately narrow, and the reason is worth stating because it looks
like under-testing. Only `_viterbi_cython.pyx` and `_forward_backward_cython.pyx` differ
between the twenty wheels; `cpu_parallel`, `cuda` and `torch` are pure Python compiled or
dispatched at runtime and are byte-identical in all of them, so running them per wheel
re-verifies code that cannot vary, at the cost of pulling torch into twenty jobs. They are
left uninstalled and skip, which is legitimate because their `needs` are not in
`_backends.ESCALATED_NEEDS`. `cython`'s **is**, which is what makes a wheel that shipped
without its `.so` abort in `pytest_configure` rather than pass green — and that escalation is
unconditional, so no environment variable is involved. The wheel test is also the first thing
in this project ever to resolve `pfsmgraph-dataseq>=0.1.0` against PyPI rather than against a
workspace path source that satisfies any constraint.

`release.yml` cannot fire from this branch — its triggers are `main`, a tag, and
`workflow_dispatch`, which GitHub offers only for workflows already on the default branch. A
temporary `chore/release-hmm-0.2.0` entry in `on.push.branches` buys one dry run, and the
plan carries removing it as an open subgoal. Two mechanics are unverified until that run: that
`CIBW_TEST_SOURCES` reaches above `package-dir`, and that `pipx run build` yields
meson-python's `git archive` sdist from a depth-1 checkout.
