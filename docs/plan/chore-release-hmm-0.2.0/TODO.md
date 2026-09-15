# chore/release-hmm-0.2.0

**Status**: active
**Created**: 2026-09-15
**Subgoal**: Release `pfsmgraph-hmm` 0.2.0 — `docs/plan/TODO.md`, revision 03-hmm-v0.2.0

## Goals

- [ ] Settle the wheel shape and where it is built
  > **Note:** the repository has no CI (no `.github/`). The master plan's route, cibuildwheel
  > in CI with Trusted Publishing, therefore also fires `DEFERRED.md`'s "CI existing"
  > trigger, whose items (trusted publishers, `PFSMGRAPH_REQUIRE_BACKENDS`, the agents-docs
  > check) must be scoped in or explicitly left behind.
  - [ ] Measure a manylinux build of the extensions: `cibuildwheel` locally on this host,
    `numpy>=2.1` and Cython as build requirements, and whether `-ffp-contract=off` survives
    the manylinux toolchain
  - [ ] Decide the platform and Python matrix (linux x86_64 / aarch64, macOS arm64,
    Windows; cp310–cp314), and whether an sdist-only platform is acceptable
  - [ ] Decide where wheels are built and uploaded from: a GitHub Actions workflow with
    Trusted Publishing, or local cibuildwheel with the existing `.envrc` token
- [ ] Adapt the release tooling to platform wheels
  - [ ] Drop the `build` recipe's `-C setup-args=-Dcompiled=false` for `pfsmgraph-hmm`, and
    decide whether `meson.options`' `compiled` stays as a source-build escape hatch
  - [ ] Teach `preflight` platform tags, and make `publish` upload every wheel plus the sdist
  - [ ] Update `docs/ops/release.md` and the `justfile` comments; keep
    `tests/test_release_runbook.py` green
- [ ] Settle what 0.2.0's immutable metadata says
  - [ ] Restore `Programming Language :: Cython`; review `description`, the `cpu-parallel`,
    `gpu` and `torch` extras' bounds, and the `pfsmgraph-dataseq` lower bound against the
    `dataseq` API 0.2.0 uses
  - [ ] State the wheel shape in the member README where it matters, and check
    `backends.md`'s remedy for a missing extension against what PyPI will carry
- [ ] Verify what a consumer installs
  - [ ] Build 0.2.0.dev0 wheels and the sdist; install each available wheel into a clean
    venv outside the workspace: `py.typed`, the `.so` files, no `pfsmgraph/__init__.py`,
    `backends()` reporting `cython ✓`, the README examples
  - [ ] Check the installed compiled kernels bit-exact against the numpy reference
    (a platform wheel built with contraction on would pass every tolerance test)
  - [ ] Build and install from the sdist alone, on a platform with no wheel
- [ ] Release: publish, tag, and close what the branch discharged
  - [ ] **User:** set up upload credentials per goal 1's decision: either attach a Trusted
    Publisher for `pfsmgraph-hmm` on PyPI, or confirm the `.envrc` token still reads
    (`direnv exec . just token >/dev/null && echo ok`)
  - [ ] Release commit: bump `version` to `0.2.0`, relock, rebuild and re-verify
  - [ ] **User:** publish (irreversible): `just release 0.2.0`, or the tagged workflow run
  - [ ] Confirm PyPI's digests match the verified build, the tag `pfsmgraph-hmm-v0.2.0`
    is on `origin`, and `just verify 0.2.0` installs from PyPI
  - [ ] Record commit: the "released" statements in `core.md`, the root README and the
    `docs/api/hmm/` pages
