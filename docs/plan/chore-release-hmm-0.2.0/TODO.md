# chore/release-hmm-0.2.0

**Status**: active
**Created**: 2026-09-15
**Subgoal**: Release `pfsmgraph-hmm` 0.2.0 — `docs/plan/TODO.md`, revision 03-hmm-v0.2.0

## Goals

- [x] Settle the wheel shape and where it is built
  > **Done:** 0.2.0 ships twenty platform wheels plus the sdist, built by a GitHub Actions
  > cibuildwheel workflow across linux x86_64/aarch64, macOS arm64 and Windows x86_64 on
  > cp310–cp314, and published by a tagged run through PyPI Trusted Publishing. The
  > repository gains its first `.github/`, so the "CI existing" trigger fires here: trusted
  > publishers are attached as part of the route, `PFSMGRAPH_REQUIRE_BACKENDS=cython`
  > escalates a missing extension in the wheel-test step, a plain `uv run pytest` job lands
  > on push and PR, and the agents-docs sync check stays deferred with its reason recorded.
  > **Note (corrected 2026-09-16, while working goal 2):** the two statements above about
  > `PFSMGRAPH_REQUIRE_BACKENDS=cython` are wrong, and are left standing rather than edited
  > because the reasoning that produced them is the thing worth not repeating. The variable
  > is a **no-op for `cython`**. `_backends.detect()` raises `BackendError` unconditionally
  > whenever a row's `needs` is in `ESCALATED_NEEDS = {None, "compiled extension"}`, and
  > `cython`'s `needs` is exactly `"compiled extension"`; the root `conftest.py` calls
  > `detect()` *before* `check_required()`, so a missing `.so` aborts the session before the
  > environment is ever read. The escalation the release wants is already unconditional and
  > costs nothing to obtain — it arrives with the root conftest. What the variable can still
  > buy is `cpu_parallel` and `torch`, whose `needs` are **not** escalated and which do skip
  > silently; goal 2's dependency decision declines to install either per wheel, so the
  > variable is not set at all. `DEFERRED.md`'s entry therefore stays open in full rather
  > than half-discharged: neither the GPU job it names nor a non-GPU substitute lands here.
  > **Note:** the repository has no CI (no `.github/`). The master plan's route, cibuildwheel
  > in CI with Trusted Publishing, therefore also fires `DEFERRED.md`'s "CI existing"
  > trigger, whose items (trusted publishers, `PFSMGRAPH_REQUIRE_BACKENDS`, the agents-docs
  > check) must be scoped in or explicitly left behind.
  > **Note:** the local measurement this goal opens with is not runnable on this host, and
  > that is what settles the goal rather than a preference. `cibuildwheel --platform linux`
  > builds inside a manylinux container, and there is no container engine here — no `docker`,
  > no `podman`. The container-free fallback measures narrower than it looks: `auditwheel
  > repair` needs no Docker, but it can only claim the host's own floor, and this host is
  > Ubuntu 24.04 on **glibc 2.39**, so a locally repaired wheel is `manylinux_2_39_x86_64` —
  > a tag PyPI accepts and almost nothing in the wild installs, against a common floor of
  > `manylinux_2_28` and a manylinux2014 floor of glibc 2.17. And this is a Linux x86_64
  > box, so macOS and Windows wheels are unbuildable here with or without containers. The
  > matrix is therefore not independent of where the build runs; the third subgoal decides
  > what the first two are allowed to answer, and was taken first.
  > **Q:** Where are 0.2.0's wheels built and uploaded from — a GitHub Actions cibuildwheel
  > workflow with Trusted Publishing, the same workflow uploading artifacts you publish
  > locally with the existing `.envrc` token, a single locally built `manylinux_2_39_x86_64`
  > wheel plus the sdist, or the sdist alone?
  > **A:** GitHub Actions + Trusted Publishing. The full matrix, published by a tagged
  > workflow run; the only route that gives macOS and Windows pip users `cython ✓`.
  > **Q:** Which platform and Python matrix?
  > **A:** linux x86_64 and aarch64, macOS arm64, Windows x86_64; cp310–cp314. Twenty
  > wheels. The repository is public, so `ubuntu-24.04-arm` is free and aarch64 builds
  > natively rather than under QEMU. Intel macOS is left to the sdist: a shrinking audience
  > on a runner image GitHub is retiring, and a fifth platform whose bit-exactness goal 4
  > would have to verify. cp315 does not exist yet, so cp310–cp314 is the whole range
  > `requires-python = ">=3.10"` admits.
  > **Q:** Which of `DEFERRED.md`'s "CI existing" items does this branch scope in, now that
  > `.github/` will exist?
  > **A:** Two of three. `PFSMGRAPH_REQUIRE_BACKENDS=cython` in the wheel-test step, and a
  > plain `uv run pytest` job on push and PR. The agents-docs sync check stays deferred:
  > wiring it up first needs a choice between vendoring `build-agents-md.sh`, reimplementing
  > it, or installing the plugin in the job, and that choice does not get better for being
  > made on a release branch.
  - [-] Measure a manylinux build of the extensions: `cibuildwheel` locally on this host,
    `numpy>=2.1` and Cython as build requirements, and whether `-ffp-contract=off` survives
    the manylinux toolchain
    > **Descoped:** not runnable here — no container engine, per the note above — and
    > moot once the build moves to CI, where cibuildwheel supplies the manylinux image.
    > The question inside it survives and is not lost: whether `-ffp-contract=off` reaches
    > the extension is answered by goal 4's bit-exactness check against the *installed*
    > wheel, which is the stronger form of the same test, since it measures what a consumer
    > gets rather than what the toolchain accepted. A contracting build passes every
    > tolerance test in the suite and fails only that one.
  - [x] Decide the platform and Python matrix (linux x86_64 / aarch64, macOS arm64,
    Windows; cp310–cp314), and whether an sdist-only platform is acceptable
    > **Note:** sdist-only is accepted for Intel macOS, 32-bit Windows, musl Linux and any
    > older glibc than cibuildwheel's default manylinux image. Those install wherever a C
    > compiler, Cython and the numpy headers are present, which `build-system.requires`
    > supplies under pip's build isolation.
    > **Note:** cp314 carries a test-only wrinkle, not a build one. `hmmlearn`, the
    > forward-backward and EM oracle added 2026-09-14, ships no cp314 wheels, so whatever
    > the wheel-test step runs on that one interpreter cannot be the full suite. Building
    > the cp314 wheel is unaffected — `hmmlearn` is in the `dev` group and is neither a
    > build requirement nor a dependency of any member.
  - [x] Decide where wheels are built and uploaded from: a GitHub Actions workflow with
    Trusted Publishing, or local cibuildwheel with the existing `.envrc` token
    > **Note:** `publish` in the justfile called itself "THE SWITCHING POINT" — moving a
    > package to Trusted Publishing was meant to be an edit to that recipe and nothing
    > else. That holds for the upload and not for the release: `release` has `build` as a
    > prerequisite and `build` runs `clean`, so a recipe chain that ends in `publish`
    > cannot also consume wheels an external job produced. Goal 2 inherits that.
    > **Note:** `PFSMGRAPH_REQUIRE_BACKENDS`'s `DEFERRED.md` entry says to wire it into the
    > GPU job, and no GitHub-hosted runner has a GPU, so that half cannot be discharged by
    > this workflow and stays deferred. The variable's use here is better than its own
    > wording: set to `cython` in the wheel-test step it escalates a missing extension from
    > a silent skip to a failure, per wheel, per platform — which is exactly how a wheel
    > built without its `.so` would otherwise pass green. It is also the first code path
    > ever to exercise the variable, so its first real use is still its first test.
- [~] Write the GitHub Actions workflows
  > **Note:** added 2026-09-16, once goal 1 chose the CI route. This is the repository's
  > first `.github/`, so nothing here has a local precedent to copy.
  > **Note:** only two files in this distribution are platform-dependent —
  > `_viterbi_cython.pyx` and `_forward_backward_cython.pyx`. `cpu_parallel`, `cuda` and
  > `torch` are pure Python compiled or dispatched at runtime, so they are byte-identical in
  > all twenty wheels and running them per wheel re-verifies code that cannot differ. That
  > is what scopes the wheel test: it exists to prove the extension shipped, imports, and is
  > bit-exact — the one check a contracting build fails while passing every tolerance test.
  > **Note:** the wheel test is also the first thing in this project ever to resolve
  > `pfsmgraph-dataseq>=0.1.0` for real. A `{ workspace = true }` path source satisfies any
  > constraint, and a workspace member's `requires-dist` records no specifier, so neither a
  > local run nor `uv.lock` can see that bound — `core.md`'s workspace footgun. cibuildwheel
  > installs the wheel from disk and pulls its dependency from PyPI, which is the first
  > resolution of that edge outside a pip user's machine.
  > **Q:** What does the per-wheel test step run?
  > **A:** The whole `pfsmgraph-hmm` suite with `pytest` as the only test requirement.
  > `CIBW_TEST_SOURCES` copies `conftest.py`, `_backends.py`,
  > `packages/pfsmgraph-hmm/tests/` and the `.scratch/hmm-lush/Training/` fixtures, since 8
  > of the 12 test files read them; the `hmmlearn` oracle is ignored, leaving 1271 collected.
  > `cpu_parallel`, `cuda` and `torch` params skip cleanly because their `needs` are not
  > escalated, while every `cython` row runs. Uniform across all twenty wheels, cp314
  > included — the cp314 wrinkle goal 1 recorded dissolves, because the oracle was the only
  > thing wanting `hmmlearn` and it is not run here.
  > **Q:** When does the workflow build wheels, as opposed to publish them?
  > **A:** Build on push to `main`, on a `pfsmgraph-hmm-v*` tag, and on `workflow_dispatch`;
  > publish only on the tag. A platform-specific compile failure then surfaces on a main
  > push, while it is still cheap, rather than at the moment a tag has already been pushed.
  > Pull requests are excluded: a 20-wheel matrix on every docs-only PR needs path filters,
  > and path filters are their own source of silent misses.
  - [x] `release.yml`: cibuildwheel over goal 1's matrix plus an sdist job, building with
    the default `-Dcompiled=true`; `CIBW_BUILD` pinned to cp310–cp314, and a
    `CIBW_TEST_COMMAND` carrying `PFSMGRAPH_REQUIRE_BACKENDS=cython`
    > **Note:** written without the environment variable, per the correction under goal 1 —
    > it would have been a no-op. `CIBW_SKIP` drops musllinux and PyPy, which goal 1 had
    > already accepted as sdist-only. Every action is pinned to the release current on
    > 2026-09-16: `checkout@v7.0.1`, `cibuildwheel@v4.2.1`, `upload-artifact@v7.0.1`,
    > `download-artifact@v8.0.1`, `setup-uv@v10.1.0`, `gh-action-pypi-publish@v1.14.2`.
    > **Note:** `upload-artifact` is v7 and `download-artifact` is v8 — their majors do not
    > track each other, and the mismatch would surface only in the publish job, which is
    > the one moment a tag has already been pushed.
    > **Note:** the sdist is built with `pipx run build`, not `uv build`. The workspace root
    > sets `[tool.uv] no-build-isolation-package` for all five members, so `uv build` looks
    > for meson-python in an environment CI has not synced; `build` uses ordinary PEP 517
    > isolation and reads `build-system.requires`, which is also what a pip user installing
    > from the sdist does.
  - [x] Decide what that wheel test can actually run: the suite lives outside the wheel
    (`packages/*/tests/` and the repo-root `tests/`), reads `.scratch/` fixtures in place,
    and wants `hmmlearn`, which has no cp314 wheels
    > **Note:** the repo-root `tests/` are excluded and should stay excluded. They check the
    > *checkout*, not the wheel — `test_meson_sources.py` reads `packages/*/meson.build`,
    > `test_release_runbook.py` reads the justfile, `test_api_docs.py` reads `docs/`. None of
    > them says anything about an installed distribution.
    > **Note (measured 2026-09-16):** a prediction here was wrong, and the measurement is
    > what caught it. The concern was that with no ini file in the copied tree, pytest's
    > `rootdir` would land on the tests directory, putting the root `conftest.py` above the
    > default `confcutdir` and silently dropping the ADR 0003 header and its escalation —
    > which is exactly the silent failure `core.md` documents for that file. Two simulated
    > test trees were built under the scratchpad, one with the root `pyproject.toml` and one
    > without, and **both printed the header and collected 1292**: with no ini file pytest
    > falls back to the *cwd*, not to the args' common ancestor, so the conftest sat inside
    > `confcutdir` either way. `pyproject.toml` is still copied, for a different reason —
    > `addopts = "-ra"`, without which pytest prints a bare `s` and swallows the skip reason
    > ADR 0003 requires the run's own output to state.
  - [x] Publish job gated on a `pfsmgraph-hmm-v*` tag, with `id-token: write` and a named
    environment; confirm a branch push uploads nothing
    > **Note:** `permissions: contents: read` at workflow level, with `id-token: write`
    > raised on the publish job alone; environment `pypi`. The gate is
    > `startsWith(github.ref, 'refs/tags/pfsmgraph-hmm-v')`, so every other trigger builds
    > and stops. The *confirmation* half comes from the dry run below, not from reading it.
    > **Note:** `gh-action-pypi-publish`'s own release notes record that GitHub's OIDC token
    > lives 5 minutes, and that projects with many large wheels have hit publish timeouts
    > inside it. Twenty wheels of two small extensions is a few MB total, well inside — but
    > the figure is worth carrying if this matrix ever grows.
  - [x] `test.yml`: `uv sync` and `uv run pytest` on push and PR; record that the 206
    numba-cuda tests skip on any hosted runner
    > **Note:** this is where `PFSMGRAPH_REQUIRE_BACKENDS` finally earns its place, set to
    > `cython,cpu_parallel,torch`. Those three *are* installed by `uv sync`, so a skip would
    > mean an import that broke after a resolution moved — the green run whose header nobody
    > reads. `cuda` is deliberately absent: no hosted runner has a device, so its 206 tests
    > skip and the GPU half of the `DEFERRED.md` entry stays open, unchanged.
  - [ ] **Before merge:** remove the temporary `chore/release-hmm-0.2.0` entry from
    `release.yml`'s `on.push.branches`, once the dry run has confirmed the two unverified
    mechanics
    > **Note:** the workflow cannot fire from this branch as designed — its triggers are
    > `main`, a tag, and `workflow_dispatch`, and GitHub offers dispatch only for workflows
    > already on the default branch. Without the temporary entry the 20-wheel matrix would
    > first execute on the merge commit, carrying two unverified mechanics across a merge:
    > whether `CIBW_TEST_SOURCES` reaches above `package-dir` (the docs say its paths are
    > relative to the *project* root, the directory cibuildwheel was called from, but this
    > has not been run), and whether `pipx run build` produces meson-python's `git archive`
    > sdist from a depth-1 checkout. The fallbacks if either fails are a `CIBW_BEFORE_ALL`
    > copy step and `fetch-depth: 0` respectively. The publish job is unaffected either way:
    > it is gated on the tag, not on the branch.
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
  - [ ] **User:** attach a Trusted Publisher for `pfsmgraph-hmm` on PyPI, per goal 1
    > **Note:** ordered after the workflow goal, not before it. PyPI's trusted publisher
    > form asks for the workflow *filename* and the environment name, so it cannot be
    > filled in until `release.yml` exists and both are settled. The `.envrc` token is not
    > retired by this — it stays as the fallback path `publish` still implements.
  - [ ] Release commit: bump `version` to `0.2.0`, relock, rebuild and re-verify
  - [ ] **User:** publish (irreversible): push the `pfsmgraph-hmm-v0.2.0` tag to trigger
    the release workflow's publish job
  - [ ] Confirm PyPI's digests match the verified build, the tag `pfsmgraph-hmm-v0.2.0`
    is on `origin`, and `just verify 0.2.0` installs from PyPI
  - [ ] Record commit: the "released" statements in `core.md`, the root README and the
    `docs/api/hmm/` pages
