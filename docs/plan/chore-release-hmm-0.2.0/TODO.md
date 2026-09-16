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
- [x] Write the GitHub Actions workflows
  > **Done:** `.github/workflows/release.yml` and `test.yml`, the repository's first CI,
  > proven by run `35049292620` before the trigger that enabled it was taken back out —
  > twenty wheels over four platforms on cp310–cp314 with `cython ✓` against each installed
  > wheel, a glibc 2.17 manylinux floor, an sdist, and `publish` correctly skipped. Two
  > findings outlived the goal and are recorded where they act: the per-run cost and the
  > public-repo condition, under this goal; and the tag-versus-version guard that moving the
  > release into CI dropped, as a new subgoal of goal 3.
  > **Note:** added 2026-09-16, once goal 1 chose the CI route. This is the repository's
  > first `.github/`, so nothing here has a local precedent to copy.
  > **Note (measured 2026-09-16, run 35049292620):** what one full matrix run costs, and the
  > condition that keeps it free. GitHub meters every run at list price regardless of
  > visibility and then applies a public-repository discount as a separate credit, so the
  > billing page reads `$0.38 consumed / -$0.38 discounts / $0.00 billable`. The `$0.38` is
  > the counterfactual — what this run would cost on a private repo — not money owed. By
  > category: macOS $0.25, Windows $0.08, Linux $0.03, Linux ARM $0.03. **macOS is ~64% of
  > the metered total despite being the third-shortest job**, entirely because of its 10x
  > multiplier; Windows is 2x, Linux 1x. Public-repo Actions minutes are free and unlimited
  > and are *not* drawn from the account's included allowance, which is why the discount is
  > a credit rather than an allowance deduction — so the Pro subscription buys nothing here
  > that "unlimited" had not already.
  > **Note:** that makes **`pfsmgraph` staying public a standing condition of this release
  > design**, not a fact about today, and it fails discontinuously rather than gradually. The
  > same workflow on a private repository draws roughly 66 billable minute-equivalents per
  > run against a 3,000/month allowance — about 45 runs before Actions' default $0 spending
  > limit halts them rather than charges. If that ever binds, the cheapest lever is dropping
  > the macOS job: two-thirds of the cost for one platform. Recorded here because the number
  > on the billing page reads `$0.00` and carries none of this with it.
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
  - [x] **Before merge:** remove the temporary `chore/release-hmm-0.2.0` entry from
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
    > **Ran (2026-09-16):** run `35049292620`, pushed with commit `df23146`. **Success, 7m48s
    > wall clock**, and both unverified mechanics are confirmed. `CIBW_TEST_SOURCES` does
    > reach above `package-dir`: the suite ran in every wheel's test venv, including the
    > tests that read `.scratch/hmm-lush/Training` four directories up. `pipx run build`
    > produced the sdist from a depth-1 checkout, so no `fetch-depth: 0` is needed. Twenty
    > wheels, five per platform, cp310–cp314; `publish to PyPI` **skipped**, as the gate
    > requires. `cython ✓` against the installed wheel on all four platforms, with
    > `cpu_parallel`, `cuda` and `torch` reporting absent and naming their extras — around
    > 748 passed / 523 skipped per interpreter, summing to the 1271 collected.
    > **Note:** the manylinux tag is the measurement goal 1 could not take locally. The
    > wheels carry `manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64` — a
    > **glibc 2.17** floor, against the `manylinux_2_39` this host's `auditwheel` was limited
    > to. That is the whole argument for the CI route, now measured rather than reasoned.
    > **Note:** the pass/skip split is not uniform across the matrix, and that is the suite
    > working rather than flaking: Linux 748, macOS 747, Windows **mixed across
    > interpreters** (two at 748, three at 747). The differing test is the libm-vs-numpy
    > `log2` discrimination check, which skips with its own reason — *"numpy's and libm's
    > log2 agree on every tied candidate drawn on this host (numpy 2.5.3), so there is no
    > split for the tie to catch"*. Its outcome varies by host **and by numpy build**, which
    > is the second-host argument for `test.yml` demonstrated rather than asserted, and it is
    > legible only because `-ra` is in the copied `pyproject.toml`.
    > **Note:** the dry run has now served its purpose, so leaving the trigger in place only
    > costs noise — every further push to this branch, through goals 3 to 5, fires the full
    > twenty-wheel matrix.
    > **Done (2026-09-16):** removed. `on.push.branches` is back to `[main]`, the tag trigger
    > and `workflow_dispatch` unchanged, and the file parses. Goals 3 to 5 now push to this
    > branch without firing the matrix; the next run of `release.yml` will be the merge
    > commit's push to `main`, which builds and publishes nothing.
- [x] Settle what the release tooling means now that CI builds the artifact
  > **Done:** the justfile now carries two release paths and refuses the wrong one in both
  > directions — `release` for members shipping a pure wheel, `release-ci` for members whose
  > artifact GitHub Actions builds, selected by a `ci_built_packages` variable rather than
  > spelled per recipe. `build` lost `-Dcompiled=false` and became a verification tool;
  > `meson.options`' `compiled` stays as a source-build escape hatch; `preflight`'s
  > `py3-none-any` assertion was examined and deliberately kept; the tag-versus-version
  > assertion exists on both sides, in `_version-matches` and in the publish job. Four guards
  > and the workflow check were each exercised in both directions before being trusted.
  > **Note:** retitled and rewritten 2026-09-16. The old title, "Adapt the release tooling to
  > platform wheels", and its three subgoals assumed `just release` stays `pfsmgraph-hmm`'s
  > release path and needs only to be taught about new filenames. It does not. The artifact
  > hmm ships is twenty wheels this host cannot build — three of the four platforms are
  > physically unreachable from here — and the question the goal has to answer first is what
  > `just release 0.2.0 pfsmgraph-hmm` *means* once that is true, not which recipes to edit.
  > **Q:** What does `just release` mean for a member whose shipped artifact this machine
  > cannot produce? (A) `release` refuses for hmm and points at the tag flow; (B) add
  > `release-ci`, which pushes the tag and watches the run, so `just` stays the single entry
  > point without ever touching artifacts; (C) `just` downloads the CI artifacts and
  > publishes them with the `.envrc` token; (D) keep the local build as a *verification*
  > tool under an honest name, severed from "release".
  > **A:** **A + B + D, agreed.** C is the only answer that literally gets the CI artifacts
  > to `just`, and it costs the thing the route was chosen for.
  > **Note:** C was declined on its cost, not on feasibility, and the difference is measured
  > rather than assumed: `gh run download 35049292620` fetched all 21 files — twenty wheels
  > and the sdist — in seconds on 2026-09-16. The artifacts are ordinary, reachable, and
  > retained 90 days. What makes C expensive is that **Trusted Publishing cannot be driven
  > from this machine at all**: it is OIDC, and the token is minted by GitHub Actions for a
  > specific repository, workflow and environment, with no local equivalent. Publishing
  > locally therefore means reverting to the `.envrc` token and losing PEP 740 attestations,
  > which are signed against the Trusted Publishing identity. The justfile already documents
  > the symptom — an empty `UV_PUBLISH_TOKEN` "falls through to trusted-publishing
  > discovery, which resolves only inside CI".
  > **Note:** the consequence to keep in view while editing: this repository will have **two
  > release mechanisms**, and that is the decision rather than an accident. `just release`
  > stays the path for the four members that ship pure wheels; a pushed tag is the path for
  > any member with a compiled artifact. `docs/ops/release.md` has to say which is which in
  > as many words, because a runbook that describes one path while the repository has two is
  > worse than a runbook that describes neither.
  - [x] Make `release` refuse for `pfsmgraph-hmm`, as a **prerequisite** and never a body line
    > **Done:** a `ci_built_packages` variable beside `default_package` names the members CI
    > builds, and `_local-release-refused` is the first prerequisite of `release`. A member
    > joins that list when its first public call reaches a compiled kernel, so the rule is
    > stated once rather than spelled per recipe.
    > **Note:** `_ci-release-refused` was added as its mirror, which was not in the subgoal.
    > Without it `just release-ci 0.1.0 pfsmgraph-dataseq` pushes a tag no workflow matches
    > and reports success, having released nothing — a guard that points one way leaves the
    > other direction silently wrong.
    > **Note:** `just release` with no arguments now refuses, because `default_package` is
    > `pfsmgraph-hmm`. That reads like a regression and is the safest available default: the
    > justfile's existing comment claims an omitted argument "can never reach a published
    > package", and this strengthens it from *fails at preflight* to *refuses before running
    > anything*.
    > **Ran:** all four guards exercised directly, both directions each —
    > `_local-release-refused` exits 1 for hmm and 0 for dataseq, `_ci-release-refused` the
    > reverse, `_version-matches` exits 0 at 0.1.0 and 1 at 0.2.0 naming the declared
    > version. Tested through the private recipes rather than through `release`, so no
    > release chain could start.
    > **Note:** the justfile states this rule about itself already — "a guard must be a
    > *prerequisite* of `release`, never a body line, because every body line runs after
    > `publish`, the irreversible step". A refusal written into the body would run after the
    > upload it exists to prevent.
    > **Note:** today the command already fails, but by accident rather than by design:
    > `preflight` demands a `py3-none-any` filename and a local hmm build no longer produces
    > one. That is a filename check standing in for a policy, and the subgoal below on
    > `preflight` is what keeps the two from being confused.
  - [x] Add `release-ci <version> [package]`: assert the tree is clean and pushed **and that
    `pyproject.toml`'s version matches `<version>`**, then tag and push, and report the run
    > **Done:** `release-ci version package=default_package: (_ci-release-refused package)
    > (_version-matches version package) test _tree-pushed`, with only the tag and its push
    > in the body — everything checkable is checked before the trigger exists.
    > **Q:** Should `release-ci` run the suite locally before tagging, and should it watch
    > the run afterwards?
    > **A:** Run the suite, yes — it mirrors `release` and is cheap against a tag that cannot
    > be un-pushed cleanly. Watch, no: print `gh run watch --exit-status` and the Actions URL
    > instead, so `gh` stays out of this file's requirements, which remain just, uv and git.
    > **Note:** `_version-matches` reads the version with `grep -m1 '^version = ' | cut -d'"'
    > -f2` rather than `tomllib`, because a recipe may run on a stock `python3` older than
    > 3.11. Each member's `pyproject.toml` carries exactly one matching line, checked.
    > **Note:** this is where the justfile's own preflight assertion survives the move. Its
    > comment — "without this assertion `just release 0.2.0` would publish 0.1.0 and tag it
    > v0.2.0. Both halves of that are irreversible" — describes a hazard the tag flow has
    > too, since the tag is what triggers the upload. Local and workflow-side guards are
    > both wanted: a tag can be pushed by `git` directly, so the workflow cannot rely on
    > this recipe having been used.
  - [x] Drop the `build` recipe's `-C setup-args=-Dcompiled=false` for `pfsmgraph-hmm`, and
    decide whether `meson.options`' `compiled` stays as a source-build escape hatch
    > **Done:** dropped; `build` is now one unconditional `uv build --package`, and the
    > package-name conditional that existed only to carry that flag is gone with it.
    > **Decided:** `compiled` **stays**, as a source-build escape hatch and never a release
    > setting. It is the only way to install from the sdist on a machine with no C compiler,
    > trading a pure install that reports `cython ✗` honestly for one that fails to build at
    > all. Nothing in the release path sets it and nothing should — a release wheel is
    > exactly the case that must not use it.
    > **Note:** under (D) this recipe's job changes rather than disappears. It stops being
    > the first step of a release and becomes the way goal 4 produces a local wheel to
    > install into a clean venv, so it should be named and commented for that.
  - [x] Re-examine `preflight`'s `py3-none-any` assertion rather than relaxing it
    > **Done:** examined and **kept unrelaxed**, which was the point. No member publishes
    > platform wheels from here, so the assertion stays correct for every member still on the
    > local path, and teaching it platform tags would have deleted the last check between a
    > local hmm release and an upload. The explicit refusal in `release` is now the policy;
    > this filename check is a second line behind it.
    > **Done:** the tree half — the dirty check and `git push origin HEAD` — is extracted into
    > `_tree-pushed` and taken as a prerequisite by both `preflight` and `release-ci`.
    > **Note:** one behaviour change, small and deliberate. As a prerequisite `_tree-pushed`
    > now runs *before* `preflight`'s `dist/` checks rather than after, so a run whose wheel
    > is missing pushes `HEAD` first. Pushing a clean committed tree is idempotent and
    > harmless, and the alternative was duplicating the two lines in two recipes.
    > **Note:** this replaces "Teach `preflight` platform tags", which was the wrong task and
    > actively harmful. Under A + B + D no member publishes platform wheels locally, so the
    > assertion stays *correct* for the four pure members — and teaching it to accept
    > platform tags would have deleted the check that currently stops a local hmm release
    > from proceeding, leaving PyPI's refusal of a bare `linux_x86_64` tag as the only
    > backstop. A check we were about to relax turns out to be load-bearing in a new way.
    > Worth splitting, though: `preflight` also asserts the tree is clean and pushes it, and
    > `release-ci` wants that half without the filename half.
  - [x] Restore the tag-versus-version guard that moving the release into CI dropped
    > **Done:** a step in `release.yml`'s publish job, between the artifact downloads and the
    > upload, failing unless every file in `dist/` carries `github.ref_name` minus the
    > `pfsmgraph-hmm-v` prefix. Deliberately redundant with `_version-matches`: a tag can be
    > pushed with plain `git`, so CI cannot assume the recipe was used, and the recipe should
    > not push a tag it has not checked. Either can be bypassed alone.
    > **Ran:** the check was tested against this workflow's own artifacts before being added.
    > Over all 21 files of run `35049292620` it passes at their real version and flags every
    > one of them against a tag claiming another.
    > **Note (found 2026-09-16 in the dry run):** `release.yml` has no equivalent of the
    > justfile's `preflight`, and nothing in it compares the pushed tag to the version
    > actually built. The dry run made that concrete: it produced
    > `pfsmgraph_hmm-0.1.0-cp312-…-manylinux….whl`, because `pyproject.toml` still reads
    > 0.1.0 until goal 5's bump — and that filename **differs** from the `py3-none-any` wheel
    > already published for 0.1.0, so PyPI would accept it as an additional file on a
    > released version rather than rejecting it. The fix is a step in the publish job that
    > fails unless the built version matches `github.ref_name` minus the `pfsmgraph-hmm-v`
    > prefix.
  - [x] Update `docs/ops/release.md` and the `justfile` comments; keep
    `tests/test_release_runbook.py` green
    > **Done:** the runbook gains a *Two release paths* section at the head of its justfile
    > chapter — a table of which member takes which, why a member moves between them, that
    > `just` refuses the wrong one in both directions, and that Trusted Publishing cannot be
    > driven from here whatever the artifacts' accessibility. Four existing passages were
    > corrected rather than appended to: the `-Dcompiled=false` paragraph, now history plus
    > the escape-hatch decision; what `build` is *for* once it is not a release step; why
    > `preflight` was not relaxed; and the "suggested posture" paragraph.
    > **Note:** that last correction is the interesting one. The posture paragraph
    > recommended piloting Trusted Publishing "on a member where a botched release costs
    > nothing" — and the pilot is happening on `pfsmgraph-hmm`, the most-developed member in
    > the family, for an unrelated reason: its wheels are built by Actions, so the upload
    > happens there. The conclusion survived its reasoning, which is worth marking rather
    > than quietly letting the text look prescient. Its other prediction is half-wrong and
    > marked so: "the `justfile` is what makes that a one-recipe edit" holds for the upload,
    > since `publish` is untouched, and fails for the release, since `release` has `build` as
    > a prerequisite and `build` runs `clean`.
    > **Ran:** `tests/test_release_runbook.py` 2 passed, and the full suite 1418 passed in
    > 4m14s. The count is worth writing down because goal 5's record commit has to correct
    > the root README's stale `709`: as of 2026-09-16 it is 1418 = 74 dataseq + 1292 hmm +
    > 52 root — but re-measure there rather than copying this, for the reason given.
    > **Note:** that test asserts every recipe `release.md` names exists, so `release-ci` has
    > to be named there rather than only in the justfile. The runbook's "Token sprawl vs.
    > Trusted Publishing" section also needs its "suggested posture" revisited: it recommends
    > piloting Trusted Publishing on a member where a botched release costs nothing, and this
    > branch is that pilot happening.
- [x] Settle what 0.2.0's immutable metadata says
  > **Done:** the metadata is frozen as: `3 - Alpha`, `Programming Language :: Cython`
  > restored, one classifier per shipped interpreter, `description` unchanged,
  > `pfsmgraph-dataseq>=0.1.0` and `numpy>=2.1` as dependencies, and the three extras'
  > bounds unchanged with the `numpy<2.5` cap kept. Three of the four bounds were checked
  > against PyPI rather than against the workspace, and the one documentation surface that
  > cannot be corrected after upload — the member README, which becomes the long
  > description — now states the wheel shape.
  > **Note:** nothing in this goal changed a shipped module, deliberately. The only
  > candidate was `_backends.py`'s missing-extension message, and it was left alone because
  > it is already correct for a world where PyPI carries platform wheels.
  > **Amended 2026-09-16, while verifying goal 5:** one more field joins the frozen set —
  > `license-files = ["LICENSE"]`. The member `LICENSE` has **never reached the wheel**.
  > `hmm` 0.1.0 is on PyPI with `License-Expression: MIT` in METADATA and no license text at
  > all: valid PEP 639, accepted by `twine check`, rendered as MIT by PyPI, and detectable
  > only by opening the archive. meson-python ships no license file unless `license-files` is
  > declared; `pfsmgraph-dataseq` 0.1.0 does carry `dist-info/licenses/` because hatchling
  > globs `LICENSE*` by default, which is why the gap survived the 0.1.0 release unnoticed.
  > Declaring it produces `dist-info/licenses/LICENSE` (1077 bytes) and a `License-File:`
  > field, matching what hatchling gave `dataseq`. None of the five members declared it;
  > `align`, `hseg` and `dl` should before their first release, where it still costs nothing.
  - [x] Restore `Programming Language :: Cython`; review `description`, the `cpu-parallel`,
    `gpu` and `torch` extras' bounds, and the `pfsmgraph-dataseq` lower bound against the
    `dataseq` API 0.2.0 uses
    > **Q:** Which classifiers should 0.2.0 carry, and should Development Status move from
    > `3 - Alpha` to `4 - Beta`?
    > **A:** `Programming Language :: Cython` plus one row per interpreter shipped —
    > `3.10` through `3.14`, mirroring `CIBW_BUILD` exactly, so PyPI's sidebar answers "is my
    > Python covered" without reading wheel filenames. Operating-system classifiers were
    > declined: they read as "only these are supported" when the sdist builds anywhere.
    > **Development Status stays 3 - Alpha.** Topology search, which the description itself
    > names as forthcoming, is revision 04, and the public surface grew substantially across
    > one minor version. Understating maturity costs a reader nothing; overstating it invites
    > reliance the version cannot support.
    > **Note:** `description` reviewed and unchanged. It already names batched decoding,
    > Baum-Welch training and selectable backends, and marks topology search forthcoming —
    > accurate for what 0.2.0 ships.
    > **Ran (2026-09-16):** `pfsmgraph-dataseq>=0.1.0` verified **against PyPI rather than
    > against the workspace**, which is the first time that bound has been checked by
    > anything. `uv run --no-project --with 'pfsmgraph-dataseq==0.1.0'` installed the
    > published wheel, and all five names `hmm` imports — `RESERVED_SYMBOLS`, `USER_BASE`,
    > `Vocabulary`, `SequenceRecord`, `pad_collate` — are in its 15-name `__all__`. This is
    > exactly what `core.md`'s workspace footgun says review is the only mechanism for: a
    > path source satisfies any constraint, and `uv.lock` records no specifier for a
    > workspace member, so neither a local run nor the lockfile can see this bound.
    > **Ran:** the extras' bounds checked against PyPI's current releases. `numba` is at
    > 0.67.0 against a `>=0.61` floor and `torch` at 2.14.0 against `>=2.13` — both honest.
    > **`numba-cuda` is still 0.30.4**, the same release whose `np.row_stack` call at import
    > breaks on numpy 2.5, so the `numpy<2.5` cap in `gpu` **stays**. Its own comment says to
    > drop it when a numba-cuda release stops doing that; re-checking rather than inheriting
    > matters here because the bound is about to become immutable.
  - [x] State the wheel shape in the member README where it matters, and check
    `backends.md`'s remedy for a missing extension against what PyPI will carry
    > **Done:** a paragraph directly under the README's `pip install pfsmgraph-hmm`, naming
    > the four platforms and cp310–cp314, saying that anywhere else pip builds from the
    > sdist and needs a C compiler, that `backends()` reports `cython ✓` either way, and that
    > a pure install remains available from source via `-Dcompiled=false` and reports
    > `cython ✗` rather than pretending otherwise. It is prose, not a code block, so the
    > ADR 0013 verifier has nothing new to execute — `tests/test_api_docs.py` 11 passed.
    > **Note:** `backends.md`'s remedy needs **no change**, and there are two reasons rather
    > than one. It reads "install a platform wheel, or build from source with a C compiler",
    > which was aspirational at 0.1.0 and becomes actionable at 0.2.0 now that PyPI carries
    > platform wheels — the check this subgoal asked for, and it passes. Separately, that
    > table quotes `_backends.py`'s actual message string, so editing the prose without
    > editing the module would silently desync the document from the code it documents. A
    > remedy that happens to be correct is not a reason to treat the text as free to edit.
- [x] Verify what a consumer installs
  > **Done:** a consumer's install was exercised three ways and all three hold, and the goal
  > found a defect in the artifact it was verifying, which is the outcome that justifies
  > having it. The shipping wheel carries **all four release files** — the README as long
  > description, `dist-info/licenses/LICENSE` at 1077 bytes with a `License-File:` field,
  > the `Typing :: Typed` classifier, and `py.typed` inside the package — plus both
  > extensions, no namespace `__init__.py`, and all ten classifiers including the restored
  > `Programming Language :: Cython`. It resolves `pfsmgraph-dataseq` from PyPI, reports
  > `cython ✓` on all three public calls, and runs its own README's examples. Its kernels are
  > **bit-exact** with the numpy reference: 748 passed / 523 skipped — the same figure CI's
  > Linux wheels produced, and produced a third time against the final artifact — plus an
  > explicit `tobytes()` comparison over a 210-cycle EM run with `max|diff| = 0.0`. The sdist
  > builds to the same result in 15.7s, and `-Dcompiled=false` yields a working pure install
  > that reports `cython ✗` accurately.
  > **Note:** the only thing not verified here is the other three platforms, which this host
  > cannot build. CI checked those on its own hardware and reached the same 748, which is
  > stronger evidence about `-ffp-contract=off` than either run alone: two independent
  > toolchains, one manylinux container and one host compiler, agreeing bit for bit.
  > **Reopened 2026-09-16, same day:** the goal's own verification found a defect in the
  > artifact it was verifying, so the artifacts that were checked are no longer the ones that
  > will ship. The `.so` files, `py.typed`, the namespace invariant and the bit-exactness
  > results are all unaffected by adding a license file, but a verification that does not
  > match what ships is worse than none, so subgoal 1's structural checks are re-run against
  > the rebuilt wheel before this closes again.
  > **Note:** the rebuild after adding `license-files` produced a wheel **still missing the
  > license**, and that is the sharpest possible confirmation of the finding recorded two
  > commits ago. `just build` builds the wheel from the sdist, the sdist's contents are
  > `git archive HEAD`, and the change was uncommitted — so the sdist's `PKG-INFO` carried
  > `License-File: LICENSE` from the working tree while the `pyproject.toml` inside the same
  > archive had **zero** `license-files` lines. **Nothing caught it**, because the divergent
  > field was not the version, which is the only field uv cross-checks between sdist and
  > wheel. The earlier `uv build --wheel` test did produce the license, because it builds
  > from the working tree and bypasses the sdist entirely. Predicted in `docs/ops/release.md`
  > on the same day, then demonstrated by accident within the hour.
  > **Done:** a consumer's install was exercised three ways and all three hold. The platform
  > wheel carries both extensions, `py.typed` and no namespace `__init__.py`, resolves
  > `pfsmgraph-dataseq` from PyPI, reports `cython ✓`, and runs its own README's examples.
  > Its kernels are **bit-exact** with the numpy reference — 748 tests plus an explicit
  > `tobytes()` comparison over a 210-cycle EM run. The sdist builds and installs to the same
  > result in 15.7s on a machine with a compiler, and `-Dcompiled=false` yields a working pure
  > install that says so. The only thing not verified here is the *other three platforms*,
  > which this host cannot build; CI checked those on its own hardware, with the same 748.
  > **Note:** the version was bumped to `0.2.0.dev0` rather than `0.2.0` before building,
  > by the repository's own rule rather than by preference: `core.md` says the `.dev0` suffix
  > stays until the release commit, because a bare version means one accidental publish burns
  > that number on PyPI permanently. PyPI already carries `0.0.0` and `0.1.0` for this
  > distribution, so the tree was, until this bump, declaring an already-published version.
  > **Ran (2026-09-16):** the first `just build` **failed**, and the failure is the finding.
  > `uv` refused with *"The source distribution declares version 0.2.0.dev0, but the wheel
  > declares version 0.1.0"*. Diagnosed precisely: the sdist is internally inconsistent —
  > its `PKG-INFO` says `0.2.0.dev0`, read from the working tree, while the `pyproject.toml`
  > **inside the same archive** says `0.1.0`, because meson-python builds the sdist with
  > `git archive HEAD` and the bump was uncommitted. The wheel is then built from those
  > contents, so it comes out `0.1.0`, and uv catches the pair.
  > **Note:** this sharpens `docs/ops/release.md`, which said an uncommitted change "reaches
  > neither artifact". It reaches the sdist's *metadata* and not its *contents*. For the
  > version field the divergence is loud; **for every other field it is silent** — an
  > uncommitted classifier, bound or description change gives an sdist whose `PKG-INFO` and
  > `pyproject.toml` describe different packages, with nothing to catch it. That is not
  > hypothetical for this branch: goal 4 edited exactly those fields, and had they been built
  > uncommitted the sdist would have advertised the new classifiers while carrying the old.
  > **Note:** so the version bump must be **committed** before `just build` yields consistent
  > artifacts, which is what goal 6's "release commit: bump, relock, rebuild" implies —
  > demonstrated here rather than assumed.
  - [x] Build 0.2.0.dev0 wheels and the sdist; install each available wheel into a clean
    venv outside the workspace: `py.typed`, the `.so` files, no `pfsmgraph/__init__.py`,
    `backends()` reporting `cython ✓`, the README examples
    > **Ran (2026-09-16):** `just build` under `0.2.0.dev0` produced
    > `pfsmgraph_hmm-0.2.0.dev0-cp312-cp312-linux_x86_64.whl` — **platform-tagged**, which is
    > the `-Dcompiled=false` drop working — and the sdist. The wheel's own listing carries
    > both `.so` files, `py.typed`, eleven modules and **no `pfsmgraph/__init__.py`**.
    > **Ran:** installed into a fresh venv under the scratchpad, outside the workspace. uv
    > resolved `pfsmgraph-dataseq==0.1.0` **from PyPI** and numpy 2.5.3 — three packages, no
    > extras. `pfsmgraph.hmm` resolves to that venv's `site-packages`, `pfsmgraph.__path__`
    > is the namespace directory alone, `py.typed` is present, both extensions are present,
    > and `backends()` reports `cython ✓` on all three public calls, with each absent backend
    > naming its extra.
    > **Ran:** the README examples against the *installed* wheel, by running
    > `tests/test_api_docs.py -k "hmm and README"` with the consumer venv's interpreter —
    > 2 passed. The session header is what proves where it ran: `cython ✓ · cpu_parallel ✗
    > (numba is not installed)` cannot appear in a checkout, where numba is in the dev group.
    > **Note:** a scripting error turned into a check worth keeping. `backends("forward_backward")`
    > raises `ValueError: no public call 'forward_backward' has backends`, which is the package
    > being right — forward-backward is a private kernel with no public call, so it exposes no
    > selection — and the refusal is loud rather than an empty list.
  - [x] Check the installed compiled kernels bit-exact against the numpy reference
    (a platform wheel built with contraction on would pass every tolerance test)
    > **Ran (2026-09-16):** the package's own suite against the installed wheel, using
    > cibuildwheel's mechanism — `conftest.py`, `_backends.py`, `pyproject.toml`, the tests
    > and the `.scratch/hmm-lush/Training` fixtures copied into a temp tree, then the consumer
    > venv's pytest. **748 passed, 523 skipped**, which is the *same* figure CI's Linux wheels
    > produced, and it includes `test_forward_backward_backends.py`'s 213 `tobytes()`
    > comparisons.
    > **Ran:** an explicit comparison besides the suite, because a test count is not a
    > demonstration. On a 12-state model over four records of length 61 to 200: `viterbi`
    > states byte-identical and `total_bits` identical as hex floats on every record;
    > `viterbi_batch` identical row by row; and `baum_welch` run to convergence — **210
    > cycles** — giving an identical `description_lengths` tuple and `max|diff| = 0.0` on all
    > three parameter arrays, compared as `tobytes()`. Bit-exact is what a contraction-enabled
    > build fails and every tolerance test passes, so this is the check the `-ffp-contract=off`
    > flag exists for, made against what a consumer installs rather than what meson accepted.
  - [x] Build and install from the sdist alone, on a platform with no wheel
    > **Ran (2026-09-16):** `uv pip install --no-binary pfsmgraph-hmm <sdist>` into a fresh
    > venv, build isolation on, so `build-system.requires` had to supply meson-python, Cython
    > and numpy with no workspace involvement. Built and installed in **15.7s**, and the
    > result is indistinguishable from the wheel: both `.so` files, `py.typed`, no
    > `pfsmgraph/__init__.py`, `cython ✓`. That is the path every platform without a wheel
    > takes — Intel macOS, musl Linux, 32-bit Windows, any glibc older than the manylinux
    > image.
    > **Ran:** the escape hatch the README now promises, since a promise made in an immutable
    > long description should be tested before it ships. `-C setup-args=-Dcompiled=false`
    > installs pure: no `.so` files, `python` backend working, and `cython` reporting
    > *"this install of pfsmgraph-hmm has no compiled extension (a pure wheel, or a build with
    > -Dcompiled=false): install a platform wheel, or build from source with a C compiler"*.
    > That also confirms goal 4's reading empirically — both halves of that remedy are
    > actionable at 0.2.0, where at 0.1.0 the first half was not.
- [~] Release: publish, tag, and close what the branch discharged
  > **Q:** Release from this branch, or merge to `main` first and release from there?
  > **A:** From this branch, then merge — 0.1.0's precedent exactly. Two facts settle it
  > rather than preference. The `pfsmgraph-hmm-v0.1.0` tag points at `75b5c47 "Release
  > pfsmgraph-hmm 0.1.0"`, a commit made on its release branch, which became an ancestor of
  > `main` through the merge. And the dry run already proved GitHub executes `release.yml`
  > from a non-default ref: run `35049292620` ran the full matrix from this branch before
  > that workflow had ever been on `main`, so a tag pushed here will trigger the publish job
  > with no merge required. An earlier worry in this session that the merge had to come first
  > was wrong, and the evidence refuting it was already in hand.
  > **Note:** this makes the **merge strategy load-bearing rather than stylistic**. The
  > tagged commit stays reachable from `main` only under a merge commit; a squash merge
  > rewrites it and leaves the release tag pointing at a commit `main` cannot reach. The
  > merge-commit preference already in use is what keeps 0.1.0's tag honest, and must hold
  > here too.
  > **Q:** How is the Trusted Publisher set up?
  > **A:** Values supplied here, entered by hand at
  > `pypi.org/manage/project/pfsmgraph-hmm/settings/publishing/`; nothing in this session
  > touches PyPI. Read from `release.yml` on 2026-09-16: owner `PanosMavromatis`, repository
  > `pfsmgraph`, workflow name `release.yml`, environment `pypi`. All four are matched
  > against the OIDC claim, and a mismatch fails at upload with the tag already pushed.
  > **Note:** the environment name is the field most likely to go wrong, because PyPI's form
  > makes it optional and `release.yml` does declare `environment: name: pypi` on the publish
  > job. A publisher configured with *no* environment still matches; one configured with a
  > *different* name does not. Nothing else is currently attached to that environment — no
  > required reviewers, no branch restriction — so it is a label today, and the place to add
  > a human gate between tag push and upload later without touching the workflow.
  - [x] **User:** attach a Trusted Publisher for `pfsmgraph-hmm` on PyPI, per goal 1
    > **Done (2026-09-16, by the user):** attached with owner `PanosMavromatis`, repository
    > `pfsmgraph`, workflow `release.yml`, environment `pypi` — the four values read from
    > `.github/workflows/release.yml` and matched against the OIDC claim at upload. The
    > `.envrc` token is not retired by this; it remains the path `publish` still implements
    > for the four members that release locally.
    > **Note:** ordered after the workflow goal, not before it. PyPI's trusted publisher
    > form asks for the workflow *filename* and the environment name, so it cannot be
    > filled in until `release.yml` exists and both are settled. The `.envrc` token is not
    > retired by this — it stays as the fallback path `publish` still implements.
  - [~] Release commit: bump `version` to `0.2.0`, relock, rebuild and re-verify
    > **Ran (2026-09-16):** bumped `0.2.0.dev0` → `0.2.0` and ran `uv lock`. The relock is
    > **not** a formality: it reported *"Updated pfsmgraph-hmm v0.1.0 -> v0.2.0"*, meaning
    > `uv.lock` had been carrying `0.1.0` since commit `58942fe`, days before this branch
    > opened. The `0.2.0.dev0` bump never relocked, so the lockfile and `pyproject.toml`
    > disagreed about the member's own version across several commits — and `uv run pytest`
    > ran clean throughout, so nothing here would ever have surfaced it. This subgoal's
    > "relock" step is the only thing in the release path that catches that drift.
    > **Note:** the rebuild must follow the *commit*, not the bump. `HEAD` still declares
    > `0.2.0.dev0` while the working tree declares `0.2.0`, so `just build` now would take
    > `0.2.0` for the sdist's `PKG-INFO` and `0.2.0.dev0` for its contents from `git archive`,
    > and uv would refuse the pair — the same trap as before, met for the third time and this
    > time anticipated rather than discovered. The subgoal's own wording already put "rebuild"
    > after "release commit"; it is now clear that the ordering is a requirement rather than a
    > description.
    > **Note:** the tree now declares a bare, unpublished `0.2.0`, which is the state
    > `core.md` warns about — one accidental publish burns the number permanently. Both paths
    > are closed by guards this branch added: `just release` refuses `pfsmgraph-hmm` before
    > running anything, and the workflow's publish job is gated on a `pfsmgraph-hmm-v*` tag.
  - [ ] **User:** publish (irreversible): push the `pfsmgraph-hmm-v0.2.0` tag to trigger
    the release workflow's publish job
    > **Note:** the exact command is **`just release-ci 0.2.0`**, run from the repo root.
    > There is no separate tag command — this recipe *is* the tag push. The package argument
    > is omitted because `default_package` is `pfsmgraph-hmm`. In order it runs
    > `_ci-release-refused` (hmm is CI-built), `_version-matches 0.2.0` (pyproject declares
    > it), the full suite, and `_tree-pushed` (refuses a dirty tree, pushes `HEAD`); only
    > then does its body tag and push. Everything checkable is checked before the trigger
    > exists, which is the whole shape of the recipe.
    > **Note:** the raw equivalent, for reference and **not** the recommended path, is what
    > that body runs once the four prerequisites pass:
    > `git tag -a pfsmgraph-hmm-v0.2.0 -m "pfsmgraph-hmm 0.2.0"` then
    > `git push origin pfsmgraph-hmm-v0.2.0`. Running these directly skips every guard above.
    > The workflow's own version assertion would still refuse a tag naming a version the
    > artifacts do not carry — which is precisely why that redundancy exists, since a tag can
    > be pushed with plain `git` and CI cannot assume the recipe was used.
    > **Note:** `_tree-pushed` refuses a dirty working tree, so **this plan file must be
    > committed before `release-ci` will run**. That ordering is not a nuisance: it is why
    > the verification results below are recorded *after* the publish rather than before,
    > where they can state what happened instead of what was expected.
  - [ ] Confirm PyPI's digests match the verified build, the tag `pfsmgraph-hmm-v0.2.0`
    is on `origin`, and `just verify 0.2.0` installs from PyPI
  - [ ] Record commit: the "released" statements in `core.md`, the root README and the
    `docs/api/hmm/` pages, **including three stale claims in the root README that this
    branch found rather than caused**
    > **Note (found 2026-09-16, during a docs sync that correctly declined to fix it):** the
    > root README carries both halves of the rot `DEFERRED.md` catalogues, and they need
    > different treatment. The *mechanical* half is line 60, `uv run pytest # run the suite
    > (709: 74 dataseq, 584 hmm, 51 root)`; measured today the figure is **1418 = 74 + 1292 +
    > 52**, so it is stale by more tests than it claims. Do not copy that number when the
    > time comes — re-measure, because the point of the entry is that figures written once go
    > silently false, and a number transcribed from this note would be the same mistake with
    > a fresher date.
    > **Note:** the *semantic* half is the status paragraph at line 5, and it is the worse of
    > the two because no count-checker could ever find it — which is exactly the limitation
    > `DEFERRED.md` records under "Check documented repo-state counts against the tree",
    > citing "`SymbolTable` is the provisional encoder implementation" as the precedent. It
    > says `pfsmgraph-hmm` "has begun", describes its surface as the numeric helpers plus
    > `HMMParams` and the Viterbi decode, and closes with "the compiled kernels stay in the
    > repository until a public call selects them". By 0.2.0 every clause is false: the
    > package trains, decodes in batches, selects among five backends, and a public call
    > reaches a compiled kernel — that last being the entire premise of this release. Line
    > 71's "released at 0.1.0" is the third, and moves to 0.2.0 with the rest.
    > **Note:** these belong in the record commit rather than earlier because the README is
    > being edited there anyway and the figures are only correct once the release has
    > happened. Fixing them mid-branch would mean writing them twice, the first time wrongly.
