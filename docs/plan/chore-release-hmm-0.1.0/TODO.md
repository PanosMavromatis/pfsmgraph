# chore/release-hmm-0.1.0

**Status**: active
**Created**: 2026-09-13
**Subgoal**: Release `pfsmgraph-hmm` 0.1.0 via `just release 0.1.0 pfsmgraph-hmm` —
`docs/plan/TODO.md`, revision 02-hmm-v0.1.0

## Goals

- [x] Remove the unimported `pfsmgraph-align` dependency and record why it returns later
  > **Q:** `DEFERRED.md` already carries the alignment-seeded revision under "`align` able to produce a multiple alignment", and gives it a trigger rather than a number on purpose. Add the dependency item under that trigger, or under a new heading naming 0.4.0?
  > **A:** Under the existing trigger, noting there that the revision is expected to be `hmm` 0.4.0, the first after 0.3.0's merge/split search.
  > **Q:** `hseg` and `dl` declare the same unimported `pfsmgraph-align>=0.1`. How far does ADR 0019's rule reach now?
  > **A:** The rule is family-wide and checked at each member's release commit, when metadata becomes immutable; only `hmm`'s bound is removed on this branch.
  > **Q:** ADR 0019's Open section: when the edge returns, is it a hard dependency or an extra?
  > **A:** Most likely an extra in 0.4.0; by the first major release (1.x) alignment will probably be so integral to HMM training that it is required. Recorded as the expected answer in ADR 0019's Open section and in the `DEFERRED.md` entry, not as a decision.
  - [x] ADR 0019 amending ADR 0009: a declared intra-family dependency lands with its first import; `hmm` may release before `align`
    > **Note:** Written as `0019-declared-dependencies-follow-imports.md`. Its Open section raises a question the deferral would otherwise settle by default: whether the edge returns at 0.4.0 as a hard dependency or as an extra, since the alignment seed replaces merge/split's starting point rather than the search.
  - [x] ADR 0009's Status pointer, the ADR index row, and the PRD rows that state the edge
    > **Note:** PRD annotated at §1.5, §3.3, §3.4 and §11 as dated qualifications, in the form §8 already uses. §5.5's example `pyproject.toml` is deliberately untouched: it shows how to declare an edge, not which edges exist.
  - [x] `DEFERRED.md` entry, with a trigger at the alignment-seeded-topology version (expected 0.4.0)
  - [x] Drop the bound and its `[tool.uv.sources]` entry, relock, and correct `docs/api/hmm/README.md` and `core.md`
    > **Note:** `uv lock` rewrote exactly two lines, `hmm`'s `dependencies` and `requires-dist` entries for `pfsmgraph-align`, so the lockfile sees an edge vanish even though it never sees a bound change (measured 2026-09-01). Suite green at 318. The pyproject carries a comment naming ADR 0019 so the PRD §5.5 pattern is not restored by copying. Root `README.md` line 70 also stated the drawn release order and was corrected, and `core.md`'s "seventeen records" was already one stale and now reads nineteen.
- [x] Commit the `justfile` default and sweep the documents that name the old one
  > **Done:** The default and the policy-worded doc sweep landed in `5fd2795`. The release runs from this Linux host as a pure `py3-none-any` wheel, with a per-package token from `.envrc` read by an OS-branching `token`, and a latent `publish` fallthrough fixed. The pure-wheel build became a goal-4 subgoal, and the user's token setup became goal-5 subgoals.
  > **Q:** The default now moves with each release. Should the docs name `pfsmgraph-hmm`, or state the policy?
  > **A:** State the policy: `default_package` names the member under development, so an omitted argument can never reach a published package. No package name appears, so nothing goes stale at the next release.
  > **Q:** `justfile` line 7 says the bare `just build` "builds pfsmgraph-dataseq", which the edit made false. Change it?
  > **A:** Yes, to "builds the default package". Line 172's `dataseq` illustration of on-disk names stays.
  - [x] Verify the hand edit, and correct `docs/ops/release.md`, `core.md` and `README.md` where they state the default
    > **Note:** The sweep found more than the three known sites: `release.md` §2's `token-set` example and §4's tag line both described the old default, and `justfile` line 7's comment stated the bare command's result. `default_package` now carries a rationale comment. Checked against the recipe: `preflight` does not test for `.dev0` as such, but matches the requested version against the built filename, so an omitted argument on a `.dev0` member stops before `publish`; the wording says that rather than "refuses `.dev0`". `just --list` parses, the default evaluates to `pfsmgraph-hmm`, and `test_release_runbook.py` passes.
  - [x] Settle where the release runs: the token recipes call macOS `security`, which this Linux host lacks
    > **Note:** Building the wheel showed the host question was downstream of a larger one. `uv build --wheel --package pfsmgraph-hmm` produces `pfsmgraph_hmm-0.1.0.dev0-cp312-cp312-linux_x86_64.whl`, because `_viterbi_cython.pyx` has made the package compiled since 2026-09-09. That breaks the release on any host: `preflight` tests for `…-py3-none-any.whl`, and PyPI rejects a plain `linux_x86_64` tag (it wants manylinux). The release path was written for `dataseq`'s single universal wheel and was never revisited when the first `.pyx` landed.
    > **Q:** How should 0.1.0 be built and published: a pure wheel without the extension, cibuildwheel in CI with Trusted Publishing, the sdist plus one Mac wheel, or the sdist only?
    > **A:** A pure `py3-none-any` wheel for 0.1.0, built without the Cython extension. No public call in 0.1.0 reaches a compiled kernel, so nothing is lost, and the existing recipes keep working. Compiled wheels wait for a version whose public API uses one.
    > **Note:** Feasibility, measured on a scratch copy: removing the `.pyx` alone still gave `cp312-cp312-linux_x86_64`, because `meson.build` calls `find_installation(pure: false)`, which routes every file to `platlib`. With the `.pyx` removed **and** `pure: true`, the wheel is `pfsmgraph_hmm-0.1.0.dev0-py3-none-any.whl` with the six modules. So the release build needs a meson option that controls both, passed by the build recipe. That is goal 4's work.
    > **Q:** Where does `just release 0.1.0` run?
    > **A:** This Linux host. The user generates a PyPI token scoped to `pfsmgraph-hmm` and adds it to `.envrc`.
    > **Q:** How do the recipes consume the `.envrc` token?
    > **A:** `.envrc` exports a per-package `PYPI_TOKEN_PFSMGRAPH_HMM`. On Linux `token` prints `$PYPI_TOKEN_<PKG>` and fails loudly when it is empty, while macOS keeps the Keychain. `publish` is unchanged. The runbook's "never in `.env` … or the repo tree" rule is rewritten to allow a gitignored `.envrc` on Linux, with the reasons.
    > **Note:** Implemented and verified against fake and stripped variables, never the live one. `token`/`token-test` route through `_token-read`, which uses the Keychain on macOS and `${!var}` indirect expansion of `PYPI_TOKEN_<PKG>` on Linux. An `hmm` token does not satisfy `just token pfsmgraph-align`, and `token-set` on Linux explains itself and exits. `.envrc` was already gitignored (`.gitignore:152`). **A latent `publish` defect was found while wiring this**: `UV_PUBLISH_TOKEN="$(just token pkg)" uv publish` still ran `uv publish` with an empty token when the read failed, and exited 0 in a scratch reproduction, because a failure inside `$(...)` does not stop the command it prefixes. It affected macOS too and never showed because the Keychain entry always existed. It is now `tok="$(…)" && UV_PUBLISH_TOKEN="$tok" uv publish`. `docs/ops/release.md` §2 and the recipe table are updated.
- [x] Settle what 0.1.0's immutable metadata says
  > **Q:** Declare the `gpu` and `cpu-parallel` extras in 0.1.0, when no public call reaches an accelerated kernel?
  > **A:** Drop both for 0.1.0. They return in the version whose public API can select a backend (a `DEFERRED.md` trigger). The modules still ship, unreachable, and the dev group keeps numba and numba-cuda. Removing an extra later strands pinned installs, whereas adding one breaks nothing.
  > **Q:** What does `description` say, when it currently claims Baum-Welch and topology search?
  > **A:** What 0.1.0 contains, declaring Baum-Welch and topology search as forthcoming.
  > **Q:** Keep `Programming Language :: Cython` on a wheel with no compiled code?
  > **A:** Drop it for 0.1.0. It returns with the first compiled wheel.
  - [x] Whether to ship the `gpu` / `cpu-parallel` extras when no public call reaches an accelerated kernel
    > **Note:** Removed from the pyproject, which now carries a pointer comment instead. The exact rows and every measurement that shaped them (`numba`'s 2.2.6 numpy ceiling, the `numpy<2.5` cap for `np.row_stack`, the bare `numba-cuda` without a toolkit extra) moved to a restore entry under `DEFERRED.md`'s "`align` acquiring a backend-selection API". `uv lock` dropped only `hmm`'s extras entries and still resolves 57 packages, because the dev group keeps both compilers. Suite green at 318 with `python ✓ · cython ✓ · cpu_parallel ✓ · cuda ✓`. Repo-local wording that said "declined the extra" (`core.md`, `_backends.py`, `tests/test_backends.py`, root `pyproject.toml`) now says numba is not a dependency. The legitimate-skip reasoning is unchanged, only its premise is stronger. `dp-compile.toml` keeps its `extra =` keys, so phase skills do not read the absence as a hard dependency.
  - [x] `description` claims Baum-Welch and topology search, neither of which 0.1.0 contains
    > **Note:** Now "Arc-emission hidden Markov models for the pfsmgraph family: frozen parameters and Viterbi decoding; Baum-Welch training and topology search forthcoming." `Programming Language :: Cython` is removed, since the 0.1.0 wheel is pure, and the header comment is corrected.
  - [x] Honest lower bounds on every intra-family dependency naming `pfsmgraph-hmm`
    > **Note:** Nothing to change. No member declares `pfsmgraph-hmm` (checked across `packages/*/pyproject.toml`), and `hmm`'s own intra-family bound, `pfsmgraph-dataseq>=0.1.0`, is satisfied by the release on PyPI and names the version that introduced every `dataseq` API `hmm` uses.
- [ ] Ship the four files and verify the first meson-built wheel
  - [ ] Member README (executed by `test_api_docs.py`), LICENSE copy, `Typing :: Typed`, `py.typed` in `install_sources`
  - [ ] Build 0.1.0 as a pure `py3-none-any` wheel: a meson option that skips `_viterbi_cython` **and** switches `find_installation` to `pure: true`, passed by the `build` recipe for `hmm`, with the editable dev build still compiling the extension so the suite keeps its `cython` row
  - [ ] Drop `.dev0`, build, and install the wheel into a clean venv outside the workspace
  - [ ] Re-measure `docs/ops/release.md`'s hatchling-era reproducibility findings
- [ ] Release: publish, tag, and close what the branch discharged
  - [ ] **User:** create the project-scoped token per `docs/ops/release.md` §2 (2FA confirmed; name `pfsmgraph-hmm-release-2026-09`; scope `Project: pfsmgraph-hmm`; copied once)
  - [ ] **User:** add `export PYPI_TOKEN_PFSMGRAPH_HMM='pypi-…'` to the repo-root `.envrc` (installing direnv first if it is absent and hooking it into the shell), then run `direnv allow`
  - [ ] **User:** confirm the recipes see it without printing it: `just token >/dev/null && echo ok`
  - [ ] Publish (the user runs it; irreversible) and tag `pfsmgraph-hmm-v0.1.0`
