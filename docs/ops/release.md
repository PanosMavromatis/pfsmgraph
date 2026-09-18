# Releasing a member to PyPI

How any one of the five `pfsmgraph` distributions gets from the workspace onto PyPI. The
procedure is the same for every member; what differs per package is the keychain entry name
and the import path, and both are parameters of the recipes in the repo-root `justfile`.

Written 2026-09-02 for the `pfsmgraph-dataseq` 0.1.0 release, generalised on the same day.

## The name appears in three forms

| Form | Where it appears | Example |
| --- | --- | --- |
| Hyphenated | the index, `uv build --package`, `--check-url` | `pfsmgraph-dataseq` |
| Underscored | filenames in `dist/`, every glob | `pfsmgraph_dataseq-*` |
| **Dotted** | **the import path** | **`pfsmgraph.dataseq`** |

Every glob, URL and `import` below is one of the three. When editing a recipe, check which
one you are in -- the recipes derive all three from the hyphenated name with `replace()`,
which is the reason the package is a parameter rather than five copies of a file.

The dotted form is a PEP 420 implicit namespace package shared by all five members. That
sharing is what makes the wheel-shape check in the pre-flight non-negotiable.

## Release order is forced

Release order follows the dependency graph, not the implementation order: a package cannot
publish before its dependencies exist on PyPI. `dataseq` is the leaf with no intra-family
dependencies, so it goes first; the other four depend on it and must follow. See
[`core.md`](../agents/core.md) under "Architecture".

---

## 0. Pre-flight

### The four files a release ships that the version bump does not imply

A member's release commit must contain all four, inside `packages/pfsmgraph-<pkg>/`:

- `README.md` -- the PyPI long description. The root README is about the workspace and
  every relative link in it 404s there.
- `LICENSE` -- a **real copy**, never a symlink to the repo-root one, **and declared as
  `license-files = ["LICENSE"]`**, without which meson-python ships no license text.
- the `Typing :: Typed` classifier.
- `src/pfsmgraph/<pkg>/py.typed` -- the PEP 561 marker, **inside** the importable package.

**Three** of these fail silently if placed or declared wrong, which is why
[`core.md`](../agents/core.md) carries them as an invariant rather than a checklist:

- A symlinked `LICENSE` builds a valid-looking sdist and then fails on **unpack** -- a
  symlink escaping the sdist root is refused. That is a consumer-side failure, caught here
  only because `uv build` routes wheel-building through the sdist.
- An **undeclared** `LICENSE` reaches no wheel at all. meson-python includes no license
  file unless `license-files` names it, and the omission is invisible from outside the
  archive: METADATA still carries `License-Expression: MIT`, which is valid PEP 639, so
  `twine check` passes and PyPI renders the license correctly. Measured 2026-09-16 --
  `pfsmgraph-hmm` 0.1.0 is on PyPI with no license text, and nobody noticed because
  `pfsmgraph-dataseq` 0.1.0 *does* carry `dist-info/licenses/`, hatchling having globbed
  `LICENSE*` by default. The rule predates this repository's move to meson-python and was
  inherited into a backend with different defaults.
- A `py.typed` at the distribution root instead of inside the importable package reaches no
  wheel at all, with no error and no warning, and a type checker then discards every
  annotation in the package. It cannot go at the `pfsmgraph/` namespace level either, for
  the same reason no `__init__.py` may: no single distribution owns that level.

`pfsmgraph-dataseq` has all four as of the 0.1.0 release commit, and `pfsmgraph-hmm` as of
its 0.1.0 release branch. Each remaining member owes them in its own release commit; see
`docs/plan/DEFERRED.md`.

**A member's release notes are its `CHANGELOG.md`, beside its README**, since
`pfsmgraph-hmm` 0.3.0 (2026-09-18), whose notes ADR 0025 §7 required. It is not a fifth
invariant file, because nothing about it fails silently: it reaches the sdist as every
tracked file at the member root does, reaches no wheel, and is linked from the README and
from `[project.urls]` as `Changelog`. Its newest heading reads `unreleased` until the
release commit dates it, for the same reason the version keeps `.dev0` until then. It holds
no fenced code blocks, since `tests/test_api_docs.py` executes only READMEs and
`docs/api/`. The other members start theirs at their first release.

**`just build` builds the last commit, not the working tree.** meson-python makes the sdist
with `meson dist`, which archives `HEAD` even under `--allow-dirty`, and `uv build` then
builds the wheel from that sdist. So an uncommitted change reaches neither artifact, and a
build run to check a change before committing it checks the previous commit instead.
Measured 2026-09-13, when a first `just build` on `pfsmgraph-hmm` produced the pre-change
wheel with none of the new metadata. At release this is harmless, because `preflight`
refuses a dirty tree. To inspect uncommitted work, build the wheel directly with
`uv build --package <pkg> --wheel --out-dir <scratch>`.

**Sharpened 2026-09-16: "reaches neither artifact" is not quite right, and the exception is
the part worth knowing.** An uncommitted change reaches the sdist's *metadata* and not its
*contents*. Measured while bumping `pfsmgraph-hmm` to `0.2.0.dev0` for the goal-5
verification: the sdist's `PKG-INFO` read `Version: 0.2.0.dev0`, taken from the working
tree, while the `pyproject.toml` **inside the same archive** read `0.1.0`, taken from
`git archive HEAD`. For the **version** field that divergence is caught loudly, because
`uv build` builds the wheel from the sdist's contents and then refuses the pair -- *"The
source distribution declares version 0.2.0.dev0, but the wheel declares version 0.1.0"*.
For **every other field it is silent**: an uncommitted classifier, dependency bound or
description change yields an sdist advertising one thing in `PKG-INFO` and carrying another
in `pyproject.toml`, with nothing to catch it. The practical rule does not change and now
has a reason behind it -- commit before building.

**`pfsmgraph-hmm` 0.1.0 was built with `-C setup-args=-Dcompiled=false`, and 0.2.0 is not.**
The option (`packages/pfsmgraph-hmm/meson.options`) skips the Cython kernels and installs to
purelib, giving a `py3-none-any` wheel; both halves are needed, since skipping the extension
alone still left a platform-tagged wheel. That was right for 0.1.0, where no public call
reached a compiled kernel. From 0.2.0 `backend="cython"` is public, so a pure wheel would
mean every pip user seeing `cython ✗`, and the `build` recipe passes the flag no longer.

**The option stays, as a source-build escape hatch rather than a release setting.** It is the
only way to install this package from the sdist on a machine with no C compiler, accepting a
pure install that reports `cython ✗` honestly rather than failing to build at all. Nothing in
the release path sets it, and nothing should: a release wheel is exactly the case that must
not use it.

**`build` is no longer the first step of `pfsmgraph-hmm`'s release**, and that changes what it
is for rather than retiring it. It produces one local platform wheel, useful for installing
into a clean venv to check what a consumer gets. The wheels that ship are built by GitHub
Actions -- see *Two release paths* below.

### Validate

```bash
just test          # the suite, including the ADR 0013 doc verifier
just build         # clean, then uv build --package
just check         # twine check -- catches long-description rendering failures
```

`just release` runs all three as prerequisites, so running them by hand is for iterating,
not for the release itself.

This document is itself checked, though narrowly: `tests/test_release_runbook.py` asserts
that every `just <recipe>` named below exists in the `justfile`. It is deliberately **not**
covered by the ADR 0013 verifier -- see that test's docstring for where the boundary sits
and why moving it would weaken ADR 0013 rather than strengthen this file.

Then confirm the wheel's shape:

```bash
unzip -l dist/pfsmgraph_dataseq-0.1.0-py3-none-any.whl
```

Expect `pfsmgraph/<pkg>/*`, a `py.typed` beside the modules, and the `dist-info/`. Expect
**no** `pfsmgraph/__init__.py` -- that file would claim the shared namespace and make the
other four members unimportable alongside this one. The bug never appears in the workspace,
only for external installers.

**A file listing shows what went into the box, not what a consumer gets out.** For a first
release of a member, install the built wheel into a clean venv outside the workspace and
import from it. That is the acceptance test; the listing is a sanity check.

### Rebuilding does not invalidate that verification

`just release` runs `clean` first, so it rebuilds rather than uploading whatever is already
in `dist/`. That would be a problem if the verified artifact and the rebuilt one could
differ. Measured 2026-09-13 on `pfsmgraph-hmm` 0.1.0.dev0, the first meson-python build of
a member:

- **The sdist is byte-identical across builds.** meson-python makes it with `meson dist`,
  a `git archive` of `HEAD`, so it carries commit times and tracked files only. That is
  also why it no longer ships the repo-root `.gitignore`, which hatchling used to add, and
  why it cannot see uncommitted work (above). It does ship the member's `tests/`, which
  read fixtures from the repository's `.scratch/` and so do not run from an unpacked
  sdist. That is inert for consumers, and noted here so a failing sdist test run is not
  mistaken for a broken package.
- **The wheel is byte-identical only with `SOURCE_DATE_EPOCH` set.** Without it,
  meson-python stamps build-time entry timestamps, and two builds of one commit differ in
  those alone: the unpacked contents are identical. The `build` recipe therefore exports
  `SOURCE_DATE_EPOCH` as the commit time, and two `just build` runs then give the same
  SHA-256 for both artifacts.

The hatchling-era findings this replaces (2026-09-02, `pfsmgraph-dataseq` 0.1.0) were the
mirror image: a reproducible wheel, because hatchling normalises timestamps, and a
non-reproducible sdist, because it walked up to the VCS root and shipped the root
`.gitignore`. Both were properties of the builder rather than of this project, which is why
[ADR 0018](../design/adr/0018-family-wide-meson-python-build-backend.md) listed the
re-measurement among its costs. The four-file invariant was re-verified against a built
wheel installed in a clean venv outside the workspace at the same time.

**So prefer `just release` over hand-publishing the artifacts already in `dist/`.** The
alternative -- `just check && just publish` plus a manual tag -- preserves bytes that are
already provably reproducible for the wheel, and pays for it by skipping the entire
`preflight` chain and the test run. That is a bad trade.

### Two rules about version numbers

- **A version can never be reused on PyPI**, even after deletion. Burning `0.1.0` on a bad
  build costs the number permanently. This is why the `.dev0` suffix stays on a member's
  version until its release commit.
- **Deleting is not rollback.** Anything already resolved against it breaks. Use **yank**
  (Manage project -> Releases -> ... -> Yank) -- it hides the version from resolvers while
  leaving existing pins working.

---

## 1. The `justfile`

### Two release paths, and which member takes which

**This repository has two release mechanisms, deliberately.** Which one a member takes
follows from whether this machine can build its shipped artifact:

| | Members | Command | Upload |
| --- | --- | --- | --- |
| **Local** | those shipping a pure `py3-none-any` wheel | `just release <version> <pkg>` | project-scoped token, from here |
| **CI** | those in the `ci_built_packages` variable -- today `pfsmgraph-hmm` | `just release-ci <version> <pkg>` | Trusted Publishing, from the workflow run the tag triggers |

A member moves from the first row to the second when its first public call reaches a compiled
kernel, because from that point its release is platform wheels for machines this one is not.
`pfsmgraph-hmm` crossed that line at 0.2.0.

**The two are not interchangeable, and `just` refuses the wrong one in both directions.**
`release` stops for a member CI builds; `release-ci` stops for a member CI does not, since
that would push a tag no workflow matches and report success having released nothing. Both
refusals are *prerequisites* rather than body lines, for the reason given below.

**Trusted Publishing cannot be driven from here at all.** It is OIDC: the token is minted by
GitHub Actions for a specific repository, workflow and environment, and there is no local
equivalent. So "download the CI wheels and publish them with `just`" is not a shortcut to the
same outcome -- it is the token path, without PEP 740 attestations, which are signed against
the Trusted Publishing identity. The artifacts *are* downloadable (`gh run download <run-id>`
fetches all of them in seconds); the reason not to is the identity, not access.

`just` is a command runner: named recipes, arguments, no build graph and no `make` tab
traps. The `justfile` sits at the workspace root beside the root `pyproject.toml`, because
the recipes assume that working directory -- `uv build --package` and the shared `dist/`
both resolve from there. It is tracked, so the release procedure is version-controlled
rather than remembered.

```bash
brew install just
just              # list all recipes
```

| Recipe | Purpose |
| --- | --- |
| `just test` | The full suite |
| `just build [pkg]` | `clean`, then `uv build --package` |
| `just check [pkg]` | `twine check` on the built artifacts |
| `just token-set [pkg]` | Store a PyPI token in the macOS Keychain (prompts, no echo); on Linux, use `.envrc` (§2) |
| `just token [pkg]` | Read it back; fails loudly if absent |
| `just preflight VER [pkg]` | Assert the version was built, the tree is clean, and push |
| `just publish [pkg]` | Upload to PyPI |
| `just verify VER [pkg]` | Install from PyPI in a throwaway env and import |
| `just release VER [pkg]` | test -> build -> check -> preflight -> publish -> tag |
| `*-test` variants | Same, against TestPyPI |

Every recipe takes an optional package name defaulting to `default_package`, which names
the member under development rather than a published one. An omitted argument therefore
builds a `.dev0` version that cannot match the requested one, so `preflight` stops the
release before `publish`, instead of acting on a member already on PyPI. One file
serves all five members: `just release 0.1.0 pfsmgraph-align`.

### Four design points worth knowing before you edit it

**Prerequisites run first, left to right, and stop at the first failure.** `release`
declares `test (build) (check) (preflight) (publish)`, so a failing test or a failing
`twine check` aborts before anything reaches PyPI, and the body -- which only tags -- runs
only after a successful upload.

**A check must be a prerequisite, not a body line.** Every body line runs after *every*
prerequisite, including `publish`. A guard written in the body of `release` would therefore
run after the irreversible step, which is why `preflight` is its own recipe positioned to
the left of `publish`.

**`clean` guards the glob, not just staleness.** `dist/` is one directory shared by all five
members, and the publish glob is scoped by package name but carries no version. `clean` is
what stops that glob from picking up a stale version of the same package.

**`publish` is the switching point.** Moving a package to Trusted Publishing means editing
that recipe body and nothing else -- `build`, `check`, `preflight`, `verify` and the tag
convention are untouched. That is the entire reason for routing through a recipe instead of
typing `uv publish` directly.

### What `preflight` asserts, and why

The version argument to `release` is otherwise inert: `build` reads the version from
`pyproject.toml`, and the publish glob carries no version at all. Without an assertion,
`just release 0.2.0` would build 0.1.0, publish 0.1.0, and push a tag `...-v0.2.0` pointing
at a commit that declares 0.1.0. Both halves are irreversible, and both exit 0.

`preflight` therefore checks that the requested version is the one sitting in `dist/`, that
the working tree is committed (a tag on a dirty tree names a state that exists nowhere
else), and pushes `HEAD` so the published commit exists on the remote before the upload.

**Do not run it standalone.** The version check matches the *filename* in `dist/`, so it
cannot distinguish a current artifact from a stale one carrying the same version number:
against an old build it passes, and reads as verification. Inside `release` the gap does not
exist, since `clean` and `build` run first and it only ever sees an artifact built moments
earlier. Observed 2026-09-02, when it passed against a wheel whose metadata three commits
had since changed.

**Its `py3-none-any` assertion was re-examined for 0.2.0 and deliberately not relaxed.** The
obvious reading of platform wheels is that `preflight` must learn to accept platform tags.
It must not. No member publishes platform wheels *from here* -- those go out from CI -- so the
assertion stays correct for every member that still takes the local path, and relaxing it
would delete the last check standing between a local `pfsmgraph-hmm` release and an upload,
leaving PyPI's refusal of a bare `linux_x86_64` tag as the only backstop. The explicit
refusal in `release` is the policy; this filename check is a second line behind it.

What did move is the half of `preflight` that is about the tree rather than the artifacts --
the dirty-tree check and the `git push origin HEAD` -- into `_tree-pushed`, so `release-ci`
can require it without a `dist/` check it has no artifacts to satisfy.

---

## 2. Create the project-scoped token

1. Confirm 2FA at https://pypi.org/manage/account/ -- token creation is gated behind it.
2. https://pypi.org/manage/account/token/
3. **Name:** `pfsmgraph-<pkg>-release-YYYY-MM`. Date-stamped names make rotation auditable.
4. **Scope:** `Project: pfsmgraph-<pkg>`. Not "Entire account". A project-scoped token turns
   a stray glob into a 403 rather than a burnt version number on another package.
5. Copy it -- **displayed exactly once**.

A project scope is available for any name already reserved under the account, which is every
member: all five hold `0.0.0` placeholders. There is no bootstrap-token step.

Where it is stored depends on the host, and the recipes choose by `uname`. Never put it in a
tracked file, `~/.zshrc` or `~/.pypirc`.

**On Linux, export it from the repo-root `.envrc`** (direnv), under a per-package name:

```bash
# .envrc -- gitignored (.gitignore line 152); run `direnv allow` after editing
export PYPI_TOKEN_PFSMGRAPH_HMM='pypi-…'
```

`token` reads `PYPI_TOKEN_<PKG>` (hyphens to underscores, uppercased; `TESTPYPI_TOKEN_<PKG>`
for TestPyPI) and fails naming the variable when it is unset or empty. `token-set` does not
apply on Linux and says so. The file is kept out of git by the ignore rule, and out of the
sdist because meson-python builds the sdist from `git archive`, which carries tracked files
only. **The cost is that direnv exports the token to every process started in the repo**,
not only to `publish`, which is why the name is per-package and only `token` reads it: a
token scoped to one member cannot reach another member's upload, which finds no variable and
stops. Adopted 2026-09-13 for `pfsmgraph-hmm` 0.1.0, the first release from a Linux host.

**On macOS, store it in the Keychain:**

```bash
just token-set                          # stores it for the default package
just token-set pfsmgraph-align          # any other member, by name
```

Both forms are shown because every recipe here takes the package as an optional argument
defaulting to `default_package` at the top of the `justfile`. Naming the default explicitly
is therefore identical to the bare form, not a different operation.

That prompts without echoing, so nothing lands in zsh history. `uv` does not read
`.pypirc` at all -- verified against the binary -- so a token placed there would appear to
be configured and do nothing.

**Rotation reuses the same command.** `token-set` passes `-U`, so it overwrites an existing
Keychain entry rather than refusing; revoke the old token on PyPI and re-run it. Without
`-U` -- how it shipped on 2026-09-02 -- `security` errors with `The specified item already
exists in the keychain` and the entry has to be deleted by hand first.

**`token-set` reads the token into the shell and verifies the round-trip, and both are
load-bearing.** `security add-generic-password`'s own interactive `-w` prompt **silently
truncates its input at 128 characters** -- measured 2026-09-02: a 200-character value went
in and 128 came back, with no error, no warning, and a non-zero exit nowhere. PyPI tokens
are longer than that, so the first release attempt stored a well-formed *prefix* of a valid
token and PyPI answered:

```
Server returned status code 403 Forbidden. Server says: 403 Invalid or non-existent
authentication information.
```

which names the credential and says nothing about where it was damaged. The recipe now
prompts with the shell's own `read -rs` and, after storing, reads the value back and
compares it to what was typed; a mismatch deletes the entry rather than leaving a broken
credential in place, and reports how many characters of how many survived. It also rejects
an empty value and one not beginning with `pypi-` (both registries' tokens do).

This is the same failure shape as the symlinked `LICENSE` and the misplaced `py.typed`: a
step that succeeds, emits nothing, and yields a broken artifact. The round-trip check is
the cheapest general guard against it -- it does not care *why* the value changed.

**Never run `just token` on its own.** It exists to be consumed by `$(...)` inside
`publish`, and standalone it prints a live credential into your scrollback. If it happens,
revoke the token and re-store it.

**`publish` captures the token before uploading, and the order is load-bearing.** Until
2026-09-13 it read `UV_PUBLISH_TOKEN="$(just token pkg)" uv publish …`. A failure inside
`$(...)` does not stop the command it prefixes, so a missing token printed its error and
then ran `uv publish` with an empty token, which falls through to trusted-publishing
discovery and fails as an OIDC error, exit status and all pointing away from the cause. The
recipe now runs `tok="$(just token pkg)" && UV_PUBLISH_TOKEN="$tok" uv publish …`. The
defect was on the macOS path too, and never showed because the Keychain entry always
existed.

The `-a`/`-s` pair the recipes pass to `security` is an arbitrary lookup key -- `$USER`
namespaces the entry to your local login, and the service string is a name this `justfile`
invents. Neither is a PyPI identity. `uv publish` supplies the literal `__token__` username
itself whenever it is handed a token rather than a username/password pair, so your PyPI
account name never enters the flow.

---

## 3. Optional: TestPyPI rehearsal

Separate registry, separate account, separate token. Worth doing once for the first release
of a package family; skippable for routine bumps.

```bash
just token-set-test        # after registering at https://test.pypi.org/
just release-test 0.1.0
just verify-test 0.1.0
```

`verify-test` passes `--index-strategy unsafe-best-match` because real dependencies --
`numpy` for `dataseq` -- live on PyPI proper, not TestPyPI.

> This consumes `0.1.0` **on TestPyPI**. To iterate, rehearse with `0.1.0.dev1`.

---

## 4. Publish

```bash
just release 0.1.0
```

Which is: test -> clean -> build -> `twine check` -> preflight -> upload -> tag
`pfsmgraph-<pkg>-v0.1.0` for the default package -> push.

Two mechanics inside `publish` worth understanding:

- `UV_PUBLISH_TOKEN` is exactly equivalent to `--username __token__ --password <token>`.
  There is no separate username.
- `--check-url` makes the upload **idempotent**: a file already present with an identical
  hash is skipped rather than failing the run. This matters when the wheel uploads and the
  sdist errors -- without it, the retry fails wholesale on "file already exists".

The package-prefixed tag is deliberate, and it is a project invariant rather than a
preference: release order forces the five members onto different versions, so a bare
`v0.1.0` goes ambiguous the moment a second member ships its own. `/smart-commit` can only
form `v<VERSION>` from a root `VERSION` file, which is one of the two reasons no such file
exists here -- see [`claude.md`](../agents/claude.md).

**The tag lands on the release branch, so the merge strategy matters.** `preflight` pushes
`HEAD`, and the tag is created on that commit. A `--merge` or `--rebase` merge keeps it
reachable from `main`; a **squash** merge does not -- the tagged commit survives only
because the tag itself pins it, and `git log main` will never show the commit the release
was cut from. Merge a release branch with `--merge`, or tag after the merge instead.

---

## 5. Verify

```bash
just verify 0.1.0
```

Pinning `==0.1.0` proves you're exercising the new release, not a cache. The recipe imports
`pfsmgraph.<pkg>` and reads the version via `importlib.metadata`, which reads installed
distribution metadata directly and doesn't depend on a `__version__` attribute.

Then eyeball https://pypi.org/project/pfsmgraph-dataseq/ -- README renders, project links
correct, both files listed.

Fastly can take a minute to serve a new version. **If it 404s immediately, wait and retry --
do not re-upload.**

---

## Token sprawl vs. Trusted Publishing

Five members means five long-lived bearer secrets on one laptop, none with expiry or
rotation reminders. **Trusted Publishing** (OIDC from GitHub Actions) removes the secret.
Configure per project under Settings -> Publishing. PyPI also supports **pending
publishers** for names that don't exist yet -- though every member here already holds a
`0.0.0` placeholder, so that path is moot for this family.

| | Project-scoped token | Trusted Publishing |
| --- | --- | --- |
| Local `just publish` | Works | No -- CI only |
| Long-lived secret on disk | One per package | None |
| PEP 740 attestations | Not in this configuration | Yes |
| Setup | ~2 min per package | ~20 min once, ~2 min each after |
| Failure mode | Token leak -> silent malicious release | Misconfig -> loud CI failure |

**On the attestations row: `uv publish` does support them.** It exposes `--no-attestations`
/ `UV_PUBLISH_NO_ATTESTATIONS` as an *opt-out*, so uploading them is the default path, and
any claim that uv cannot generate them is wrong. The constraint is upstream of uv --
attestations are signed against a Trusted Publishing identity, so a token-authenticated
upload has nothing to sign with. Confirm against uv's current docs before relying on this
row either way; it was the one row in the original draft that was stated backwards.

**Suggested posture:** ship with the project-scoped token, which is ready and frictionless.
Pilot Trusted Publishing on a member where a botched release costs nothing, and if it holds,
move the rest and revoke the tokens. The `justfile` is what makes that a one-recipe edit
rather than a habit rewrite.

**That pilot is happening, and it was not chosen for the reason above.** `pfsmgraph-hmm`
0.2.0 moves to Trusted Publishing because its wheels are built by GitHub Actions and the
upload therefore happens there, not because hmm was picked as a low-stakes subject -- it is
the most-developed member in the family. The paragraph's conclusion survives its reasoning:
the posture is still token-first, and the other four members are unaffected.

**One prediction it made is worth marking as half-wrong.** "The `justfile` is what makes that
a one-recipe edit" holds for the *upload* -- `publish` is untouched by any of this -- and not
for the release. `release` has `build` as a prerequisite and `build` runs `clean`, so a recipe
chain ending in `publish` cannot consume wheels an external job produced; the tag flow is a
second path beside it rather than an edit to it.

---

## Incident reference

| Situation | Action |
| --- | --- |
| Bad version already uploaded | **Yank** it, ship the next patch. Do not delete. |
| Token leaked | Revoke at https://pypi.org/manage/account/ immediately, audit release history. |
| Upload fails halfway | Re-run; `--check-url` skips what's already there. |
| `400 File already exists` | Version is spent. Bump and rebuild. |
| Wheel contains `pfsmgraph/__init__.py` | Namespace collision with siblings. Fix the build, bump, re-release. |
| Wheel contains no `py.typed` | Marker is at the distribution root, not inside the package. Under meson-python, also check `meson.build`'s `install_sources` names it — meson does not glob, and `py.typed` is not a `.py` file. Bump, re-release. |
| sdist fails on unpack | `LICENSE` is a symlink. Replace with a real copy, bump, re-release. |
| `just verify` fails on import | Check the import path is `pfsmgraph.<pkg>`, not `pfsmgraph_<pkg>`. |
| `just preflight` rejects the version | `pyproject.toml` declares something else. The argument is not the source of truth. |
| `403 Invalid or non-existent authentication information` | The stored token is wrong, revoked, or **truncated**. Re-run `just token-set`, which now verifies the round-trip. Nothing was uploaded; the version is not spent. |
