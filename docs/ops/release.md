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
- `LICENSE` -- a **real copy**, never a symlink to the repo-root one.
- the `Typing :: Typed` classifier.
- `src/pfsmgraph/<pkg>/py.typed` -- the PEP 561 marker, **inside** the importable package.

Two of these fail silently if placed wrong, which is why
[`core.md`](../agents/core.md) carries them as an invariant rather than a checklist:

- A symlinked `LICENSE` builds a valid-looking sdist and then fails on **unpack** -- a
  symlink escaping the sdist root is refused. That is a consumer-side failure, caught here
  only because `uv build` routes wheel-building through the sdist.
- A `py.typed` at the distribution root instead of inside the importable package reaches no
  wheel at all, with no error and no warning, and a type checker then discards every
  annotation in the package. It cannot go at the `pfsmgraph/` namespace level either, for
  the same reason no `__init__.py` may: no single distribution owns that level.

`pfsmgraph-dataseq` has all four as of the 0.1.0 release commit. Each remaining member owes
them in its own release commit; see `docs/plan/DEFERRED.md`.

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
differ in substance. Measured 2026-09-02 on `pfsmgraph-dataseq` 0.1.0, from an unchanged
package tree:

- **The wheel is byte-identical** -- same SHA-256 across builds. Hatchling normalises
  member timestamps (every entry reads `02-02-2020 00:00`), so the wheel is reproducible
  and a rebuild is a no-op you can trust. This is the artifact essentially every consumer
  installs.
- **The sdist is not**, and the reason is worth knowing rather than dismissing as
  timestamps: hatchling finds no `.gitignore` in the member directory, walks up to the VCS
  root, and ships **the repo-root `.gitignore`** inside the sdist. So an edit to root
  housekeeping -- a rule about a scratch directory, with nothing to do with this
  distribution -- changes the sdist. That is exactly how the two builds diverged.

The practical consequence is small: the shipped `.gitignore` is inert noise, not a leak,
and the sdist is otherwise a pure function of the member. But it means **the sdist is not a
function of the member alone**, which is worth remembering before concluding that two
differing sdists indicate a real change. `exclude = ["/.gitignore"]` under
`[tool.hatch.build.targets.sdist]` does *not* remove it -- tried and measured, the file
still ships -- so this is filed in `docs/plan/DEFERRED.md` rather than fixed in passing.

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
| `just token-set [pkg]` | Store a PyPI token in the macOS Keychain (prompts, no echo) |
| `just token [pkg]` | Read it back; fails loudly if absent |
| `just preflight VER [pkg]` | Assert the version was built, the tree is clean, and push |
| `just publish [pkg]` | Upload to PyPI |
| `just verify VER [pkg]` | Install from PyPI in a throwaway env and import |
| `just release VER [pkg]` | test -> build -> check -> preflight -> publish -> tag |
| `*-test` variants | Same, against TestPyPI |

Every recipe takes an optional package name defaulting to `pfsmgraph-dataseq`, so one file
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

Store it in the Keychain, never in `.env`, `~/.zshrc`, `~/.pypirc`, or the repo tree:

```bash
just token-set             # or: just token-set pfsmgraph-align
```

That prompts without echoing, so nothing lands in zsh history. `uv` does not read
`.pypirc` at all -- verified against the binary -- so a token placed there would appear to
be configured and do nothing.

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
`pfsmgraph-dataseq-v0.1.0` -> push.

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

---

## Incident reference

| Situation | Action |
| --- | --- |
| Bad version already uploaded | **Yank** it, ship the next patch. Do not delete. |
| Token leaked | Revoke at https://pypi.org/manage/account/ immediately, audit release history. |
| Upload fails halfway | Re-run; `--check-url` skips what's already there. |
| `400 File already exists` | Version is spent. Bump and rebuild. |
| Wheel contains `pfsmgraph/__init__.py` | Namespace collision with siblings. Fix the build, bump, re-release. |
| Wheel contains no `py.typed` | Marker is at the distribution root, not inside the package. Bump, re-release. |
| sdist fails on unpack | `LICENSE` is a symlink. Replace with a real copy, bump, re-release. |
| `just verify` fails on import | Check the import path is `pfsmgraph.<pkg>`, not `pfsmgraph_<pkg>`. |
| `just preflight` rejects the version | `pyproject.toml` declares something else. The argument is not the source of truth. |
