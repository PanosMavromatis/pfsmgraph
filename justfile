# Release recipes for the pfsmgraph workspace.
#
# Run `just` with no arguments to see everything available.
# Every recipe takes an optional package name, defaulting to the one below,
# so the same recipes serve all five members:
#
#   just build                          # builds the default package
#   just build pfsmgraph-align          # builds something else
#   just release 0.1.0                  # full release of the default package
#   just release 0.1.0 pfsmgraph-align  # full release of another
#
# Note on the comments below: `just --list` takes the LAST contiguous comment
# line above a recipe as its description, so any explanatory prose is separated
# from the recipe by a blank line and only the one-line summary sits adjacent.

# The member under development, never an already-published one: a release that
# omits the package argument then builds a .dev0 version that cannot match the
# requested one, so preflight stops it before publish, instead of acting on a
# member that is already on PyPI. Move it to the next member at each release.
default_package := "pfsmgraph-hmm"

# Show available recipes.
default:
    @just --list --unsorted


# --- build -----------------------------------------------------------------

# Remove build artifacts.
clean:
    rm -rf dist/

# `dist/` is one directory shared by all five workspace members, and the publish
# glob below is scoped by package name but NOT by version. `clean` is what keeps
# that glob from picking up a stale version of the same package -- it is not
# merely tidiness, and removing it makes the glob a live hazard.

# Build sdist + wheel from a clean dist/.
build package=default_package: clean
    uv build --package {{ package }}

# Validate that artifacts will render on PyPI before uploading.
check package=default_package:
    uvx twine check dist/{{ replace(package, "-", "_") }}-*

# The suite includes the ADR 0013 verifier, which executes every code block in
# `docs/api/*/*.md` and in `packages/*/README.md` against its pasted output. A
# member README becomes a PyPI long description under an immutable version, so
# that check is the one whose failure cannot be corrected in place after upload.
# It gates `release` for exactly that reason.

# Run the full test suite.
test:
    uv run pytest


# --- credentials -----------------------------------------------------------
#
# Tokens live in one of two places, chosen by `uname`, and never in a tracked
# file. On macOS, in the Keychain: `token-set` prompts without echoing, so
# nothing lands in shell history. On Linux, in a per-package environment
# variable exported from the repo's gitignored .envrc (direnv), e.g.
# PYPI_TOKEN_PFSMGRAPH_HMM. Per-package so a token scoped to one member can
# never be handed to another member's upload -- that run finds no variable and
# stops. direnv exports it to every process started in the repo, which is why
# only `token` reads it and `publish` hands it to `uv publish` explicitly.
#
# The rest of this block is about the macOS path.
#
# `-U` is what makes `token-set` idempotent, and it is required rather than
# tidy: without it `security add-generic-password` REFUSES when the item
# already exists, so rotating a token would fail with "The specified item
# already exists in the keychain" and force a manual delete first. Rotation is
# the common case for a setter that outlives one release.
#
# `-a`/`-s` are an arbitrary composite lookup key, not credentials. $USER is
# the local login name and namespaces the entry; the service string is a name
# this file invents and only has to match between `token-set` and `token`.
# Neither is a PyPI identity: `uv publish` supplies the literal `__token__`
# username itself whenever it is given a token rather than a user/password.

# Store a PyPI token for a package (macOS Keychain; prompts for the value).
token-set package=default_package: (_token-prompt ("pypi-" + package) ("PYPI_TOKEN_" + uppercase(replace(package, "-", "_"))))

# Store a TestPyPI token for a package (macOS Keychain; prompts for the value).
token-set-test package=default_package: (_token-prompt ("testpypi-" + package) ("TESTPYPI_TOKEN_" + uppercase(replace(package, "-", "_"))))

# Prompt for a token, store it under `service`, and verify it round-trips.
#
# Only the service name crosses a process boundary here; the token itself is
# read into this shell and reaches the argv of `security` alone, which is the
# one exposure the Keychain CLI offers no way to avoid.

_token-prompt service var:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "$(uname -s)" != Darwin ]]; then
      echo "token-set writes the macOS Keychain; on Linux export {{ var }} in the repo's gitignored .envrc and run: direnv allow" >&2
      exit 1
    fi
    read -rsp "token for {{ service }} (input hidden): " tok; echo
    [[ -n "$tok" ]] || { echo "empty token -- nothing stored" >&2; exit 1; }
    [[ "$tok" == pypi-* ]] || { echo "token does not begin with 'pypi-' -- both PyPI and TestPyPI tokens do; nothing stored" >&2; exit 1; }
    security add-generic-password -U -a "$USER" -s '{{ service }}' -w "$tok"
    back="$(security find-generic-password -a "$USER" -s '{{ service }}' -w)"
    if [[ "$back" != "$tok" ]]; then
      security delete-generic-password -a "$USER" -s '{{ service }}' >/dev/null 2>&1 || true
      echo "stored ${#back} of ${#tok} characters -- entry deleted rather than left broken" >&2
      exit 1
    fi
    echo "stored ${#tok} characters under {{ service }}"

# A missing token must fail loudly. An empty UV_PUBLISH_TOKEN is not an error
# to `uv publish`: it falls through to trusted-publishing discovery, which
# resolves only inside CI, so outside CI the result is a confusing OIDC failure
# rather than "no token stored".

# Read a stored PyPI token to stdout.
token package=default_package: (_token-read ("pypi-" + package) ("PYPI_TOKEN_" + uppercase(replace(package, "-", "_"))) ("token-set " + package))

# Read a stored TestPyPI token to stdout.
token-test package=default_package: (_token-read ("testpypi-" + package) ("TESTPYPI_TOKEN_" + uppercase(replace(package, "-", "_"))) ("token-set-test " + package))

# Print the token from the Keychain (macOS) or from `var` (Linux), or fail.
#
# `${!var}` is indirect expansion: just interpolates the variable's *name*, and
# the value never passes through just's templating. Every argument above is
# parenthesised because just reads `package (...)` as a call to a function
# named `package`.

_token-read service var hint:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "$(uname -s)" == Darwin ]]; then
      security find-generic-password -a "$USER" -s '{{ service }}' -w \
        || { echo "no keychain entry {{ service }} -- run: just {{ hint }}" >&2; exit 1; }
    else
      var='{{ var }}'
      tok="${!var:-}"
      [[ -n "$tok" ]] || { echo "{{ var }} is unset or empty -- export it in the repo's gitignored .envrc and run: direnv allow" >&2; exit 1; }
      printf '%s\n' "$tok"
    fi


# --- publish ---------------------------------------------------------------
#
# >>> THIS IS THE SWITCHING POINT <<<
# Moving a package from local-token publishing to Trusted Publishing means
# editing the body of `publish` and nothing else. Every other recipe, and
# every habit built on top of them, stays identical.
#
# The token is captured first and `&&` chains the upload. The earlier form,
# `UV_PUBLISH_TOKEN="$(just token pkg)" uv publish`, ran `uv publish` with an
# empty token when the read failed -- a failure inside `$(...)` does not stop
# the command it prefixes -- and so produced exactly the OIDC confusion the
# credentials block warns about. Measured 2026-09-13; it affected macOS too.

# Upload built artifacts to PyPI.
publish package=default_package:
    tok="$(just token {{ package }})" \
      && UV_PUBLISH_TOKEN="$tok" uv publish \
        --check-url https://pypi.org/simple/{{ package }}/ \
        dist/{{ replace(package, "-", "_") }}-*

# Upload built artifacts to TestPyPI instead.
publish-test package=default_package:
    tok="$(just token-test {{ package }})" \
      && UV_PUBLISH_TOKEN="$tok" uv publish \
        --publish-url https://test.pypi.org/legacy/ \
        dist/{{ replace(package, "-", "_") }}-*


# --- preflight -------------------------------------------------------------
#
# Everything here must pass BEFORE anything irreversible happens, which is why
# it is a prerequisite of `release` sitting to the left of `publish` rather than
# a check in the recipe body -- a body line runs after every prerequisite,
# including the upload.
#
# The version argument is otherwise inert: `build` reads the version from
# pyproject.toml and the publish glob carries no version at all, so without this
# assertion `just release 0.2.0` would publish 0.1.0 and tag it v0.2.0. Both
# halves of that are irreversible.
#
# ONLY MEANINGFUL AS A PREREQUISITE OF `release`, never on its own. The version
# check matches the *filename* in dist/, so it cannot tell a current artifact
# from a stale one carrying the same version number -- run standalone against an
# old build it passes and reads as verification. Inside `release` that gap does
# not exist, because `clean` and `build` run first and it only ever sees an
# artifact built moments earlier. Observed 2026-09-02: it passed against a wheel
# whose metadata three commits had since changed.

# Assert the requested version was built and the tree is committed and pushed.
preflight version package=default_package:
    @test -f "dist/{{ replace(package, "-", "_") }}-{{ version }}-py3-none-any.whl" \
      || { echo "dist/ has no {{ package }} {{ version }} wheel -- pyproject.toml declares a different version" >&2; exit 1; }
    @test -f "dist/{{ replace(package, "-", "_") }}-{{ version }}.tar.gz" \
      || { echo "dist/ has no {{ package }} {{ version }} sdist" >&2; exit 1; }
    @git diff --quiet && git diff --cached --quiet \
      || { echo "working tree is dirty -- a tag would name a state that exists nowhere else" >&2; exit 1; }
    git push origin HEAD


# --- verify ----------------------------------------------------------------
#
# Install a published version in a throwaway env and import it.
# Pinning the version proves you're exercising the new release, not a cache.
#
# Note the two different name transforms. The distribution is
# "pfsmgraph-dataseq", its files on disk are "pfsmgraph_dataseq-*", and it
# imports as "pfsmgraph.dataseq" -- a PEP 420 namespace package shared with
# its four siblings. Hyphen -> underscore for globs, hyphen -> dot for imports.

# Install the published version from PyPI and import it.
verify version package=default_package:
    uv run --no-project --with '{{ package }}=={{ version }}' \
      -- python -c "import {{ replace(package, '-', '.') }}; \
         from importlib.metadata import version; print(version('{{ package }}'))"

# `unsafe-best-match` is required because real dependencies -- numpy here --
# live on PyPI proper, not on TestPyPI.

# Same, against TestPyPI.
verify-test version package=default_package:
    uv run --no-project --with '{{ package }}=={{ version }}' \
      --index https://test.pypi.org/simple/ \
      --index-strategy unsafe-best-match \
      -- python -c "import {{ replace(package, '-', '.') }}; \
         from importlib.metadata import version; print(version('{{ package }}'))"


# --- release ---------------------------------------------------------------
#
# Prerequisites run first, left to right, and the run stops at the first
# failure: test -> build -> twine check -> preflight -> publish, and only then
# does the body tag. Nothing reaches PyPI until every check has passed, and
# nothing is tagged until the upload has succeeded.
#
# The tag is package-prefixed deliberately: bare `v0.1.0` becomes ambiguous
# the moment a second package in this repo ships its own 0.1.0.

# Test, build, validate, upload, and tag.
release version package=default_package: test (build package) (check package) (preflight version package) (publish package)
    git tag -a {{ package }}-v{{ version }} -m "{{ package }} {{ version }}"
    git push origin {{ package }}-v{{ version }}

# Rehearse on TestPyPI: test, build, validate, upload there, no tag.
release-test version package=default_package: test (build package) (check package) (publish-test package)
    @echo "Rehearsed {{ package }} {{ version }} on TestPyPI. Nothing tagged."
