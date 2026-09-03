# Release recipes for the pfsmgraph workspace.
#
# Run `just` with no arguments to see everything available.
# Every recipe takes an optional package name, defaulting to the one below,
# so the same recipes serve all five members:
#
#   just build                          # builds pfsmgraph-dataseq
#   just build pfsmgraph-align          # builds something else
#   just release 0.1.0                  # full release of the default package
#   just release 0.1.0 pfsmgraph-align  # full release of another
#
# Note on the comments below: `just --list` takes the LAST contiguous comment
# line above a recipe as its description, so any explanatory prose is separated
# from the recipe by a blank line and only the one-line summary sits adjacent.

default_package := "pfsmgraph-dataseq"

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
# Tokens live in the macOS Keychain, never in a dotfile or the repo tree.
# `token-set` prompts without echoing, so nothing lands in zsh history.
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

# Store a PyPI token for a package (prompts for the value).
token-set package=default_package:
    security add-generic-password -U -a "$USER" -s pypi-{{ package }} -w

# Store a TestPyPI token for a package (prompts for the value).
token-set-test package=default_package:
    security add-generic-password -U -a "$USER" -s testpypi-{{ package }} -w

# A missing keychain entry must fail loudly. An empty UV_PUBLISH_TOKEN is not an
# error to `uv publish`: it falls through to trusted-publishing discovery, which
# resolves only inside CI, so on a laptop the result is a confusing OIDC failure
# rather than "no token stored".

# Read a stored PyPI token to stdout.
token package=default_package:
    @security find-generic-password -a "$USER" -s pypi-{{ package }} -w \
      || { echo "no keychain entry pypi-{{ package }} -- run: just token-set {{ package }}" >&2; exit 1; }

# Read a stored TestPyPI token to stdout.
token-test package=default_package:
    @security find-generic-password -a "$USER" -s testpypi-{{ package }} -w \
      || { echo "no keychain entry testpypi-{{ package }} -- run: just token-set-test {{ package }}" >&2; exit 1; }


# --- publish ---------------------------------------------------------------
#
# >>> THIS IS THE SWITCHING POINT <<<
# Moving a package from local-token publishing to Trusted Publishing means
# editing the body of `publish` and nothing else. Every other recipe, and
# every habit built on top of them, stays identical.

# Upload built artifacts to PyPI.
publish package=default_package:
    UV_PUBLISH_TOKEN="$(just token {{ package }})" \
      uv publish \
        --check-url https://pypi.org/simple/{{ package }}/ \
        dist/{{ replace(package, "-", "_") }}-*

# Upload built artifacts to TestPyPI instead.
publish-test package=default_package:
    UV_PUBLISH_TOKEN="$(just token-test {{ package }})" \
      uv publish \
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
