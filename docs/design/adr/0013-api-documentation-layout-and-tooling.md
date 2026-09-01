# 0013. API documentation lives in a repo-level `docs/api/`, hand-written, with executed examples

- **Status:** **Accepted** (2026-09-01)
- **Date:** 2026-09-01
- **Source:** the `feat/dataseq-merge` branch plan, goal "Write the API documentation for
  `dataseq`"; `dataseq` is the first member with a public surface to document.

## Context

`pfsmgraph` is five independently publishable distributions sharing one PEP 420
namespace. `dataseq` is the first with an implementation, so it is the first to need API
documentation — and whatever is decided for it binds the other four, because the moment a
second package documents anything, an inconsistent layout is a permanent cost.

Three things had to be settled at once: **where** the documentation lives, **how** it is
produced, and **what stops it from going stale**. The third is the one that usually goes
unstated and then decides the outcome anyway.

Two facts about this repository constrained the answer. Documentation is shipped in no
wheel under any layout under consideration, so "which distribution owns this page" is not
a packaging question. And the existing docstrings are already written in reStructuredText
idiom — `Typical use::` literal blocks, `:class:` and `:mod:` roles — which is a partial
commitment made before this decision was taken.

## Decision

API documentation lives at **`docs/api/`, at the repository root, with one subdirectory
per distribution** (`docs/api/dataseq/`) and an index at `docs/api/README.md` naming all
five members and marking those without code. A member gets its subdirectory when it gets
an implementation.

The pages are **hand-written Markdown**. No documentation generator, no build step, and no
addition to the `dev` dependency group.

Two rules make that sustainable:

- **The docstrings are normative for signatures; `docs/api/` is normative for contracts.**
  Names, parameters, defaults, and types are read from the source, which is also where an
  editor and `help()` look. The invariants a caller may rely on, the reasons behind them,
  and the seams between distributions are stated in `docs/api/` and are contracts even
  when stated nowhere else.
- **Every code block is executed and its output pasted from the run**, never transcribed
  from memory. This includes error messages and tracebacks.

## Consequences

### Positive

- The pages render on GitHub today, with no host, no CI, and no build. For a family whose
  members are all at `0.1.0.dev0`, that is the difference between documentation that
  exists and documentation that is one unbuilt toolchain away.
- A repo-level tree is the only place a **cross-package** contract can sit. The sharpest
  thing `dataseq` has to document is that `pfsmgraph-align` reads `Vocabulary.sym_to_code`
  across a distribution boundary — a fact that belongs to neither package alone, and that
  a per-package layout would force into one of them or duplicate into both.
- The per-distribution subdirectory keeps a future split cheap: if the packages ever move
  to separate repositories, each `docs/api/<pkg>/` is a `git mv`.
- The executed-examples rule catches a class of error nothing else here catches. Writing
  this ADR's companion pages, it caught two: a stale module docstring still calling the
  settled encoder API "provisional", and a `KeyError` whose displayed text differs from
  its message because `KeyError.__str__` is `repr(args[0])`.

### Negative / costs

- **Prose can drift from code, and only the executed-examples rule guards it.** This is
  the real cost and it should not be minimised: a generator makes drift impossible for
  signatures, and this does not. The mitigation is narrower than the risk — an example
  that still runs does not prove the paragraph above it is still true.
- Adding a member means writing its pages by hand. That is a per-package cost a generator
  would have amortised.
- Nothing enforces the executed-examples rule automatically. Until CI exists it is a
  discipline, and it is worth a doctest-style check when CI does.

## Alternatives considered

**Sphinx with `autodoc`.** Rejected *for now, not on the merits* — this is a deferral. It
is the natural endpoint: the docstrings already speak reST, so the migration is mostly
configuration rather than rewriting. What rules it out today is that it produces HTML that
is not committed and has nowhere to be served, so until a docs host exists it would
replace pages that render on GitHub with `.rst` sources that do not. Revisit when there is
somewhere to publish, or when a second member needs documenting.

**MkDocs with `mkdocstrings`.** Rejected outright, not deferred. It expects Google- or
NumPy-style docstrings, and every `:class:` and `:mod:` role in the six existing modules
would render as literal text — so adopting it requires rewriting all of them out of reST
first. That is a large, purely mechanical change made to satisfy a tool, and it would
throw away the partial commitment the docstrings already represent.

**Per-package `packages/pfsmgraph-<pkg>/docs/`.** Rejected. Its usual justification is
shipping docs in the wheel, which does not apply — `docs/` is in no wheel under either
layout. Without that, it only scatters the tree, and it puts cross-package contracts in
the one place they cannot sit cleanly.

**A section in `docs/api/README.md` instead of this record.** Rejected. The decision binds
all five members, and the ADR index is only worth consulting if it is the complete list of
family-wide decisions. A binding decision recorded outside it makes every other entry less
trustworthy.

## Open

**Whether the executed-examples rule becomes a doctest run.** The examples are written as
`>>>` blocks, which is most of the way to being executable by `pytest --doctest-glob`.
What stands in the way is that several show tracebacks, and one deliberately shows the
escaped form of a `KeyError` — doctest's `IGNORE_EXCEPTION_DETAIL` and friends would have
to be settled first. Deferred to the trigger "CI existing"; see `docs/plan/DEFERRED.md`.
