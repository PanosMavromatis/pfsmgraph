# 0018. The build backend is family-wide: all five members on meson-python

- **Status:** Accepted — supersedes
  [ADR 0012](0012-align-and-hmm-temporarily-on-hatchling.md) and overrides both halves of
  [ADR 0008](0008-per-package-build-backends.md).
- **Date:** 2026-09-04
- **Source:** branch `exp/meson-python-namespace`; the plan and its measurements at
  `docs/plan/exp-meson-python-namespace/TODO.md`

## Context

[ADR 0008](0008-per-package-build-backends.md) split the build backend by package:
meson-python for the compiled members (`align`, `hmm`), hatchling for the pure ones
(`dataseq`, `hseg`, `dl`). [ADR 0012](0012-align-and-hmm-temporarily-on-hatchling.md)
then suspended the meson-python half, because applying it broke the workspace, and
deferred the resolution **to the moment the first `.pyx` lands**.

**That deferral's premise is false.** The finder that breaks the workspace is injected by
the *editable install*, not by compilation, so it appears identically with every
extension block dormant. ADR 0012 names the right mechanism in its Context — a
`sys.meta_path` finder — and then reasons from the wrong one in its Alternatives, where
it rejects solving the problem now because "choosing between them without a compiled
kernel to evaluate against would be guessing". The information needed to choose was
already present, and this branch used it.

**The mechanism, measured rather than remembered.** A meson-python editable install
writes a loader module and installs a `MesonpyMetaFinder` whose claimed prefix set is
`{'pfsmgraph'}` — not `{'pfsmgraph.hmm'}`. That claim is *structural*: meson-python
derives it from the top-level installed name, and under PEP 420
([ADR 0005](0005-namespace-prefix-and-pep-420-layout.md)) the top-level name **is** the
shared namespace. With `hmm` alone on meson-python, all four siblings fail
`ModuleNotFoundError`, and `pfsmgraph.__path__` collapses from five entries to a single
synthetic entry inside the loader file. The namespace is **replaced, not extended**,
which is why no `.pth` ordering trick could ever have helped.

**ADR 0012's second premise is refuted outright.** It records that "two meson-python
editable installs also conflict with each other", and rests its candidate 2 — a single
combined compiled distribution, "so only one meson-python finder ever exists" — on that
claim. This branch could not reproduce it: **finders chain.** A `MesonpyMetaFinder` that
does not recognise a submodule returns `None` and the import falls through to the next
finder. One finder is not the problem; *any* finder is. A combined compiled distribution
would therefore still shadow `dataseq`, `hseg` and `dl`.

Refuting that premise makes visible a fourth option ADR 0012 does not list, and which is
only conceivable once the boundary is understood as **meson-python versus plain `.pth`**
rather than as one finder versus two: give *every* member a finder.

## Decision

**All five members build through meson-python, pure ones included.** Each
`packages/pfsmgraph-*/` carries a `meson.build`; the three pure members' files declare
`project()` with no languages and `find_installation(pure: true)`, and install their
sources and nothing else. No member is left on a plain `.pth`, so no member can be
shadowed.

Stated positively, because the negative framing misleads: **the fix is not to stop the
finder replacing `pfsmgraph.__path__` — it still does — it is to leave no member relying
on `__path__`.** Every member has a finder that claims its own submodule, so nothing
depends on path-based discovery any more.

Two operational requirements are part of the decision rather than incidental to it:

- **`[tool.uv] no-build-isolation-package` must list all five members.** The generated
  editable loader bakes an *absolute* path to `ninja` at build time and never consults
  `PATH`. Under build isolation that path points inside a temporary directory uv deletes
  once the build finishes, so every later import dies `FileNotFoundError` before the
  namespace question is even reachable. Omitting one member restores isolation for that
  member alone, and its baked path dies the same way. PRD §6.1's "`ninja` must be on
  `PATH`" is necessary and **not** sufficient; this is the other half.
- **The root `dev` group carries `numpy` alongside `meson-python`, `cython` and
  `ninja`.** `numpy` is a *build* requirement of `align`/`hmm`, and with build isolation
  off it cannot be supplied by `build-system.requires` at all.

## Consequences

### Positive

- **`uv sync` produces a working, fully importable workspace**, which is what ADR 0012
  bought by retreating to hatchling and is now had without the retreat.
- **The best available dev loop, and the gap widens with compiled code.** Editing an
  existing `.py` is visible with **no sync at all**, because the finder maps to the
  source tree. The alternative needs a rebuild-and-reinstall for the same edit, becoming
  a full recompile at the first `.pyx`.
- **The characteristic failure is loud.** A module missing from `install_sources` is a
  `ModuleNotFoundError` at import, and `tests/test_meson_sources.py` already guards it —
  parameterised over `packages/*/meson.build`, so it grew 7 → 16 tests by itself when
  the three new files appeared. This is the property that decided the choice; see
  Alternatives.
- **One build system, not two.** This inverts ADR 0008's "two build systems in one
  repository, with two sets of conventions and failure modes to know" cost.
- **`align`'s and `hmm`'s `meson.build` files stop being dormant.** ADR 0012 listed
  their being unexercised as a cost, to be discovered on revert; that cost is now paid
  and settled — both were in fact wrong, listing only `__init__.py`, and were repaired
  on this branch before anything was evaluated against them.

### Negative / costs

- **Three pure-Python members acquire a compiled member's build backend** and a
  hand-maintained source list for code that will never be compiled. This is the real
  price, and it is exactly what ADR 0008's D7 was written to avoid. It is paid because
  the namespace is shared and PEP 420 gives the finder no finer granularity to claim.
- **The baked-`ninja` footgun is retained rather than retired.** The rejected
  alternative removes it entirely.
- **`meson` has no recursive install.** `dl` needs three separate `install_sources`
  calls, one per subpackage, and a fourth submodule would need a fourth.
- **The four-file release invariant now runs through `install_sources`.** meson does not
  glob, so `py.typed` must be named explicitly. A PEP 561 marker that fails to reach a
  wheel does so silently, and a type checker then discards every annotation in the
  package.
- **`pfsmgraph-dataseq` 0.1.0 was published from hatchling.** Its next release therefore
  ships the first meson-built wheel this project has produced, and the four-file
  invariant must be re-verified against an actual built wheel installed into a clean
  venv rather than assumed to carry over.
- **`docs/ops/release.md`'s reproducibility findings were measured against hatchling** —
  the byte-identical wheel rebuild and the non-reproducible sdist. They are properties of
  that builder. Nothing there is now known false; nothing there is known to still hold.

## Alternatives considered

- **ADR 0012 candidate 1 — non-editable install of the compiled members.** Measured, and
  it works: with no editable install anywhere there is no finder at all, and the suite
  was green. It is the cleaner model and it retires the baked-`ninja` footgun. **Rejected
  on how it fails, not on elegance.** `editable = false` must be repeated at *every*
  `[tool.uv.sources]` declaration site, because a member-level declaration beats the
  workspace root's — and omitting it at any one site resurrects a finder that breaks all
  five members. Separately, without `[tool.uv] cache-keys` a source edit is served stale
  with no error, since uv's default cache key for a local path is its `pyproject.toml`
  rather than its sources. Both failures are silent, where the chosen option's
  characteristic failure is a `ModuleNotFoundError` that a test already catches.
- **ADR 0012 candidate 2 — a single combined compiled distribution.** **Refuted**, not
  deprioritised: its premise was a two-finder conflict that does not occur. See Context.
- **ADR 0012 candidate 3 — an upstream fix.** Not viable, and not for the obvious reason.
  The obvious objection — this project cannot block on someone else's release — is true
  but secondary. The finding is that **upstream does not treat this as a defect**:
  meson-python's editable-installs guide documents the stub-and-finder mechanism without
  mentioning PEP 420 anywhere, and there is no open issue describing the shadowing.
  Combined with the structural derivation of the `{'pfsmgraph'}` claim, there is nothing
  here to report as broken. It would be a *feature request* for finder composition —
  worth filing on its own merits, and not a candidate for this decision.
- **Stay on hatchling indefinitely.** Rejected: it forecloses meson-python permanently
  rather than temporarily, and the rebuild-on-import loop is wanted for exactly the code
  that is hardest to debug ([ADR 0008](0008-per-package-build-backends.md)).
- **Revert to `setuptools.build_meta`, which composes across namespace members.**
  Rejected on the same grounds ADR 0012 rejected it: it reintroduces the stale-`.so` dev
  loop that motivated leaving setuptools in the first place.
- **Drop the shared namespace.** Rejected outright — it is the foundation of
  [ADR 0005](0005-namespace-prefix-and-pep-420-layout.md).

## Evidence

All measured on `exp/meson-python-namespace`, 2026-09-04, and recorded in that branch's
plan. Observations, not inferences, unless marked.

- **The shadowing, reproduced against today's workspace.** With `hmm` alone on
  meson-python, all four siblings fail `ModuleNotFoundError: No module named
  'pfsmgraph.<member>'`, and `pfsmgraph.__path__` holds one synthetic loader entry rather
  than five source directories.
- **Finders chain.** With two and later five meson-python editable installs present, all
  members import; `sys.meta_path` holds one `MesonpyMetaFinder` per member.
- **The landed state, verified from a deleted venv and a plain `uv sync`** — not
  `--reinstall`, which reuses an environment that already has `ninja` on disk and so
  cannot test the question. uv reported *"Prepared 5 packages without build isolation in
  2.06s"*; all seven import paths resolve (`dataseq`, `align`, `hmm`, `hseg`, `dl`,
  `dl.rnn`, `dl.transformer`); five finders sit on `sys.meta_path`; the suite is green at
  280 and `uv lock --check` is clean.
- **Bootstrap ordering resolves on its own.** The five members need `meson-python`,
  `cython`, `ninja` and `numpy` present *before* they build, yet those arrive via the
  `dev` group in the same `uv sync`. uv sequences it correctly — but that ordering is
  *emergent from the resolver rather than declared anywhere in the configuration*, which
  is why it was measured rather than reasoned about.
- **The dev-loop claim was checked, not assumed.** Appending a name to
  `hseg/__init__.py` and importing under `uv run --no-sync` showed it in 0.15 s, with no
  sync at all.
- **A trap for anyone reproducing this**, and not a defect of the decision:
  `no-build-isolation-package` does **not** invalidate uv's build cache. Switching a
  member to it reused an editable wheel built *with* isolation, whose loader carried a
  baked `.../builds-v0/.tmpXXXX/bin/ninja` path uv had already deleted — so one member
  failed while the other four worked, a state that looks like a defect in this decision
  and is a stale cache. `uv sync --reinstall` clears it.
- **External, and reasoned rather than observed:** `microsoft/pylance-release#3002`
  reports the same shape — an editable namespaced install shadowing siblings that share
  the prefix — in a different toolchain. Taken as evidence that the interaction is
  generic to finder-based editable installs rather than a meson-python bug.

## Open

- **What a real `.pyx` still adds**: whether rebuild-on-import works for an actual
  extension, whether a stale `.so` can appear, and the true dev-loop friction under
  compilation. These bear on the *quality* of the arrangement, not on its viability,
  which is why the decision did not wait for them — the distinction ADR 0012 missed.
- **The upstream feature request** for finder composition is unfiled.
- ADR 0008's Open section asked whether `hseg` would need a compiled backend. That
  question survives for [ADR 0002](0002-three-phase-algorithm-lifecycle.md) purposes —
  whether `hseg` has its own DP recurrence — but is now **moot for backend selection**:
  its backend is settled either way.
