# exp/meson-python-namespace

**Status**: active
**Created**: 2026-09-04
**Subgoal**: Resolve the meson-python namespace shadowing and move `hmm` off hatchling,
reverting ADR 0012 by whichever of its three candidates survives contact.

## Goals

- [x] Reproduce the shadowing against today's workspace with `hmm` alone, so the revert
      is measured rather than remembered.
  > **Q:** Subgoal 1 is complete but the tree is in a deliberately broken experimental
  > state (all five imports failing). Commit the plan file only, defer entirely, or
  > commit everything as-is?
  > **A:** Plan file only — capture the findings, leave the experimental `pyproject.toml`
  > and `uv.lock` changes uncommitted, so the branch history never contains a workspace
  > where `uv run pytest` cannot execute.
  > **Result:** ADR 0012 confirmed and extended. With `hmm` alone on meson-python all
  > four siblings fail `ModuleNotFoundError: No module named 'pfsmgraph.<member>'`, and
  > `pfsmgraph.__path__` collapses from five entries to one synthetic entry inside the
  > loader file — the namespace is *replaced*, not extended, which is why no `.pth`
  > ordering trick could have helped. Three things ADR 0012 does not record. (1) The
  > generated loader's `install()` claims `{'pfsmgraph'}`, not `{'pfsmgraph.hmm'}`, and
  > structurally so: meson-python derives the claim from top-level installed names, and
  > under PEP 420 the top-level name *is* `pfsmgraph`. Candidate 3 is therefore an
  > upstream design change, not a bug report. (2) PRD §6.1's "`ninja` must be on `PATH`"
  > is necessary but **not sufficient** — the loader bakes an absolute ninja path at
  > build time and never consults `PATH`; under uv that path is inside the
  > build-isolation temp dir uv deletes, so every import dies `FileNotFoundError` before
  > the shadowing is even reachable. Fix: `[tool.uv] no-build-isolation-package`.
  > (3) `hmm` itself fails too, `No module named 'pfsmgraph.hmm._params'`, because
  > `install_sources` lists only `__init__.py` — the branch-time drift finding,
  > demonstrated. Also noted: the loader honours `MESONPY_EDITABLE_SKIP=<build_path>`
  > to suppress rebuild-on-import.

- [x] Repair the `meson.build` files before evaluating anything.
  > **Q:** How should `install_sources` be kept from drifting again — a repo-root test,
  > reliance on goal 5's import check, or globbing from meson via `run_command`?
  > **A:** A repo-root test parsing each `meson.build` and asserting its `install_sources`
  > matches the package data on disk. The import check only catches modules `__init__.py`
  > transitively imports, so a private module or a `py.typed` would slip through silently
  > — which is the exact failure being guarded against. Meson's no-glob stance is
  > deliberate and a glob would be verified by nothing at test time.
  > **Q:** What should `hmm`'s dormant extension block target — `_viterbi` only, both
  > kernels, or a generalised list?
  > **A:** `_viterbi` only. Baum-Welch gets its own block in revision 03, written against
  > a kernel that exists; two dormant blocks would double exactly the unexercised config
  > ADR 0012 flagged as liable to rot.
  > **Q:** `_viterbi.pyx` would collide with `_viterbi.py`, which the ADR 0003 registry
  > already names as the phase-1 backend, and the two phases must coexist. Name the kernel
  > `_viterbi_cython.pyx`, mirror `align`'s subpackage layout, or use a flat `_cython.pyx`?
  > **A:** `_viterbi_cython.pyx`. Flat beside the reference, reads directly in the registry
  > as `Backend("cython", "pfsmgraph.hmm._viterbi_cython")`, and leaves
  > `_baum_welch_cython` free for revision 03. `align`'s subpackage layout would
  > restructure a surface the registry already points at.
  - [x] Add `_numeric.py`, `_params.py`, `_viterbi.py` to `hmm`'s `install_sources`, and
        decide how to keep that list from silently drifting again.
    > **Result:** Repair verified at the build level, not just in the file: meson's own
    > `intro-install_plan.json` now lists all four modules, and the import error moved
    > from `No module named 'pfsmgraph.hmm._params'` to `No module named
    > 'pfsmgraph.dataseq'` — nothing of `hmm`'s own is missing any more, leaving only the
    > sibling shadowing. Drift prevention is `tests/test_meson_sources.py` (7 tests, suite
    > 264 → 271), mutation-tested both ways it matters: dropping a module from
    > `install_sources` fails it, and so does a `py.typed` appearing on disk unlisted —
    > the release-commit scenario, which is the case a "every module is listed" guard
    > would have missed. A dormant `if fs.exists()` install block for `py.typed` was
    > rejected: adding unexercised build config to guard against unexercised build config
    > rotting is not a repair.
  - [x] Retarget the dormant extension block: the first `.pyx` is Viterbi, not
        Baum-Welch.
    > **Result:** `_baum_welch.pyx` → `_viterbi_cython.pyx`, module `_viterbi_cython`.
    > Also corrected two stale statements the reading turned up: `hmm`'s header claimed
    > only the Baum-Welch core is ever cythonized, which predates ADR 0002's lifecycle
    > being applied to the decode; and `align/meson.build` asserted `ninja` "is in the
    > workspace-root [dependency-groups].dev", which ADR 0012 had removed — and which is
    > insufficient regardless, per the previous goal's finding.

- [x] Confirm ADR 0012's second, independent failure — two meson-python editable installs
      conflicting with each other — by putting `align` on meson-python too.
  > **Result:** **Not confirmed — refuted.** The two finders chain rather than conflict.
  > With `align` and `hmm` both on meson-python, `pfsmgraph.align` imports fine even
  > though `hmm`'s finder sits ahead of it at `meta_path[0]`: a finder that does not
  > recognise a submodule returns `None` and the import falls through to the next.
  > Ordering is deterministic — each loader does `sys.meta_path.insert(0, …)`, so the last
  > `.pth` processed ends up first — and stable across three fresh interpreters. `hmm`'s
  > own failure is not a lookup failure either: the traceback originates at
  > `hmm/_params.py:41`, having already executed `hmm/__init__.py`, and dies importing
  > `pfsmgraph.dataseq`. The controlled comparison settles it — `align` failed in the
  > previous goal and succeeds here, with nothing changed but its own build backend.
  > So the boundary is **meson-python vs plain `.pth`**, not meson-python vs
  > meson-python: members with a finder resolve, members relying on PEP 420 path-based
  > discovery (`dataseq`, `hseg`, `dl`) are shadowed, because the first finder replaced
  > `pfsmgraph.__path__` with a synthetic single entry. Likely explanation for the
  > original claim: without `no-build-isolation`, every import dies `FileNotFoundError`,
  > which from outside looks exactly like everything conflicting with everything.
  > **Q:** The "evaluate the three candidates" goal is now inaccurate — one is refuted and
  > a fourth has appeared. Rewrite to three live candidates, keep candidate 2 for
  > evaluation anyway, or test candidate 4 before deciding?
  > **A:** Rewrite to three live candidates. Record candidate 2 as refuted with its
  > evidence and carry that into the superseding ADR.
  > **Q:** This needs both members on meson-python, but `hmm`'s `meson.build` drift would
  > confound the two-finder measurement. Repair `meson.build` first, run it anyway, or
  > descope it?
  > **A:** Repair `meson.build` first, then return to this against a clean `hmm` — measure
  > one thing at a time. Split out of the reproduction goal and placed below the repair so
  > `/hitl-step`'s default selection matches the decision, since `[~]` would otherwise
  > take priority over `[ ]` wherever the goals sat.

- [x] Settle the sequencing question — throwaway extension here, or interleave with the
      Viterbi `.pyx` from the next master-plan goal.
  > **Q:** The candidate evaluation turns out not to need compiled code. Decide and land
  > here with no `.pyx`, write a throwaway extension to evaluate against, or decide here
  > and land the revert inside the phase-2 goal?
  > **A:** Decide and land here, no `.pyx`. The phase-2 goal then writes the real
  > `_viterbi_cython.pyx` into a workspace already on meson-python, which is itself the
  > first genuine exercise of the config.
  > **Done:** **ADR 0012's stated premise for deferring is falsified.** Its Alternatives
  > section holds that "choosing between them without a compiled kernel to evaluate
  > against would be guessing. The information needed to choose arrives with the first
  > `.pyx`." Three measurements on this branch contradict it: the failure is caused by the
  > *editable install's* `sys.meta_path` finder, which is injected identically whether the
  > `if fs.exists()` extension block fires or not; candidate 2 was refuted with dormant
  > blocks; and candidate 4 was *discovered* with them. The error is specific and worth
  > naming in the superseding ADR — 0012 states the right mechanism in its Context
  > ("meson-python's editable-install import hook injects a finder") and then reasons from
  > the wrong one two sections later, treating the deferred cost as *compilation* when the
  > mechanism it had just described is *editable installation*. Those are separable: a
  > meson-python member with no compiled code at all still injects the finder.
  > What a real `.pyx` would still add — whether rebuild-on-import works under the winning
  > candidate, whether a stale `.so` can appear, how much dev-loop friction each option
  > costs — bears on the *quality* of candidates 1 and 4, not on which are viable, and
  > arrives one master-plan goal later regardless.
  > **Coupling to carry forward:** "when to land" is downstream of "what to choose", not
  > independent of it. Candidate 1 (non-editable) makes every `.py` edit in `hmm` require
  > a reinstall — a daily cost paid now for a benefit that does not exist until the
  > `.pyx`, and strictly worse than today's hatchling + editable. Candidate 4 (all five on
  > meson-python, all editable) costs nothing extra today. So if candidate 1 wins, the
  > apply step may still want deferring even though the decision does not.

- [x] Evaluate the three **live** candidates and choose. No compiled member is required,
      per the sequencing decision above.
  > **Q:** Both live candidates work and the suite is green under each — candidate 1
  > (compiled members non-editable, no finder at all, 271 passed) and candidate 4 (all
  > five on meson-python, five finders chaining, 280 passed). Which resolves ADR 0012?
  > **A:** **Candidate 4 — all five members on meson-python.**
  > **Done:** Chosen on failure loudness and dev loop, not on elegance. Candidate 1 has
  > the cleaner mental model — no finder exists, PEP 420 resolution stays ordinary, and
  > the baked-`ninja` footgun is retired — but both of its characteristic mistakes fail
  > *silently*, and one of them fails **globally**: `editable = false` omitted at any one
  > of four declaration sites resurrects a finder that breaks all five members, and a
  > missing `cache-keys` serves a stale module with no error. Candidate 4's characteristic
  > mistake is a missing `meson.build` entry, which is a `ModuleNotFoundError` at import
  > and is already caught by `tests/test_meson_sources.py` — the guard this branch built
  > two goals ago, which extended itself from 7 to 16 tests the moment the three new
  > `meson.build` files appeared. That is the same criterion this project has applied to
  > the misplaced `py.typed`, the inert `.gitignore` rule and the workspace version
  > footgun: prefer the arrangement whose mistakes are detectable, and pay for it.
  > The dev loop reinforces it rather than deciding it — 0.15 s with no sync against
  > 2.7 s, and the gap *widens* at the first `.pyx`, where candidate 1 degrades to a full
  > recompile per edit while candidate 4 keeps rebuild-on-import. Note this reverses the
  > naive reading of ADR 0012, which treated non-editable installation as the low-cost
  > fallback and rebuild-on-import as the thing being given up.
  > **What the superseding ADR must argue, not assume.** (1) It overrides **ADR 0008**'s
  > per-package build backends, which is a real conflict and not a technicality: three
  > members with no compiled code acquire meson-python and a hand-maintained
  > `install_sources` list. The argument is that ADR 0008's purpose is served by the
  > *distributions* staying independent, which they do — what is shared is a build
  > backend, not a release cadence — and that the alternative is a silent, workspace-wide
  > failure mode. (2) `pfsmgraph-dataseq` **0.1.0 is already published, built with
  > hatchling**; its next release ships a meson-built wheel, so the four-file release
  > invariant (README, LICENSE copy, `Typing :: Typed`, `py.typed` inside the package)
  > must be re-verified against an actual built wheel in a clean venv at that point. The
  > `py.typed` is already listed in its new `install_sources` and the meson-sources test
  > covers the listing, but the listing is not the wheel.
  - [x] **Settle candidate 1's precondition first:** can uv install one workspace member
        non-editable while the other four stay editable? `uv sync --no-editable-package
        <PKG>` exists in uv 0.12.9, but the candidate needs it to hold for a plain
        `uv sync`, so the question is whether `[tool.uv]` mirrors the flag the way it
        mirrors `no-build-isolation-package`. If it does not, candidate 1 costs a
        non-default sync invocation that every contributor and every CI job must
        remember, which is a different proposition from what ADR 0012 records.
    > **Result:** **The precondition holds, but not by the mechanism the question
    > assumed, and the dev-loop cost is worse than ADR 0012 records.** Measured on uv
    > 0.12.9. (1) `[tool.uv]` does **not** mirror `--no-editable-package`: uv's own
    > unknown-field error enumerates the accepted keys, and `no-editable-package` is
    > absent where `no-build-isolation-package` is present. Nor is there an env var —
    > `--no-editable` carries `[env: UV_NO_EDITABLE=]` but is all-or-nothing across the
    > whole workspace, and the per-package flag carries none. (2) The mechanism that
    > *does* work is better than either: `[tool.uv.sources]` accepts an `editable` key,
    > so `pfsmgraph-hmm = { workspace = true, editable = false }` in the workspace-root
    > `pyproject.toml` is honoured by a plain `uv sync`. Verified on a two-member probe
    > workspace: the `editable = false` member lands as a real copied directory in
    > `site-packages` with no `.pth`, while its sibling keeps its
    > `_editable_impl_<name>.pth`. So nothing non-default has to be remembered by a
    > contributor or a CI job, and the objection this subgoal was written to test does
    > not apply.
    > **(3) The cost is not "a manual reinstall step" — it is silent staleness.** With the
    > member non-editable, editing only a `.py` and running plain `uv sync` reports
    > `Checked 2 packages`, rebuilds nothing, and leaves the previous copy installed; the
    > import then returns the *old* value with no error and no warning. uv's default cache
    > key for a local source is its `pyproject.toml`, not its sources. The repair is
    > `[tool.uv] cache-keys = [{ file = "src/**/*.py" }]` in the member's own
    > `pyproject.toml`, after which a plain `uv sync` does rebuild and reinstall on a
    > `.py` edit. Controlled both ways, with the confound removed — the first measurement
    > of the fix changed `pyproject.toml` in the same step, which is itself the default
    > cache key, so it was re-run touching only the `.py`: with `cache-keys` the edit
    > propagates, without it the module stays stale.
    > **Bearing on the choice:** candidate 1 is viable and cheaper to invoke than
    > believed, but it needs *two* pieces of configuration rather than none, and the
    > failure mode of forgetting the second is a silently stale module — the same class
    > as the stale `.so` this branch already flagged as an open question for the `.pyx`,
    > arriving a revision early and for pure Python.
  - [x] **Candidate 1 — non-editable install of the compiled members.** No finder exists
        at all, so nothing is shadowed; costs a manual reinstall step in the dev loop,
        close to the setuptools status quo ante.
    > **Result:** **Works, end to end, and removes a requirement rather than adding one.**
    > With `align` and `hmm` reverted to meson-python and both declared
    > `editable = false`, a clean `uv sync` yields `site-packages/pfsmgraph/` holding
    > `align/` and `hmm/` as real directories, three `_editable_impl_*.pth` for the pure
    > members, **no meson loader and no meson finder in `sys.meta_path` at all**.
    > `pfsmgraph.__path__` composes to four portions and all five members import. Suite
    > green at 271. Notably `[tool.uv] no-build-isolation-package` is **not** needed: it
    > existed to keep the *editable* loader's baked absolute `ninja` path valid for
    > rebuild-on-import, and a non-editable install has no loader and never rebuilds. So
    > candidate 1 retires the footgun measured in goal 1 instead of inheriting it.
    > **Two costs, both discovered by measurement rather than reasoning.**
    > **(a) `editable = false` is not centrally declarable, and getting it wrong fails
    > workspace-wide.** The first attempt set it only in the workspace root; `hmm` went
    > non-editable but `align` stayed editable, and its surviving finder then broke *all
    > five* imports with the baked-`ninja` `FileNotFoundError`. The cause is that
    > `hmm`, `hseg` and `dl` each declare `pfsmgraph-align = { workspace = true }` in
    > their own `[tool.uv.sources]`, and a member-level declaration beats the root's;
    > `hmm` was unaffected only because no member depends on it. Making it work took
    > `editable = false` repeated at all four declaration sites. The failure mode is
    > therefore: a future member that depends on `align` and copies the existing
    > one-line source declaration silently re-editables it and breaks the whole
    > workspace — the same shape as the `install_sources` drift this branch already
    > fixed, and not guarded by anything.
    > **(b) Silent staleness, confirmed on the real workspace.** Editing
    > `hmm/__init__.py` and running plain `uv sync` reports `Checked 26 packages` and
    > leaves the stale copy installed; the marker is absent with no error.
    > `uv sync --reinstall-package pfsmgraph-hmm` repairs it, and so does
    > `[tool.uv] cache-keys = [{ file = "src/**/*.py" }]` in the member — after which
    > **`uv run` alone suffices**, rebuilding and reinstalling in ~2.7 s before running.
    > That is the dev loop actually in use here (`uv run pytest`), so with `cache-keys`
    > the ergonomics are acceptable today. What it becomes at the first `.pyx` is a full
    > recompile per edit rather than a 2.7 s pure-Python wheel rebuild, since uv builds
    > into a fresh directory with no incremental reuse — which is precisely the
    > dev-loop-friction question this plan recorded as arriving with the real kernel.
  - [x] **Candidate 4 — all five members on meson-python.** Every member gets a finder and
        they chain, which is what this branch measured. Not in ADR 0012. Cuts against
        ADR 0008's per-package backends, and hands three pure-Python members a compiled
        member's build backend for no compiled code.
    > **Result:** **Works, and has the better dev loop of the two.** Three new
    > `meson.build` files were written for `dataseq`, `hseg` and `dl` — `project()` with
    > no languages, `find_installation(pure: true)` — and all five members moved to
    > `mesonpy`. After a clean sync all five import, `sys.meta_path` holds five
    > `MesonpyMetaFinder`s, and the suite is green at **280**. `pfsmgraph.__path__` is
    > still a single synthetic entry, which no longer matters: the namespace is replaced,
    > but every member now has a finder that claims its own submodule, so nothing depends
    > on path-based discovery any more. That is the fourth option stated positively —
    > the fix is not to stop the finder replacing `__path__`, it is to leave no member
    > relying on `__path__`.
    > **The dev loop is the strongest argument for it.** Editing an existing `.py` is
    > visible with **no sync at all** (`uv run --no-sync`, 0.15 s) because the finder maps
    > to the source tree; candidate 1 needs a 2.7 s rebuild-and-reinstall for the same
    > edit, growing to a full recompile once a `.pyx` exists. Adding a *new* module does
    > require a `meson.build` edit — unlisted, it is `ModuleNotFoundError` — but once
    > listed, rebuild-on-import picks it up with no sync either. Crucially that failure is
    > **loud**, and `tests/test_meson_sources.py` already guards it: parameterised over
    > `packages/*/meson.build`, it went 7 → 16 tests by itself when the three files
    > appeared, which is where 271 → 280 comes from. Contrast candidate 1's staleness,
    > which is silent and guarded by nothing.
    > **Costs, stated fairly.** Every member now needs `no-build-isolation-package` and
    > the root `dev` group needs `meson-python`, `cython`, `ninja`, `numpy` — the goal-1
    > footgun is retained rather than retired, where candidate 1 removes it. Three
    > pure-Python members acquire a compiled member's build backend and a hand-maintained
    > source list for no compiled code, which is what cuts against ADR 0008.
    > **One trap found, and it is not candidate 4's fault but will bite anyone
    > reproducing this:** `no-build-isolation-package` does **not** invalidate uv's build
    > cache. Switching a member to it reused an editable wheel built *with* isolation,
    > whose loader had a baked `.../builds-v0/.tmpXXXX/bin/ninja` path uv had already
    > deleted, so `align` alone failed while the other four worked — a state that looks
    > like a candidate-4 defect and is a stale cache. `uv sync --reinstall` clears it.
  - [x] **Candidate 3 — an upstream fix**: a finder that defers to other finders for
        prefixes it does not own. Now known to be a design change rather than a bug
        report, since the `{'pfsmgraph'}` claim is derived structurally from the PEP 420
        layout rather than chosen.
    > **Result:** **Not viable as this branch's resolution, and not for the obvious
    > reason.** The obvious objection — we cannot block on someone else's release — is
    > true but secondary. The finding is that upstream does not treat this as a defect:
    > meson-python's own editable-installs guide documents the stub-and-finder mechanism
    > without mentioning PEP 420 anywhere, and there is no open issue describing the
    > shadowing. The nearest published report is `microsoft/pylance-release#3002`, which
    > is the same shape (an editable namespaced install shadowing siblings sharing the
    > prefix) in a different toolchain, which is evidence that the interaction is generic
    > to finder-based editable installs rather than a meson-python bug. Combined with
    > goal 1's structural finding — the `{'pfsmgraph'}` claim is derived from the
    > top-level installed name, and under PEP 420 that name *is* the shared namespace —
    > there is nothing here to report as broken. Filing upstream would be a feature
    > request for finder composition, worth doing on its own merits but not a candidate
    > for this decision. Keep as a note in the superseding ADR, not as an open option.
  > **Note:** ADR 0012's candidate 2 — a single combined compiled distribution, "so only
  > one meson-python finder ever exists" — is **refuted**, not merely deprioritised. Its
  > premise was the two-finder conflict, which this branch could not reproduce. One finder
  > is not the problem; any finder is, so a combined compiled distribution would still
  > shadow `dataseq`, `hseg` and `dl`.

- [x] Land the choice.
  > **Done:** All five members are on meson-python in the working tree, verified from a
  > genuinely clean venv. Scope note: this goal's subgoals were drafted while candidates
  > 1–2 were still live, so the first one says "both members" — the chosen candidate makes
  > it five, and `dataseq`, `hseg` and `dl` had no revert recipe to follow because nobody
  > expected them to move.
  - [x] Revert recipe applied to both members; `[tool.hatch.*]` dropped.
    > **Result:** Applied to all five, not two. The two dead `[tool.hatch.build.targets.wheel]`
    > blocks that were still present (`dataseq`, `dl`) are gone; `align`/`hmm`/`hseg` had
    > none, so the tree had been inconsistent. **Every one of the five build-system
    > comments was false or misleading**, not just the two carrying "TEMPORARILY
    > hatchling": `dataseq` said switch to meson-python *only if* a compiled inner loop is
    > found, `hseg` said "hatchling while hseg is pure orchestration", and `dl`'s "Pure-Python
    > (PRD §6)" was true but had stopped explaining the backend. The namespace argument is
    > now written once in `dataseq`'s — the member whose comment was most wrong, and the base
    > of the dependency graph — with the other four pointing at it.
    > Two further stale spots found by sweeping for the *claim* rather than the word
    > "hatchling": `align/meson.build` said ninja "returns when that ADR is reverted" (it
    > returns because 0012 was **superseded**, which is a different thing), and the root dev
    > group still explained itself in terms of align/hmm moving *back*.
    > **A mechanical trap worth recording**, since the next such edit will hit it: the old
    > revert recipes contained `build-backend = "mesonpy"` as a *commented* line, so a
    > replacement bounded by that string terminated inside the comment and left a duplicated
    > `requires`/`build-backend` pair below the new block. TOML's last-key-wins made the file
    > still parse, so it would not have failed loudly. Caught by counting `^build-backend`
    > per file.
  - [x] `meson-python`, `cython`, `ninja` back in the root `dev` group, **plus whatever
        makes the loader's baked ninja path stable** — PATH alone is not enough, as
        measured.
    > **Result:** The dev group carries `meson-python`, `cython>=3.0`, `ninja` and also
    > **`numpy>=2.1`** — the fourth is not decoration: it is a *build* requirement of
    > `align`/`hmm`, and with build isolation off it cannot be supplied by
    > `build-system.requires` alone. What stabilises the baked path is
    > `[tool.uv] no-build-isolation-package` listing **all five** members; the comment now
    > says explicitly that omitting one restores isolation for that member alone and its
    > baked path dies the same way. Each dev-group entry now states what needs it.
  - [x] All five members import after a clean `uv sync`; full suite green.
    > **Result:** `rm -rf .venv` then a plain `uv sync` — no `--reinstall`, which was the
    > point, since a reused venv already has ninja on disk and cannot answer the question.
    > uv reported *"Prepared 5 packages without build isolation in 2.06s"*, so the bootstrap
    > ordering resolves on its own: the dev group is installed before the members that need
    > it to build. That ordering is emergent from the resolver rather than stated anywhere in
    > the config, which is why it needed measuring rather than reasoning about.
    > All seven import paths resolve (`dataseq`, `align`, `hmm`, `hseg`, `dl`, `dl.rnn`,
    > `dl.transformer`), **five editable finders on `sys.meta_path`**, and
    > `pfsmgraph.__path__` is the single synthetic
    > `_pfsmgraph_hseg_editable_loader.py/pfsmgraph` entry — the namespace *is* still
    > replaced, and it no longer matters, which is the positive statement of the fix.
    > Suite green at **280**. `uv lock --check` clean.
    > The dev-loop claim that decided the choice was checked rather than assumed: appending a
    > name to `hseg/__init__.py` and importing with **no sync at all** showed it. (The probe
    > was reverted with `git checkout --`; note that a plain `cp` restore silently did
    > nothing here, because the interactive `cp -i` alias prompted and defaulted to "no" —
    > `git checkout --` is the reliable revert for a scratch probe.)

- [ ] Record the resolution as a new ADR superseding 0012, and clear the footnotes it
      planted in `README.md` and `docs/agents/core.md`.
