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

- [ ] Settle the sequencing question — throwaway extension here, or interleave with the
      Viterbi `.pyx` from the next master-plan goal.

- [ ] Evaluate the three **live** candidates against a member that actually compiles, and
      choose.
  - [ ] **Candidate 1 — non-editable install of the compiled members.** No finder exists
        at all, so nothing is shadowed; costs a manual reinstall step in the dev loop,
        close to the setuptools status quo ante.
  - [ ] **Candidate 4 — all five members on meson-python.** Every member gets a finder and
        they chain, which is what this branch measured. Not in ADR 0012. Cuts against
        ADR 0008's per-package backends, and hands three pure-Python members a compiled
        member's build backend for no compiled code.
  - [ ] **Candidate 3 — an upstream fix**: a finder that defers to other finders for
        prefixes it does not own. Now known to be a design change rather than a bug
        report, since the `{'pfsmgraph'}` claim is derived structurally from the PEP 420
        layout rather than chosen.
  > **Note:** ADR 0012's candidate 2 — a single combined compiled distribution, "so only
  > one meson-python finder ever exists" — is **refuted**, not merely deprioritised. Its
  > premise was the two-finder conflict, which this branch could not reproduce. One finder
  > is not the problem; any finder is, so a combined compiled distribution would still
  > shadow `dataseq`, `hseg` and `dl`.

- [ ] Land the choice.
  - [ ] Revert recipe applied to both members; `[tool.hatch.*]` dropped.
  - [ ] `meson-python`, `cython`, `ninja` back in the root `dev` group, **plus whatever
        makes the loader's baked ninja path stable** — PATH alone is not enough, as
        measured.
  - [ ] All five members import after a clean `uv sync`; full suite green.

- [ ] Record the resolution as a new ADR superseding 0012, and clear the footnotes it
      planted in `README.md` and `docs/agents/core.md`.
