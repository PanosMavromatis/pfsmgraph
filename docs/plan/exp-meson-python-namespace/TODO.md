# exp/meson-python-namespace

**Status**: active
**Created**: 2026-09-04
**Subgoal**: Resolve the meson-python namespace shadowing and move `hmm` off hatchling,
reverting ADR 0012 by whichever of its three candidates survives contact.

## Goals

- [ ] Reproduce the shadowing against today's workspace, so the revert is measured rather
      than remembered.
  - [ ] Apply the revert recipe to `hmm` alone and record exactly which imports fail.
  - [ ] Apply it to both members and confirm the second, independent failure ADR 0012
        records — two meson-python editable installs conflicting with each other.
- [ ] Repair the `meson.build` files before evaluating anything.
  - [ ] Add `_numeric.py`, `_params.py`, `_viterbi.py` to `hmm`'s `install_sources`, and
        decide how to keep that list from silently drifting again.
  - [ ] Retarget the dormant extension block: the first `.pyx` is Viterbi, not
        Baum-Welch.
- [ ] Settle the sequencing question — throwaway extension here, or interleave with the
      Viterbi `.pyx` from the next master-plan goal.
- [ ] Evaluate the three candidates against a member that actually compiles, and choose.
- [ ] Land the choice.
  - [ ] Revert recipe applied to both members; `[tool.hatch.*]` dropped.
  - [ ] `meson-python`, `cython`, `ninja` back in the root `dev` group; `ninja` on PATH.
  - [ ] All five members import after a clean `uv sync`; full suite green.
- [ ] Record the resolution as a new ADR superseding 0012, and clear the footnotes it
      planted in `README.md` and `docs/agents/core.md`.
