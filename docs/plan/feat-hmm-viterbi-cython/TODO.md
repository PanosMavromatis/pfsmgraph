# feat/hmm-viterbi-cython

**Status**: active
**Created**: 2026-09-09
**Subgoal**: Implement Viterbi at ADR 0002 phase 2 (Cython), the first `.pyx` in
a distribution — `docs/plan/TODO.md`, revision 02-hmm-v0.1.0

## How this branch is driven

Phases 2, 3 and 4 run under the `dp-compile` plugin, so this is the plugin's first
real use on the kernel it was revised for. Two commands and one skill are in scope:

| Invocation | Role |
|---|---|
| `/dp-compile:phase-check viterbi` | Reporting only. Never advances, never suggests advancing. |
| `/dp-compile:next-phase viterbi` | Resolves the target and routes to the skill for **the target's own phase**. |
| `dp-compile:cython-translation` | The phase-2 skill `/next-phase` routes to. Not invoked directly — see below. |

**Enter through `/next-phase`, not by invoking `dp-compile:cython-translation`
directly.** The two are not equivalent here. `/next-phase` includes
`@references/phase-detection.md`, which carries the recovered-graph rules; the skill's
own prerequisite 2 does not, and defines phase-1 staleness as *"the hash recorded in
its `derived-from` header no longer matches the formalization's current hash"* — a
header this kernel correctly does not have. Routing through `/next-phase` is what
supplies the missing carve-out.

**Phase 0 already exists, and `algorithm-recover` has already run.** This Viterbi did
not begin in a natural-language specification: it was translated from Lush
(`update-viterbi-path`, `hmm-trainer.lsh:216-227`), so `algorithm-formalize` never ran
and never could have. The recovery entrance was taken instead, on 2026-09-09 in PR #20.
Verified on this branch before any of the goals below were written:

- `docs/design/algorithms/viterbi/FORMALIZATION.md` exists, and its Metadata table
  reads `Source: **Recovered from the phase-1 implementation.**`
- Its `Derived from` row names `_viterbi.py` at
  `sha256:48666ca8a3126ba141ec1a0ddba8ed1823e502f93d72483181906afd24135742`, which
  **matches** the current file. The document is `fresh`, so phase 0 is not a target and
  `algorithm-recover` does not need re-running.
- `_viterbi.py` carries **no** `# dp-compile: derived-from` line, which is correct
  rather than missing. In a recovered graph the kernel is the root: it derives from
  nothing the plugin tracks, and `phase-detection.md` forbids the mtime fallback giving
  it an incoming edge from the document that was written from it — that would report
  the kernel stale the moment a recovery lands and regenerate it from a description of
  itself.

So the graph branches at the root rather than running as a chain, and the expected
`/phase-check` report is the last row of `phase-detection.md`'s worked-examples table:
nominal phase **1**, `formalization` fresh, `cython` / `cpu_parallel` / `cuda` absent,
target **write `cython`**.

## Goals

- [ ] Confirm the plugin's view of the algorithm matches this branch's premise
  - [ ] Run `/dp-compile:phase-check viterbi` and record the report verbatim. Expect
        nominal 1 and target *write cython*; anything else means the premise above is
        wrong and the branch stops here rather than proceeding on it
  - [ ] Note whether the report names `_viterbi.py` under "any provenance header that
        was missing". It should — the header genuinely is absent — but the mtime
        fallback must **not** be reported as in use for it, since the recorded edge
        from the formalization already runs the other way
  - [ ] Confirm the report says which entrance produced phase 0. `/phase-check`
        requires this, and "recovered" is the answer a reader needs in order to weigh
        how far the document can be trusted: it was read off the code it now governs
- [ ] Settle what this branch may claim about equivalence
  - [ ] Reconcile the subgoal's "enforced by the parameterized suite" with `core.md`'s
        finding that ADR 0003's parameterization cannot be in force until `align`;
        whichever is wrong is corrected on `main`, not worked around here
  - [ ] Triage the three `first .pyx` DEFERRED items: close the meson-python revert
        (spent by ADR 0018), schedule the tokalign comma-form indexing fix here, leave
        the `numba-cuda` pin to phase 4
  - [ ] Decide whether the `cython-translation` prerequisite gap earns a `DEFERRED.md`
        trigger of its own. There is no `dp-compile` trigger section yet, only a
        `workflow-claude` one
- [ ] Advance to phase 2 via `/dp-compile:next-phase viterbi`
  - [ ] Expect its step-3 oracle gate to pass: all three `.vpath.xls` files the
        manifest declares are exercised by `test_viterbi.py` through `load_vpath`.
        The gate matters most for exactly this algorithm — a recovered formalization
        and its extracted test cases both came from the kernel, so the oracles are
        the only evidence left that could disagree with it
  - [ ] Let it route to `dp-compile:cython-translation`; the `.pyx` is written against
        the phase-0 specification, whose min-plus objective and smallest-index
        tie-break are contract rather than description
  - [ ] Apply the deferred comma-form indexing fix to the tokalign template **before**
        it is used as the reference template. That file is deliberately untracked, so
        this goal closes with no diff to point at — record it as a note
  - [ ] Confirm the emitted `.pyx` carries `# dp-compile: derived-from` naming
        `_viterbi.py` with its current hash. The kernel is headerless as a root; the
        `.pyx` is not a root and must record its edge
- [ ] Make it build, install and import
  - [ ] `packages/pfsmgraph-hmm/meson.build`: un-dormant the extension block and name
        every new source in `install_sources` — meson does not glob, and
        `tests/test_meson_sources.py` is what catches the omission
  - [ ] Add the backend row to `_backends.py`; a backend implemented but not importable
        is a hard failure under ADR 0003, never a skip
  - [ ] `dp-compile`'s `PreToolUse` hook arms on this path from the first commit that
        stages it — `[commands] build` and `test` run before `git commit` and block on
        failure. Expect it inside `/smart-commit`; that is the design, not a leak
- [ ] Establish equivalence
  - [ ] Against phase 1 on constructed inputs, including the exact-tie case the learned
        fixtures cannot exhibit — there are 0 exact ties in 3804 oracle positions, so a
        differential suite alone would accept a last-wins port
  - [ ] Against the three oracles, one of which differs at position 0 by design: the
        δ-seeding defect is fixed rather than reproduced
- [ ] Update whatever the change makes stale
  - [ ] `core.md`: the backend matrix stops having one row, and the four-phase lifecycle
        gains its first compiled occupant
  - [ ] `claude.md` only if something names a Claude Code feature; the `dp-compile`
        section already describes the hook and the manifest
  - [ ] Neither is edited directly for `core.md` — `/agents-docs-build` regenerates the
        artifacts, and `/smart-commit` runs `/agents-docs-update` ahead of the commit
