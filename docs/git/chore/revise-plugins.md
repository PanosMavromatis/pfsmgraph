# chore/revise-plugins

**Created**: 2026-09-07
**Base**: main at beb0637
**Status**: active

## Purpose

Revise the two Claude Code plugins this project depends on so they fit pfsmgraph rather
than the proof-of-concept they were written against. `tokalign-dev` was built for the
`tokalign` alignment library under ADR 0002's three-phase lifecycle; it is unaware of
[ADR 0016](../../design/adr/0016-numba-cpu-parallel-phase.md)'s amendment, tightly coupled
to a repository layout pfsmgraph does not share, and carries assumptions that would be
*wrong* here rather than merely narrow. It gets renamed — `dp-compile`, the emphasis
staying on dynamic-programming kernels — generalised to any repository doing a progressive
pseudocode → Python → Cython → parallel translation, taught the four-phase chain, and given
a second entrance for algorithms that begin in a legacy implementation instead of in prose.
`workflow-claude` is checked alongside it so the two interoperate without duplicating each
other.

**Almost none of that lands here.** Both plugins are separate repositories, parked as
clones under the gitignored `tmp/`, and every plugin edit is committed from that plugin's
own root. What this branch contributes to `main` is the record: the plan, its inline Q&A,
and the hand-backs the revised plugin needs *from* pfsmgraph.

## Scope

- Rename and generalise `tokalign-dev` → `dp-compile`: strip the `src/tokalign/algorithms/`
  layout, the `_types.py` trio, `_registry.py`, `setup.py build_ext` and the pre-ADR-0011
  reserved block, replacing each with a per-repository manifest.
- Implement ADR 0016's chain end to end: five stages, a new `cpu-parallelization` skill for
  phase 3, and a real `gpu-parallelization` for phase 4 — currently a ten-line stub over
  zero-byte references.
- Add a legacy-code entrance: recover a `FORMALIZATION.md` *from* an existing
  implementation, so `hmm`'s Lush-descended kernels get a pseudocode reference they never
  had. Dry-run against `_viterbi.py`.
- Verify `dp-compile` and `workflow-claude` interoperate: presence check, delegation
  without duplication, hook coexistence, and the combined workflow documented in
  `dp-compile`'s README.
- Collect the pfsmgraph-side hand-backs rather than landing them piecemeal.

## Context

- **The plan is `tmp/TODO.md`**, tracked from this root as the only file under `tmp/`
  (`.gitignore` reads `tmp/*` with a `!tmp/TODO.md` negation). Work it with
  `/hitl-step N tmp/TODO.md` — **always pass the path**, since nothing under `docs/plan/`
  resolves to it. Six sections: decisions first, then rename/generalise, the ADR 0016
  chain, the legacy-code entrance, interop, and verification with a hand-back list.
- **Three decisions are already settled** (2026-09-07, logged in the plan): the clones'
  nested `CLAUDE.md` files stay in place, this branch gets a pointer plan rather than a
  real one, and pfsmgraph's untracked `workflow-claude` copy becomes a symlink into `tmp/`.
- **Standalone — no master-plan subgoal.** This is meta-work interpolated before revision
  02 resumes at Viterbi phase 2, and it belongs to no revision's subgoal list. Consequence
  worth knowing at merge: `/file-plans` derives a plan's revision from its master-plan
  backlink, so with none it reports `no backlink` and leaves the plan flat. That is the
  correct outcome, not a defect.
- **Prior art**: [ADR 0016](../../design/adr/0016-numba-cpu-parallel-phase.md) (the phase
  this plugin has never heard of), [ADR 0002](../../design/adr/0002-three-phase-algorithm-lifecycle.md),
  [ADR 0003](../../design/adr/0003-one-parameterized-test-suite-per-algorithm.md) (the
  backend matrix a phase skill must register into), and `DEFERRED.md`'s "no trigger yet"
  entry asking how the development plugin fits a multi-package family — which this branch
  answers.

## Notes

- 2026-09-07 — Branch opened from `beb0637`. Plan and its three settled decisions already
  on `main` (`84b2edb`, `beb0637`), so this branch starts with its planning done.
- Two assumptions in the old plugin would be **wrong** for `hmm`, not merely narrow, and
  both are called out in the plan: "normalize to max (similarity)" inverts a min-sum-over-
  bits Viterbi, and the anti-diagonal wavefront as *the* parallel decomposition is exactly
  what the master plan leaves open for this kernel.
- The plugin's mtime staleness runs backwards for a formalization recovered from code —
  the recovered document is necessarily newer than its own source, so phase 1 would read
  as stale and be regenerated from its own derivative. Replacing it is goal A3.
