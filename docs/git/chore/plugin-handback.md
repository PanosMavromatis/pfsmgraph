# chore/plugin-handback

**Created**: 2026-09-09
**Base**: main at e384832
**Status**: active

## Purpose

Land the seven pfsmgraph edits collected while `chore/revise-plugins` revised the
`dp-compile` and `workflow-claude` plugins (PR #19). That branch's first ground rule
forbade it from editing pfsmgraph, so the edits were recorded there as a handoff manifest
and cut into this branch's plan at its merge. They are independent of one another and none
of them blocks anything today — but **two are corrections rather than additions**, and both
correct claims that ADR 0002 phase 2 and phase 3 would otherwise be planned against, which
is why this is scheduled ahead of phase 2 rather than alongside the release.

## Scope

The seven goals are in the branch plan, with their measurements and reasoning. In short:

- Close `DEFERRED.md`'s "how the development plugin fits the multi-package family" entry.
- **Correction.** ADR 0016's phase-3 naming Open section, and the claim it rests on: it
  cites a `_cuda.py` in `.scratch/align-poc/tokalign` that does not exist, so its stated
  reason is void rather than unmet.
- Land the recovered Viterbi `FORMALIZATION.md`, regenerated rather than reused.
- Two `pfsmgraph-hmm` test gaps at the validation boundary — TC-19 (a code exactly equal to
  the vocabulary size) and TC-20 (impossibility at position 0).
- **Correction.** Master plan lines around the phase 2-4 items: the anti-diagonal subgoal
  has an answer already, and it is that the Viterbi recurrence has no anti-diagonals at all.
- The marketplace note in both plugin READMEs.
- Re-scope, do not delete, pfsmgraph's `/smart-commit` override note in
  `docs/agents/claude.md`.

## Context

- **PR #19** — the plugin revision this hands back from. Its plan closed at 33 of 33 across
  sections A-G and is on `main` as the record, archived at
  `docs/plan/chore-revise-plugins/_TODO.md` (it ran from `tmp/TODO.md` and was moved in
  afterwards).
- **`workflow-claude` PR #21** — the commit-convention fix, which is what makes the
  `/smart-commit` override note redundant as a *correction* and worth keeping as a
  *statement*.
- The plan for this branch was written and committed **before** the branch existed, on
  `chore/revise-plugins`, so `/new-branch` did not create it. Pre-filing is safe for
  `/new-branch`, which does not check for an existing plan directory; it would not be for
  `/open-revision`, which refuses a label whose directory exists.

## Notes

- Two goals touch `docs/agents/` — the ADR correction does not, but the `/smart-commit`
  override re-scoping edits `docs/agents/claude.md` directly. Use `/smart-commit` for
  anything in that pass so `/agents-docs-update` runs; `claude.md` is a *source* and needs
  no rebuild, but a `core.md` or `codex.md` edit would.
- The `FORMALIZATION.md` goal writes into a path the `dp-compile` manifest names, so it is
  also the first real exercise of the revised plugin against this repository.
