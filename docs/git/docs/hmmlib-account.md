# docs/hmmlib-account

**Created**: 2026-09-03
**Base**: main at 478d9d7
**Status**: active

## Purpose

Read `.scratch/hmm-lush/Code/HMMlib/` — 2,044 lines across four files — and write
`.scratch/hmm-lush/HMMLIB-ACCOUNT.md` describing what that code *is*, following the
conventions `ACCOUNT.md` established for the container half. This is subgoal 1 of revision
`02-hmm-v0.1.0`, and it is the reading that revision 02 was explicitly drafted *without*.

Every boundary in revisions 02, 03 and 04 rests on a structural survey — definition maps,
call-site counts, comment headers — not on the source. The master plan says so in as many
words and names three findings that would falsify those boundaries. So this branch has two
duties and they are not the same: produce the account, and check the falsifiers against it.
A held falsifier is a success of the method, not a setback, and revising the master plan is
in scope here rather than deferred to whoever discovers it later.

**No code lands under `packages/pfsmgraph-hmm/src/`.** The account is a reading aid under
`.scratch/`, which belongs to no distribution and which nothing outside it may import.
Settling the public surface of `pfsmgraph.hmm` is subgoal 2 and deliberately downstream of
this one — an API designed against a survey would be designed against a guess.

## Scope

- Read `hmm.lsh` (319), `hmm-param.lsh` (386), `hmm-trainer.lsh` (1102) and `hmm-trainer-view.lsh` (237) in their own terms
- Write `HMMLIB-ACCOUNT.md` to `ACCOUNT.md`'s conventions: a **Sources** block, quantitative claims measured against the two tracked specimen corpora with an appendix, and **provenance unknown** wherever the code admits a behaviour that may never have been exercised
- Check the three falsifiers the master plan names, and record each verdict — including the negative ones, which are findings too
- Revise `docs/plan/TODO.md` if any falsifier holds, and say in the plan why the boundary moved

## Context

- `docs/plan/TODO.md` — `## Subgoals — revision 02-hmm-v0.1.0`; this branch executes its first item and states the three falsifiers to check
- `docs/plan/docs-hmm-migration-plan/TODO.md` — the planning branch that drafted revision 02 and recorded, under its own first goal, that the reading was deliberately left out
- `.scratch/hmm-lush/ACCOUNT.md` — the template. Its framing is the part that matters: an account written with the merge in view "tends to describe the original as a deviation from the thing it predates by fifteen years"
- `.scratch/hmm-lush/COMPARISON.md` and `translation/` — the other two reading aids, both scoped to `Code/SeqData/`. Neither has an `HMMlib` counterpart, and neither is owed by this subgoal
- `.scratch/hmm-lush/Training/set01z0/set01z0_100.sds/` and `Training/set11a_dInt/set11a_dInt.sds/` — the two tracked specimen corpora every measurement in `ACCOUNT.md` was made against; both are tracked in full
- `.scratch/hmm-lush/Code/Utility/util.lsh` (574) and `mc.lsh` (47) — read only as far as `HMMlib` calls into them; subgoal 3 migrates them
- [ADR 0014](../../design/adr/0014-scratch-retention-and-per-package-scoping.md) — why `.scratch/` is retained across branches, and why `hmm-lush`'s policy needs no widening: it is already scoped to the whole live library
- `.scratch/hmm-lush/.gitignore:42` — the `!/*.md` negation that makes `HMMLIB-ACCOUNT.md` visible to `git status`. Verified with `git check-ignore -v` before this branch was created, because the failure mode is silence rather than an error

## Notes

**2026-09-03 — the branch took on work its Scope does not list, and deliberately.** A
design handoff from an earlier conversation (2026-06-13) surfaced in an untracked `tmp/`
directory while goal 1 was closing. It independently confirms the account's central
finding — the model is arc-emission (Mealy) — as *intent* with a lineage, where the
account could only establish it as behaviour. Two independent derivations agreeing is
what promoted that finding from a plan note to [ADR
0015](../../design/adr/0015-arc-emission-mealy-formulation.md), which is the first record
in this repository about a *model* rather than about packaging, tooling or process.

Four things follow, all committed here rather than deferred, because the handoff was
sitting in a gitignored directory and existed on one machine only:

- The handoff moved to `docs/design/arc-emission-hmm-handoff.md`, verbatim under a
  provenance preamble that marks each of its claims confirmed, unchecked, or —
  in one case — **contradicted by the source**: it presumes epsilon machinery that a
  search of `Code/HMMlib/` and `Code/Utility/` shows does not exist.
- Its alignment-derived seed is the mechanism `core.md`'s "alignment is a training
  accelerant for HMM topology search" has been standing on since the PRD. Recorded in
  `DEFERRED.md` under a **trigger** rather than a revision number, per the decision that
  it follows all three planned `hmm` revisions *and* substantial `align` work.
- Two `planned/` drafts amended while amending them is still free: revision 04's resize
  subgoal had two representation options where there are three, and revision 03 gained an
  external-oracle subgoal that catches a class of error its `torch` equivalence test
  structurally cannot.
- The two-part versus refined/NML question registered at PRD §8 — *not* in `DEFERRED.md`,
  which routes open questions there by its own header.

**What this branch's remaining goals inherit.** Goal 2's falsifier verdicts are untouched
and still owed. Goal 3 is partly discharged by the above, but only partly: the account's
consequences for subgoal 2's public surface and subgoal 3's Utility migration are still
unwritten.
