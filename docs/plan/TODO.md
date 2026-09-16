# Master plan — pfsmgraph

**Status**: active

The two-tier plan convention for this repository:

- **Master plan** — this file, `docs/plan/TODO.md`, lives on `main`. It defines
  **revisions** (milestones) and their **subgoals**; each subgoal spawns a branch.
- **Branch plan** — `docs/plan/<type>-<slug>/TODO.md`, created by `/new-branch`, worked
  by `/hitl-step`, stamped `merged` by `/smart-merge`. It **survives on `main`** as the
  durable record of how a subgoal was executed, and is filed into its revision's
  directory by `/file-plans`.
- **PR body** — the distilled description plus a pointer to the branch plan directory.
  Not a verbatim archive.

`TODO.md` rather than `DO.md` is deliberate: branches inherit the master plan's model,
so every subgoal here runs under `/hitl-step` with its Q&A logged inline. The `dataseq`
merge reconciles three existing implementations and settles a public API that four other
packages depend on — the questions are genuinely open, and the answers are worth keeping.

**Related documents.** Decided-but-not-yet-actionable work is _not_ listed here; it lives
in [`DEFERRED.md`](DEFERRED.md), indexed by the trigger that unblocks it. Open design
questions live in [PRD §8](../design/PRD.md) and the `Open` sections of the
[ADRs](../design/adr/README.md). This file tracks work that is active now, plus the
revisions already drafted and waiting to be opened.

## Planned revisions

Drafted ahead of being opened, one file each under [`planned/`](planned/), carrying the
same `## Subgoals` section `/close-revision` archives — so opening one is a splice rather
than an authoring job. They sit in `planned/` rather than in `docs/plan/<label>/` because
`/open-revision` refuses a label whose directory already exists, on the sound assumption
that only `/file-plans` creates one; a draft filed there would make every planned revision
look already-opened. The detail lives in those files and not here, which is the point: this file stays
short enough to read on every session, and a revision's subgoals enter it only while that
revision is in progress.

The `hmm` migration is **three** releases rather than one. The Lush trainer is 1,102 lines
spanning three problems that fail differently — a decode, a fixed-topology estimator, and
a search over model shapes — and each raises its own questions about parallelism and data
structures. Conflating them would put the project's first `.pyx`, its first EM loop and its
first resizing search in one revision, where a failure in any of them would be diagnosed
against all three.

The first two of the three are **closed** (see below), and `pfsmgraph-hmm` is released at
0.2.0; the third, `04-hmm-v0.3.0`, is still planned, so no revision is currently open. Each
entry moves up as its revision is opened.

- **Revision 04-hmm-v0.3.0** — topology search by state merge and split, scored by
  minimum description length. See [`planned/04-hmm-v0.3.0.md`](planned/04-hmm-v0.3.0.md).

All three were drafted from a structural survey of `.scratch/hmm-lush/Code/HMMlib/` —
definition maps and call-site counts — **before the source was read**. Each names the
findings that would falsify its boundaries, and revision 02's first subgoal is the reading
that checks them.

## Closed revisions

Extracted by `/close-revision` once finished. The subgoals, their `> **Done:**` records,
and the branch plans that executed them all live in the revision's directory.

- **Revision 01-dataseq-v0.1.0** — closed. See `docs/plan/01-dataseq-v0.1.0/_TODO.md`.
- **Revision 02-hmm-v0.1.0** — closed. See `docs/plan/02-hmm-v0.1.0/_TODO.md`.
- **Revision 03-hmm-v0.2.0** — closed. See `docs/plan/03-hmm-v0.2.0/_TODO.md`.
