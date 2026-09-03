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

**Related documents.** Decided-but-not-yet-actionable work is *not* listed here; it lives
in [`DEFERRED.md`](DEFERRED.md), indexed by the trigger that unblocks it. Open design
questions live in [PRD §8](../design/PRD.md) and the `Open` sections of the
[ADRs](../design/adr/README.md). This file tracks work that is active now, plus the
revisions already drafted and waiting to be opened.

## Planned revisions

Drafted ahead of being opened, one directory each, in the same shape `/close-revision`
leaves behind — so opening one is a splice from its `_TODO.md` rather than an authoring
job. The detail lives in those files and not here, which is the point: this file stays
short enough to read on every session, and a revision's subgoals enter it only while that
revision is in progress.

The `hmm` migration is **three** releases rather than one. The Lush trainer is 1,102 lines
spanning three problems that fail differently — a decode, a fixed-topology estimator, and
a search over model shapes — and each raises its own questions about parallelism and data
structures. Conflating them would put the project's first `.pyx`, its first EM loop and its
first resizing search in one revision, where a failure in any of them would be diagnosed
against all three.

- **Revision 02-hmm-v0.1.0** — Viterbi path and the `dataseq` interface. Carries the
  project's firsts: first DP kernel, first `.pyx`, first non-empty ADR 0003 backend
  matrix, and the meson-python namespace resolution that ADR 0012 stands down.
  See [`02-hmm-v0.1.0/_TODO.md`](02-hmm-v0.1.0/_TODO.md).
- **Revision 03-hmm-v0.2.0** — Baum-Welch on a fixed topology, with an optional `torch`
  backend whose autograd E-step is held against the numpy reference's explicit
  forward-backward. See [`03-hmm-v0.2.0/_TODO.md`](03-hmm-v0.2.0/_TODO.md).
- **Revision 04-hmm-v0.3.0** — topology search by state merge and split, scored by
  minimum description length. See [`04-hmm-v0.3.0/_TODO.md`](04-hmm-v0.3.0/_TODO.md).

All three were drafted from a structural survey of `.scratch/hmm-lush/Code/HMMlib/` —
definition maps and call-site counts — **before the source was read**. Each names the
findings that would falsify its boundaries, and revision 02's first subgoal is the reading
that checks them.

## Closed revisions

Extracted by `/close-revision` once finished. The subgoals, their `> **Done:**` records,
and the branch plans that executed them all live in the revision's directory.

- **Revision 01-dataseq-v0.1.0** — closed. See `docs/plan/01-dataseq-v0.1.0/_TODO.md`.
