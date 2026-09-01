# 0014. Imported migration source is retained in `.scratch/`, scoped per package

- **Status:** **Accepted** (2026-09-01) — supersedes the delete-at-merge intent this
  repository was created with; see *Alternatives considered*.
- **Date:** 2026-08-31 (decided), 2026-09-01 (recorded)
- **Source:** — no PRD counterpart. The PRD describes the proof-of-concept
  ([§1.2](../PRD.md)) but says nothing about how source imported *for a migration* is
  held.

## Context

`dataseq` was built by merging three existing implementations
([ADR 0010](0010-dataseq-composition-merging-three-implementations.md)), which required
reading them side by side. They were imported into `.scratch/` — four directories
holding six source trees, 4.8 GB on disk — and the `feat/dataseq-merge` plan's last goal
was to delete the directory once the merge landed.

That premise was wrong, and it was noticed on 2026-08-31 while writing the third
import's `.gitignore`. The same four imports are the migration source for the *next two*
packages as well: `.scratch/hmm-lush/` is what `pfsmgraph-hmm` 0.1.0 is translated from,
and `.scratch/align-poc/tokalign/` is the direct ancestor of `pfsmgraph-align`. Deleting
at the end of the first migration would have thrown away the input to the second and
third.

Two facts make that worse than a re-import would suggest. `hmm-lush` **is under no
version control at all** — it is a personal tree with mtimes spanning 2008–2011 and no
upstream to clone from, so deleting it deletes it. And the reasoning about *what each
tree contributes* was worked out while the tree was in front of us; that reasoning is
recorded in the `.gitignore` headers and would have to be re-derived from scratch by
whoever re-imported.

## Decision

**`.scratch/` is retained across branches as this project's migration-source area.
Each import owns a `.gitignore` beside it, deny-by-default, and the tracked set is
scoped to the package currently being migrated — widening as each migration begins.**

Four rules follow from that and are part of the decision:

- **The tracked set only ever widens.** `.gitignore` has no effect on a file git already
  tracks, so a policy cannot be *narrowed* in any meaningful sense: narrowing requires
  `git rm --cached`, which is a deletion commit and re-opens the retention hazard below.
  A phase advance therefore adds rules or adds nothing; it never removes.
- **Inert rules are written and commented, not omitted.** Where a policy is phased
  (`.scratch/align-poc/.gitignore` is the first), the rules for phases not yet reached
  are kept commented with their reasoning. Advancing a phase is an uncomment, not a
  re-derivation.
- **Nothing outside `.scratch/` may import from it.** No file there is part of any
  distribution. The leading dot matches pytest's default `norecursedirs` entry `.*`, and
  the directory sits outside `packages/`, so neither the test collector nor the uv
  workspace glob ever claims it.
- **An imported tree's agent instructions and VCS metadata are renamed on arrival** —
  `.git` → `.git-disabled`, `CLAUDE.md` → `CLAUDE.md.orig`, and `AGENTS.md` /
  `AGENTS.override.md` → `*.orig`. Presence on disk is the trigger for all three
  hazards, not tracking, so gitignoring them does not help.

[`.scratch/README.md`](../../../.scratch/README.md) holds the operational detail: the
provenance table with each tree's upstream and revision, what is tracked per import and
why, and the recipe below.

## Consequences

### Positive

- **`hmm-lush` is preserved.** It is irreplaceable, and this is the only copy under
  version control anywhere.
- **The next two migrations start with their source already present**, together with the
  reasoning about what each file contributes — written while the tree was readable,
  which is the only time it is cheap to write.
- **Per-import locality means one policy can widen without touching the other three.**
  This was originally justified by the opposite argument (deleting `.scratch/` would
  remove the rules along with what they describe); it survived the reversal and now
  earns its place better than it did before.
- **The retention hazard no longer arises.** A squash merge collapses the commit that
  adds a tree and the commit that deletes it into nothing, silently losing the code from
  `main`. Nothing is deleted, so nothing can be collapsed away.
- **Deny-by-default keeps the cost near zero.** 261 files, 1.6 MB tracked out of 4.8 GB
  on disk — 0.03%. The excluded 4.8 GB is virtualenvs, tool caches, saved model
  checkpoints, and engraved score images.

### Negative / costs

- **261 files that are not this project's code are tracked permanently.** They appear in
  repo-wide greps, in `git log --stat`, and in any tooling that walks the tree. The
  leading dot hides them from pytest and from a casual `ls`, not from `grep -r`.
- **A tracked file is effectively permanent.** Track one by mistake and removing it costs
  a `git rm --cached` — a deletion commit, with the squash-merge hazard back in play for
  that file. The asymmetry is the reason the policies are deny-by-default rather than
  allow-with-exclusions.
- **Four `.gitignore` files carry substantial prose that must be kept true.** They are a
  documentation-rot surface like any other, and one of them is phased, so it has a
  correct state that changes over time.
- **A file written under `.scratch/` without a matching negation is invisible to
  `git status`.** It fails silently: the work looks committed and is not. This has now
  caught out three of the four imports, and nearly cost the `hmm-lush` translation —
  those files were written, verified to run, and were invisible until `git check-ignore`
  was run on them deliberately. Every policy carries a `!/*.md` negation for our own
  writing for exactly this reason.
- **The rename hazards recur on any future import**, and one of them is not obvious: a
  nested `AGENTS.md` from a project using the same agent-docs toolchain is not merely
  another project's instructions but the *generated artefact* of one, and the
  `protect-agent-docs.py` hook matches on filename.

## Alternatives considered

- **Delete `.scratch/` when `feat/dataseq-merge` merges** — the original plan, and the
  reason this record exists. Rejected once it was clear the imports are the migration
  source for `hmm` and `align` too, and decisively so for `hmm-lush`, which has no
  upstream to re-clone.
- **Delete at merge, but keep the code reachable via a tag, a merge commit, or an
  unmerged branch.** This was the retention mechanism the original plan specified, and it
  works — but it makes the code *reachable*, not *present*, and a migration wants the
  tree on disk beside the work. **It remains the correct recipe if `.scratch/` is ever
  deleted for real**, and is recorded as such in `.scratch/README.md`.
- **A separate repository for imported source.** Rejected: it separates the imports from
  the ADRs and plan files that argue from them — several ADRs cite specific files by path
  — and adds a checkout step to every migration for no gain.
- **Keep `.scratch/` entirely untracked.** Rejected: the analysis written *about* the
  imports is our own work and is cited as evidence by
  [0010](0010-dataseq-composition-merging-three-implementations.md) and
  [0011](0011-fixed-reserved-symbol-block-and-strict-encoding.md) — `RESERVED-BLOCK.md`
  is authoritative for the reserved-block comparison table. An untracked working area is
  absent from a fresh clone, which would make those citations unverifiable.
- **Track each import whole.** 4.8 GB, ~99% of it virtualenvs, tool caches and 2008–2011
  model checkpoints that are outputs of the algorithm being translated rather than inputs
  to it.

## Evidence

**Observed.**

- `git ls-files .scratch/ | wc -l` → **261**; those files total **1.6 MB** against
  `du -sh .scratch/` → **4.8 GB**.
- `hmm-lush` carries no VCS metadata; source mtimes span 2008-01-24 – 2011-02-01 and the
  tree was last reorganised 2022-08-26. The other five trees are clean checkouts of live
  GitHub repositories at revisions recorded in `.scratch/README.md`.
- Renames needed on import ranged from **zero** (`hmm-lush`) to **eight**
  (`align-poc`: three nested `.git` directories and five agent-instruction files).
- `tokalign/dev/plugins/workflow-claude/` is mode 555, so renaming its children required
  `chmod u+w` on the parent first. The failure without it is a bare "Permission denied"
  that reads as a sandbox restriction rather than a file mode.
- `uv run pytest` collects **zero** items from `.scratch/` with all of the above present,
  including a runnable Python translation under `.scratch/hmm-lush/translation/`.

**Reasoned.** That the tracked set can only widen is a property of how `.gitignore`
interacts with the index, not a measurement. It is stated here because the
`feat/dataseq-merge` plan asked for the policies to be "narrowed", which is not an
operation git offers on tracked files.

## Open

**When, if ever, `.scratch/` is deleted.** All four imports will have been consumed once
`pfsmgraph-align` 0.1.0 lands, which is the earliest point the question is even askable.
Nothing forces the deletion then — the trees remain the evidence behind ADRs 0001–0004 and
0010–0011 — so the trigger, if there is one, is repository size becoming a real cost
rather than a theoretical one. If it happens, the recipe in *Alternatives considered*
applies and this record is superseded rather than amended.
