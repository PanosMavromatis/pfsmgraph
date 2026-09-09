# chore/plugin-handback

**Status**: active
**Created**: 2026-09-09
**Subgoal**: Land the hand-back from the plugin revision — registered in
[`docs/plan/TODO.md`](../TODO.md) under revision 02-hmm-v0.1.0.

## Where this came from

These seven goals were Section H of `tmp/TODO.md`, the plan for `chore/revise-plugins`,
which revised the `dp-compile` and `workflow-claude` plugins in two clones parked under the
gitignored `tmp/`. That plan's own work — sections A through G — is complete at 33 of 33.
Section H was never part of it: every item edits **pfsmgraph**, and that plan's first ground
rule limited it to `tmp/TODO.md` plus a two-file manifest exception. The items were
collected there as a handoff manifest and cut here at merge time.

**The cut is why this file exists rather than a pointer.** Left in place, seven permanently
open items would sit inside the archived record of a finished plan, making it read as
incomplete forever. That is the same failure the source plan already fixed once, one level
down: Section H was itself created on 2026-09-08 by moving these items *out* of Section F,
because a structurally uncloseable goal inside a work section made the section-completion
check permanently red for a reason that was not incomplete work. The fix that worked at
section level is the same fix at file level.

**The branch does not exist yet.** This plan is filed ahead of it, so `/hitl-step` can find
it by rung 4 today and by rung 2 — the branch-plan rung — once `chore/plugin-handback` is
opened. Pre-filing is safe for `/new-branch`, which creates a plan directory without
checking whether one is there; it would **not** be safe for `/open-revision`, which refuses
any label whose directory already exists. That asymmetry is why this is a branch plan and
not a `docs/plan/planned/` draft: `planned/` holds *revisions*, whose subgoals each spawn a
branch, and is consumed by `/open-revision`. This is one branch's work.

**The directory name fixes the branch name.** Resolution flattens `/` to `-`, so the branch
must be `chore/plugin-handback` for rung 2 to match.

## Goals

- [ ] Close `DEFERRED.md`'s "how the development plugin fits the multi-package family"
      entry: one generic plugin, per-repo manifest, this plan as the record.
- [ ] **ADR 0016's Open section on phase-3 naming: close it, and correct the claim it
      rests on.** The record says its proposal mirrors "the existing `_python.py` /
      `_cython.pyx` / `_cuda.py` per-phase naming seen in `.scratch/align-poc/tokalign`".
      That repository has only `_python.py` and `_cython.pyx`; there is no `_cuda.py` and
      no `_numba.py`, and its third phase was never implemented under any name. So the
      deferral's stated reason is void, not merely unmet — settle
      `_<algorithm>_cpu_parallel.py` per A5, matching `meson.build`'s existing
      `_viterbi_cython` pattern, and fix the factual claim in the same edit rather than
      leaving a closed decision resting on a wrong sentence.
- [ ] Land the recovered Viterbi `FORMALIZATION.md` from D4 where the manifest says.
      > **Note:** F2's read-only real-tree run drafted this document in full and was
      > correctly **denied the write** — nothing reached disk, and phase 0 is not done. It
      > verified every line citation the kernel's docstrings claim
      > (`hmm-trainer.lsh:216-218`, `:219`, `:197-198`, `:712-727`, `util.lsh:431-444`) and
      > measured rather than quoted: oracle agreement **3806/3807** across three models,
      > divergence at position 0 of `m008_0001_008` and nowhere else, seed margin
      > **0.004218 bits**, and **0 exact ties in 3804** recurrence positions. Regenerate
      > rather than reuse, but these numbers are checkable against it.
- [ ] **Two test-coverage gaps in `pfsmgraph-hmm`, found by F2's live run.** Both sit at the
      validation boundary rather than in the recurrence, which is why the 165-test suite
      misses them.
      > **TC-19 — a code exactly equal to the vocabulary size.** Existing tests use `99` and
      > `-5`, both far from the edge, so a guard written `>` where it should be `>=` passes
      > the entire suite while letting the boundary code through to the emission lookup.
      > Verified as correct today (`A = 8`; code 7 decodes, code 8 raises) — but nothing
      > pins it, so the next edit is free to break it silently.
      > **TC-20 — impossibility at position 0.** Every existing impossibility test places the
      > dead symbol at position 1, so `ImpossibleSequenceError`'s position report is never
      > checked at its own lower boundary.
- [ ] Master plan lines 192–196 (phases 2–4 of Viterbi): note they run under
      `dp-compile`, so the first real use of the revised plugin is on the kernel it was
      revised for.
      > **Note: phase 3 has no anti-diagonal, and F2's run says so with a reason.** The
      > Viterbi recurrence is 1-D over time with dense S×S coupling, so there are **no
      > anti-diagonals at all** — the decomposition ADR 0016 names is not merely unchosen
      > here, it does not exist for this kernel. ADR 0016's own Open section anticipated
      > this. Carry "undetermined" forward as the honest answer rather than letting phase 3
      > discover that an invented decomposition is a race.
- [ ] Marketplace: nothing to build yet — **verified 2026-09-07, no `marketplace.json`
      exists in any of the user's repositories**, and `claude-plugin-tools` is unrelated.
      Note in both READMEs that the `--plugin-dir` line is interim.
- [ ] **Re-scope pfsmgraph's `/smart-commit` override note, do not delete it** (from G3).
      `docs/agents/claude.md` carries a standing correction of `/smart-commit` Step 3:
      *"This repository does not use them ... the history is the authority, not the
      command's template."* G2 makes the command say what that note has been saying, so the
      note stops being a correction — but it should survive as a **statement of pfsmgraph's
      own convention**, which is still worth writing down and is now what the plugin will
      go looking for.
      > **Do not touch the paragraphs above it.** The `VERSION`-file rule and the
      > release-tagging note in that same section are untouched by G2 and remain true; only
      > the Step 3 paragraph changes meaning. Re-scoping is a smaller edit than it looks —
      > the sentences stay, their framing shifts from "the command is wrong here" to "this
      > is what the command will detect".
      >
      > **There is a second edit to make in the same pass, and it is the more useful one.**
      > pfsmgraph's history is *uncontaminated* — zero conventional-prefixed commits in the
      > whole repository — precisely because every operator overrode the templates by hand
      > every time. That is worth one sentence, because it is what makes the detection rule
      > trivially correct there and it explains why the override note existed at all.
      > Contrast `workflow-claude`, where the same templates polluted the history 19 times
      > in 40 commits. Same plugin, opposite outcomes, decided entirely by whether the
      > consumer noticed.
