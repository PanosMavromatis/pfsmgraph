# chore/plugin-handback

**Status**: active
**Created**: 2026-09-09
**Subgoal**: Land the hand-back from the plugin revision — registered in
[`docs/plan/TODO.md`](../TODO.md) under revision 02-hmm-v0.1.0.

## Where this came from

These seven goals were Section H of the plan for `chore/revise-plugins`, which revised the
`dp-compile` and `workflow-claude` plugins in two clones parked under the gitignored `tmp/`.
That plan lived at `tmp/TODO.md` while the work ran and is now archived at
[`../chore-revise-plugins/_TODO.md`](../chore-revise-plugins/_TODO.md); its own work —
sections A through G — is complete at 33 of 33. Section H was never part of it: every item
edits **pfsmgraph**, and that plan's first ground rule limited it to its own file plus a
two-file manifest exception. The items were
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

- [x] Close `DEFERRED.md`'s "how the development plugin fits the multi-package family"
      entry: one generic plugin, per-repo manifest, this plan as the record.
  > **Done:** closed in place per that file's own convention — the entry is never deleted,
  > it gains a bolded `**Settled (date) … and the entry is closed.**` paragraph. Eight of
  > its thirty-five entries are now closed.
  > **Note:** the entry posed a **false dichotomy**, and the closure says so rather than
  > picking the nearer horn. "One family-wide dev plugin, or one per package" assumed the
  > plugin would know the family either way; what shipped makes it know *nothing* about
  > the family, pushing every repository-specific fact into `dp-compile.toml`. Per-package
  > would have duplicated one lifecycle five times and drifted; family-wide-in-the-sense-of-
  > knowing-this-family is the exact defect the revision removed, since the plugin began as
  > `tokalign-dev` built around one repository's layout.
  > **Note:** **PRD §10 was deliberately left alone**, and the closure records why. It says
  > the question "was not discussed" — a true claim about that document's scope, in a
  > section titled "Out of scope for this document". A reader seeing a cited section that
  > looks stale would otherwise "fix" it, turning a scope declaration into a live-questions
  > list.
- [x] **ADR 0016's Open section on phase-3 naming: close it, and correct the claim it
      rests on.** The record says its proposal mirrors "the existing `_python.py` /
      `_cython.pyx` / `_cuda.py` per-phase naming seen in `.scratch/align-poc/tokalign`".
      That repository has only `_python.py` and `_cython.pyx`; there is no `_cuda.py` and
      no `_numba.py`, and its third phase was never implemented under any name. So the
      deferral's stated reason is void, not merely unmet — settle
      `_<algorithm>_cpu_parallel.py` per A5, matching `meson.build`'s existing
      `_viterbi_cython` pattern, and fix the factual claim in the same edit rather than
      leaving a closed decision resting on a wrong sentence.
  > **Done:** ADR 0016's `## Open` now keeps only the anti-diagonal bullet; the naming
  > bullet moved to a new `## Resolved` section, settled as `_<algorithm>_cpu_parallel.py`
  > — `_viterbi_cpu_parallel.py` for the decode. Format copied from ADR 0003, which already
  > carries both sections; the record stays `Accepted`, so the index row is unchanged.
  > **Note: the claim was false, but not in the way this goal recorded it.** There is no
  > `_cuda.py` and never was — but a **zero-byte `_numba.py`** did exist, in the *upstream*
  > tree rather than the import. A stale mypy cache entry records `size: 0` for
  > `tokalign.algorithms.needleman_wunsch._numba` against a path under
  > `Developer/Projects/tokalign/`, last modified 2026-04-18, and tokalign's `_backends.py`
  > maps `"numba": "_numba"` — a live pointer to a module nobody wrote. A5 found the
  > registry string; the cache entry is the other half. Together they say the phase was
  > *scaffolded and abandoned*, which is a weaker thing to mirror than a convention in use.
  > **Note: the citation could not have transferred, and that is what produced the wrong
  > proposal.** tokalign identifies an algorithm by *directory*, so its phase files carry no
  > algorithm in their names. This family's packages are flat, and phase 2 is already
  > `_viterbi_cython.pyx` — named so deliberately because `_viterbi.py` is the phase-1
  > reference and the two must coexist. A bare `_cpu_parallel.py` names no algorithm and
  > breaks at the second kernel. So the error was not a missing file in a sound analogy; the
  > analogy was unsound, and the missing file was the visible symptom.
  > **Note:** the anti-diagonal bullet was **deliberately left open**, with F2's
  > corroboration recorded beneath it. The master plan schedules that decision for phase 3,
  > "against a kernel that exists"; closing it on corroborating evidence would preempt a
  > deferral taken on purpose. The note exists so the next reader sees it was weighed, not
  > missed. ADR 0002 names no module files, so it carries no matching defect.
- [x] Land the recovered Viterbi `FORMALIZATION.md` from D4 where the manifest says.
      > **Note:** F2's read-only real-tree run drafted this document in full and was
      > correctly **denied the write** — nothing reached disk, and phase 0 is not done. It
      > verified every line citation the kernel's docstrings claim
      > (`hmm-trainer.lsh:216-218`, `:219`, `:197-198`, `:712-727`, `util.lsh:431-444`) and
      > measured rather than quoted: oracle agreement **3806/3807** across three models,
      > divergence at position 0 of `m008_0001_008` and nowhere else, seed margin
      > **0.004218 bits**, and **0 exact ties in 3804** recurrence positions. Regenerate
      > rather than reuse, but these numbers are checkable against it.
  > **Done:** written to `docs/design/algorithms/viterbi/FORMALIZATION.md` (329 lines),
  > the path the manifest's `[phases] formalization` names — the directory did not exist
  > and this created it. Every measurement regenerated, not pasted; all four ADR links and
  > the three oracle paths resolve; `Derived from` carries
  > `_viterbi.py sha256:48666ca8…`, re-verified against the file after writing.
  > **Note: the agreement figure reproduced exactly — 3806/3807**, diverging only at
  > position 0 of `m008_0001_008`. But the archive's *labels* were wrong twice, and both
  > are now pinned. **"Seed margin 0.004218 bits" is the arc margin**, not the seed
  > difference: the two best arcs are 1.594225 and 1.598444, while the seed difference is
  > **0.789531 bits**. `core.md` had it right ("the two best outgoing arcs differ by 0.004
  > bits"); the plan's paraphrase did not. And **"0 ties in 3804 positions" measures a
  > coarser denominator** than the cell-level count — 0 in **6081 live recurrence cells**.
  > Both say zero; the denominators are not interchangeable.
  > **Note: the divergence is now demonstrated causally rather than inferred.**
  > Substituting the defective raw-probability seed into this implementation reproduces the
  > original's choice exactly — start state 0 — where the corrected seed gives 5. The
  > archive argued the seed *could* decide it because the arc margin is small; this shows
  > it *does*, on demand. That is a stronger claim and it is what the oracle section states.
  > **Note: TC-19's premise was wrong as inherited, and drafting caught it.** The archive
  > says "code 7 decodes, code 8 raises". Verified on a constructed model with `A = 8`:
  > **both raise** — code 7 passes the range guard and dies downstream on the arithmetic,
  > code 8 is refused *by* the guard. Whether `A-1` decodes depends on the model emitting
  > it, so a case written decodes-vs-raises is model-dependent and passes for the wrong
  > reason. The specification now asserts *which error and from where*, which is the
  > observable a `>`/`>=` slip actually changes. Recovering a spec from code found a defect
  > in the spec's own source note — the reverse entrance earning its keep.
- [x] **Two test-coverage gaps in `pfsmgraph-hmm`, found by F2's live run.** Both sit at the
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
  > **Done:** both landed in `packages/pfsmgraph-hmm/tests/test_viterbi.py`; the suite went
  > **280 → 282** and is green. The formalization's TC-19 and TC-20 lose their
  > *NOT YET IMPLEMENTED* markers, so no case there is now unimplemented.
  > **Note: both were mutation-tested, not merely run.** A passing test says nothing about
  > whether it would fail on the defect it names — this repository already learned that
  > when reversing the tie-break broke nothing. Weakening the range guard to `>` fails
  > TC-19 with `IndexError` at `_viterbi.py:98`, which is numpy's, not the `ValueError` the
  > contract promises. Starting `_dead_symbol`'s `enumerate` at 1 fails TC-20. Both are
  > genuine discriminators.
  > **Note: TC-19's premise needed correcting twice, and the second correction was mine.**
  > The source note said "code 7 decodes, code 8 raises"; a constructed model showed both
  > raising, so the previous goal recorded "both raise"; the suite's own `build` helper then
  > showed code 7 *does* decode there, because `code(1) = 7` is emittable under `np.full`.
  > **Both claims were right about their own model.** The stable statement is narrower than
  > either: whether `A-1` decodes is a property of the model, so the case is only meaningful
  > once the model is pinned — and pinning one where `A-1` decodes gives the sharpest
  > contrast, decode-versus-refuse rather than refuse-versus-refuse. The formalization now
  > says that, and records the two-step correction rather than the conclusion alone.
  > **Note:** `cp` is interactive in this shell (`cp -i`), so restoring a mutated file with
  > `cp backup target` hangs on an overwrite prompt with no visible cause. `cat backup >
  > target` is the non-interactive form. The kernel's SHA-256 was re-verified against the
  > formalization's `Derived from` line afterwards, since a mutation left behind would have
  > silently invalidated it.
- [x] Master plan lines 192–196 (phases 2–4 of Viterbi): note they run under
      `dp-compile`, so the first real use of the revised plugin is on the kernel it was
      revised for.
      > **Note: phase 3 has no anti-diagonal, and F2's run says so with a reason.** The
      > Viterbi recurrence is 1-D over time with dense S×S coupling, so there are **no
      > anti-diagonals at all** — the decomposition ADR 0016 names is not merely unchosen
      > here, it does not exist for this kernel. ADR 0016's own Open section anticipated
      > this. Carry "undetermined" forward as the honest answer rather than letting phase 3
      > discover that an invented decomposition is a race.
  > **Done:** the three phase items are annotated in `docs/plan/TODO.md`. One note under
  > phase 2 covers the trio: they run under `dp-compile`, their phase-0 specification
  > already exists, and each phase is written against **it** rather than against the
  > phase-1 source — naming the two things there that are contract (the tie-break and the
  > min-plus objective) and the settled module paths. The anti-diagonal subgoal gains the
  > corroboration **without** the decision: it still owns the call, because the deferral to
  > "a kernel that exists" was taken deliberately.
  > **Note: the goal's own line numbers were stale, and locating by content is what caught
  > it.** It says "lines 192–196"; the items are at 199, 200 and 203, displaced by a
  > subgoal inserted into the master plan earlier the same day. A line number in a plan is
  > a pointer into a file that other goals edit — the master plan especially, since it
  > grows without bound. Cite by content.
  > **Note: the sweep found a semiring inversion, in two places, and it is the one error
  > this project guards against hardest.** The anti-diagonal subgoal said "an associative
  > scan over time in the **(max, +)** semiring", and revision 02's preamble called Viterbi
  > "a single **max-plus** pass with a backtrace". Both are now `(min, +)` / min-plus. The
  > accumulated quantity is a description length in bits, which *grows* as probability
  > falls, so the decode is a min-sum — `core.md`, `_viterbi.py`'s docstring and the
  > formalization all say so, and the general Viterbi literature's max-plus is where the
  > phrasing came from. It mattered most where it sat: an associative scan is the one
  > candidate that would be **built** from that sentence, by someone reading it at the
  > moment they implement phase 3. Recorded in place rather than fixed silently, since a
  > silent fix is indistinguishable from the error never having been there.
  > **Note: a third instance was found and deliberately not touched.**
  > `planned/03-hmm-v0.2.0.md` asks whether "the max-plus/logsumexp associative scan" is
  > worth implementing — but that is the **forward** recurrence, and whether it should read
  > min-plus depends on whether revision 03 keeps the bit domain, which is that revision's
  > undecided call. Flagged in the master plan for it to resolve. Confidence about a
  > specified, measured decode does not transfer to an unimplemented forward pass.
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
