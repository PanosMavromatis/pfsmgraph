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
- [x] **Prepare both plugins for a marketplace release.** Reshaped 2026-09-09: this goal
      read *"Marketplace: nothing to build yet — verified 2026-09-07, no `marketplace.json`
      exists in any of the user's repositories… note in both READMEs that the
      `--plugin-dir` line is interim."* That premise no longer holds — the plugins are
      going into an organisation marketplace, which inverts the goal from *record the
      absence* to *prepare the release*.
      > **Note: this is the one place A1's conclusion inverts, and it is why the reshape
      > matters rather than being bookkeeping.** A1 settled that no version bump was owed,
      > on an explicit premise: no marketplace install exists, so nothing consumes the
      > version as a cache key. The reasoning was sound and is recorded as such. Publishing
      > removes the premise — the version becomes how a consumer's installation decides
      > whether it already has the current plugin. Measured 2026-09-09: `dp-compile` was
      > **20 commits and 42 files** past `0.2.0`, and `workflow-claude` **103 commits and
      > 39 files** past `0.1.0`, whose number predates its entire current command set.
      > **Verified ready, so these are not blockers:** no absolute paths in either plugin's
      > sources — `${CLAUDE_PLUGIN_ROOT}` throughout, which is the portability check that
      > actually matters; both repos clean, on `main`, synced; `dp-compile` no-ops safely
      > in a repo with no manifest (F2's negative check); and the two `PreToolUse` hooks
      > match disjoint tool sets, so they stack rather than compete.
      >
      > **Done (2026-09-09).** Both plugins are public, MIT-licensed, imported with full
      > history into `PanosMavromatis/claude-plugins` as the `mavromatis-ai-labs`
      > marketplace, and consumed here through a tracked `.claude/settings.json` —
      > `8c8d1ec` on this branch, `b6a5b17` and `d640d65` there. The two source
      > repositories are **archived, not deleted**, their descriptions naming where the
      > code went, which keeps `.scratch/README.md`'s provenance rows resolving. They had
      > 0 forks and 0 stars, so the window in which retirement is free was still open.
      >
      > **The runbook was wrong in three places, each corrected in it.** `plugin.json`
      > *does* carry a `dependencies` field; the version check it prescribed read
      > `installed_plugins.json`, which is empty for plugins enabled declaratively rather
      > than installed; and `git log --follow plugins/<name>/<file>` does **not** reach
      > imported history — it returns zero commits, where `git blame` attributes correctly
      > to the original repository's commits. The last is the one worth carrying forward:
      > the history is preserved, and the tool named for reading it was the wrong one.
      >
      > **Verification (§14).** Catalog and both plugins validate with only accounted-for
      > warnings; no shadowing, the skills-dir copy having reported itself inactive before
      > the symlink was removed; every hook path quoted `"${CLAUDE_PLUGIN_ROOT}"` and all
      > eight scripts `755`; about 2.2k tokens always-on with both enabled. Both plugins
      > resolve to the same version `88997d2c120c` — the marketplace repo's HEAD, and a
      > commit touching neither plugin, which is Stage 1 versioning in a monorepo behaving
      > as designed. One item is outstanding and needs a session start: installing
      > `dp-compile` alone into a fresh project to exercise the `dependencies` field.
  - [x] Bump both versions to describe the tree being published — `dp-compile` to `0.3.0`
        and `workflow-claude` to `0.2.0`. Minor rather than patch: each shipped new
        commands and changed behaviour, not fixes.
        > **Superseded the same day, and worth recording as a reversal.** The runbook's
        > §10 Stage 1 posture is to **omit** `version` from `plugin.json` entirely, so the
        > commit SHA becomes the cache key and every push is an update with zero ceremony.
        > That solves the same trap this bump solved — a forgotten bump silently freezing
        > every consumer — by removing the field a human could forget, rather than
        > maintaining it. Strictly better for plugins under active development. The Phase 3
        > normalize commit rewrites each manifest into that shape, so `0.3.0` and `0.2.0`
        > exist in history (`233a311`, `4cca98a`) carrying the cache-key reasoning and
        > reach no consumer. Stage 2 — explicit versions plus `<name>--v<version>` tags —
        > arrives only with the first *constrained* dependency, which nothing here needs.
  - [x] Declare `dp-compile`'s dependency on `workflow-claude` where a consumer will see
        it. ~~`plugin.json` has no dependency field~~ — **wrong, corrected 2026-09-09**: it
        does, since v2.1.110 (`"dependencies": ["workflow-claude"]`), and installing a
        plugin installs its dependencies. The assumption had lived only in
        `references/workflow-claude.md`. A consumer installing `dp-compile` alone gets a
        plugin that correctly detects the absence and stops at every delegation point — by
        design, but discovered at first use rather than at install. It belongs in the
        marketplace description and the README's opening.
        > **Done:** `dp-compile` `4db80b0`, on both surfaces an installer actually sees —
        > `plugin.json`'s description, which is what a marketplace listing renders and
        > often the only text read before installing, and a third opening README paragraph
        > beside the one saying the plugin knows no repository's layout. The manifest was
        > re-parsed after editing; an unparseable `plugin.json` is an unloadable plugin.
        > Folded into `0.3.0` rather than bumped again: that version is pushed but
        > published nowhere, so nothing has cached it — the cache-key argument that made
        > the bump necessary is what makes a second one unnecessary.
        > **Correction, same day:** the prose declaration was the right *human-facing*
        > fix and stays. But I asserted the manifest had no dependency field, and the
        > marketplace runbook — docs verified 2026-09-01 — shows it does. The
        > machine-readable form, `"dependencies": ["workflow-claude"]` in `dp-compile`'s
        > `plugin.json`, is what actually enforces co-installation and is added in the
        > Phase 3 normalize commit. Unconstrained, so it forces no tagging (runbook §9–§10).
  - [x] **Pre-import audit of both plugins, six checks.** Run 2026-09-09 against a
        checklist supplied for exactly this purpose: manifest present, component layout,
        `SKILL.md` `name` frontmatter, hook script paths, paths escaping the plugin root,
        and executable bits. **Nothing failed.**
        > **Result.** Manifest present in both. `dp-compile` is 7 skills + 4 commands,
        > `workflow-claude` 14 commands and no skills — `commands/` is legacy but
        > supported, so that is a future migration rather than a blocker. **All 7
        > `SKILL.md` files set `name`, each matching its directory**, which was the
        > highest-value check: without it Claude Code falls back to the install-directory
        > basename, a version string that changes on every update, and the breakage is
        > invisible until the first one. Both `hooks.json` files use the quoted
        > `"${CLAUDE_PLUGIN_ROOT}"/…` form; nothing writes state beside itself, so
        > `${CLAUDE_PLUGIN_DATA}` is not needed; zero `../` escapes; all 9 scripts are
        > `100755` in the index, which git carries.
        > **Note: the absent `hooks` key in both manifests is correct, and "fixing" it
        > would break both plugins completely.** It reads as the classic omission. It is
        > not: `f94242d` added `"hooks": "./hooks/hooks.json"` and `875f8ee` removed it,
        > recording that it pointed at the **auto-loaded** `hooks/hooks.json` and raised
        > *"Duplicate hooks file detected"*, **failing the entire plugin load**. The
        > absence is load-bearing. Found only because the history contradicted the tip —
        > reading the manifest alone shows a missing key and no reason for it.
  - [x] **After the first marketplace install, check whether each plugin's root
        `CLAUDE.md` leaks into a consumer's session.** Both ship one — 221 lines in
        `dp-compile`, 205 in `workflow-claude` — and both are *plugin-development*
        instructions: component placement, `${CLAUDE_PLUGIN_ROOT}` discipline, version-bump
        rules, `allowed-tools` discipline.
        > **Why this is a post-install check rather than a pre-flight fix.** pfsmgraph's
        > `docs/agents/claude.md` records the governing behaviour — *"Claude Code loads a
        > `CLAUDE.md` found beside files it reads"* — and whether that fires for a plugin
        > copied into `~/.claude/plugins/cache` is not answerable from this side. Test it,
        > do not assume it either way.
        > **The calculus inverts on publication, which is why it becomes a question now.**
        > Inside this workspace the file is exactly the guidance wanted, and the ground
        > rules accepted the cost deliberately: a session under `tmp/` *is* editing the
        > plugin. A marketplace consumer never intends to, so the same file becomes 200+
        > lines of irrelevant instruction pulled into a session that only wanted a command.
        > If it does fire, the fix is the rename pfsmgraph already prescribes for imported
        > trees — `CLAUDE.md` to `CLAUDE.md.orig` — since presence on disk is the trigger
        > and gitignoring does not help.
        > **Resolved by citation, no test needed (2026-09-09).** The marketplace runbook
        > §0 states it outright: *"a `CLAUDE.md` at a plugin root is not loaded as project
        > context; plugins contribute context only through skills, agents, and hooks."*
        > So it does **not** leak to a consumer — and it **does** load for an agent opened
        > at the monorepo root, which is that layout's stated second goal: agents
        > *developing* the plugins see the guidance. Both files stay, unrenamed. The
        > question was well-posed; the answer was already written down elsewhere.
  - [x] **Make both plugin repositories public.** Found 2026-09-09 by checking the README
        link just added: `PanosMavromatis/dp-compile` and `PanosMavromatis/workflow-claude`
        are **both `PRIVATE`**. A public marketplace entry points at a repository URL, so
        neither is installable by anyone until this changes, and the new README link 404s
        for exactly the audience it was written for.
        > **Done 2026-09-09.** Both now `PUBLIC`, and the README link added in `4db80b0`
        > — which 404'd for exactly the audience it was written for — resolves `200`.
        > **Note: this exposes nothing new in kind, which is the useful measurement.**
        > pfsmgraph is already public and already carries all three committer emails —
        > including both of the ones in the plugin histories — and **164**
        > `claude.ai/code/session` URLs in its commit messages. The plugins add 99 more of
        > the same pattern and no new address, and neither has a local username in any
        > tracked file. So this is not a fresh exposure decision; it is the one already
        > taken for pfsmgraph. Worth stating because "make a private repo public" sounds
        > like it needs a security review, and the review that would matter has in effect
        > already happened.
  - [x] Add a `LICENSE` to each repository. Neither has one, so a recipient gets no grant
        of rights. Arguably moot inside one organisation; not moot beyond it.
        > **Q:** MIT, to match pfsmgraph?
        > **A:** Yes. The marketplace will be **public**, which promoted this from
        > last-of-four to first.
        > **Done:** `dp-compile` `93369cf`, `workflow-claude` `7d6b1b2`. All three
        > `LICENSE` files verified byte-identical (`c345985d…`), the same check pfsmgraph
        > applies to its member copies — and for the same reason: a copy that drifts means
        > the shipped artifact no longer says what the root one says.
        > **Note: the copyright line names one person, and that is a decision.** Both
        > `plugin.json` files credit *"Panos Mavromatis & Claude"*, and that stays — an
        > author field is credit. A copyright notice is the statement of **who grants the
        > rights**, so it takes the legal name, for the reason pfsmgraph already records
        > ("a license is read by lawyers"), and it omits "& Claude" because an AI system is
        > not a legal person and cannot hold copyright. A recipient who cannot identify the
        > grantor cannot rely on the grant.
        > **Note:** nothing is vendored in either plugin — every script imports only the
        > standard library — so there was no third-party license to stay compatible with,
        > which is what made a single-holder MIT straightforward rather than a question.
  - [x] Rewrite the loading sections once the marketplace exists. `workflow-claude`'s
        README currently states the opposite of what is about to be true — *"loaded
        deliberately per-project via the CLI flag, **not** through a marketplace"* — and
        `dp-compile`'s table calls a marketplace install conditional on one being
        published. Marketplace install becomes primary; `--plugin-dir` demotes to the
        development path.
        > **Third site, found by `/agents-docs-update` and added here rather than edited
        > early:** pfsmgraph's own `docs/agents/claude.md:95` says *"Loading is interim.
        > Neither plugin is on a marketplace yet."* That is **still true today**, so no
        > edit is owed — but it is false the moment the marketplace exists, and it sits in
        > a *different repository* from the two READMEs this subgoal named, which is
        > exactly how it would have been missed. Editing it now would make a document
        > wrong in order to keep it from becoming wrong later.
        >
        > **Site 3 landed 2026-09-09; the subgoal stays open because the other two are now
        > in a different repository.** `docs/agents/claude.md`'s *"Loading is interim"*
        > paragraph is replaced — both plugins install from `mavromatis-ai-labs`
        > (`PanosMavromatis/claude-plugins`) through a tracked `.claude/settings.json`, and
        > `--plugin-dir` is demoted to the development path. The symlink and its
        > `.git/info/exclude` rule went with it, a removal with **no diff at all**: the
        > path was never tracked, which is precisely the defect the settings file fixes —
        > what a fresh clone got was nothing, silently. The two READMEs this subgoal named
        > are now `plugins/<name>/README.md` inside the marketplace monorepo, so they are
        > edits handed to the operator rather than made here; the same holds for the
        > absence message below. **A subgoal whose sites span two repositories cannot be
        > closed from either one** — that is the shape to expect for the rest of this goal.
        >
        > **Closed 2026-09-09 with the other two sites, in `claude-plugins` `b6a5b17`.**
        > `workflow-claude`'s README led with *"loaded deliberately per-project via the CLI
        > flag, not through a marketplace"* and now leads with the `mavromatis-ai-labs`
        > declaration; `dp-compile`'s table offered a marketplace install *"once one is
        > published"* and now names `workflow-claude@mavromatis-ai-labs`. Two further
        > claims were **stale rather than merely incomplete** and neither was in the
        > subgoal's scope as written: both READMEs said the `.claude/skills/` route "is how
        > at least one repository loads it today", which stopped being true when this
        > repository retired its symlink an hour earlier. Also completed
        > `_meta/consumer-settings.template.json`, which registered no marketplace and so
        > was half a template the moment one existed.
  - [x] Fix the prescribed absence message's three load paths. F2 measured that **both**
        live runs paraphrased it away and neither named any of the three. Behaviour was
        right and only the message drifted, which is the advisory layer's cost — but
        publishing makes marketplace install the *primary* path, so the message needs
        rewriting on its own account.
        > **Blocked on the marketplace existing** for the last two: both name a path that
        > has no value until there is a marketplace to name.
        >
        > **Closed 2026-09-09** (`claude-plugins` `b6a5b17`). The message now names the
        > marketplace as `workflow-claude@mavromatis-ai-labs` rather than saying "a
        > marketplace", and all three paths survive.
        >
        > **What the fix turned up is worth more than the fix.** `dp-compile`'s manifest
        > now declares `workflow-claude` in `dependencies`, so a marketplace install of one
        > installs the other — which makes this message **unreachable** by the path most
        > consumers will use. It is not dead: what survives is the `--plugin-dir` session,
        > where nothing enforces co-loading. So a runtime check that was the primary
        > safeguard is now a development-path fallback, and both READMEs say so rather than
        > leaving a reader to wonder why a carefully worded message never appears.
- [x] **Re-scope pfsmgraph's `/smart-commit` override note, do not delete it** (from G3).
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
      >
      > **Done (2026-09-09).** The Step 3 paragraph now states this repository's convention
      > instead of correcting the command's template, and the two bullets above it are
      > untouched — neither the `VERSION` invariant nor the release-tagging note was
      > affected by PR #21.
      >
      > **The second edit was made, and measuring it falsified the reason given for it.**
      > The note above says pfsmgraph's history is uncontaminated "precisely because every
      > operator overrode the templates by hand every time." Measured over the last 40
      > non-merge subjects: 40 plain imperative, 0 conventional — so the first half holds.
      > But **3 of those 40 are template-shaped** (`Add the branch doc for …`, `Record the
      > merge of …`, `Remove the branch doc …`), so the templates were plainly *not*
      > overridden every time. The reason there is no contamination **effect** is different
      > and more general: those templates are plain imperative and so is this repository,
      > so the rule's discount changes nothing here — 40/40 either way. **A template
      > pollutes a history only when its form disagrees with the repository's.** That is
      > what went into `claude.md`.
      >
      > It also corrects the contrast drawn above. `workflow-claude`'s outcome was not
      > "decided entirely by whether the consumer noticed" — its convention is
      > conventional-commits, so the templates' plain-imperative form *fought* it, and 19
      > template-written subjects against 21 human ones nearly inverted the reading. Same
      > plugin, opposite outcomes, decided by whether the template's form agreed with the
      > repository's. No consumer chooses that, and none can notice it.
