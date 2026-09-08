# Master plan — plugin revisions (`tokalign-dev` → `dp-compile`, with `workflow-claude`)

**Status**: active
**Created**: 2026-09-04
**Driven by**: `/hitl-step N tmp/TODO.md` — always pass the path. This file sits outside
`docs/plan/`, so no resolution rung finds it on its own; without the argument `/hitl-step`
resolves the pfsmgraph branch plan or master plan instead and works the wrong file.

## What this file is

Meta-work for pfsmgraph: the plan for revising two Claude Code plugins that live in their
own repositories, parked as clones under the gitignored `tmp/`. **This is the only file
under `tmp/` that pfsmgraph tracks** — the root `.gitignore` reads `tmp/*` with a
`!tmp/TODO.md` negation for exactly that reason (a bare `tmp/` would ignore the directory
itself, and git never descends into an ignored directory, so no negation beneath it could
work). Every plugin edit is committed in the plugin's own repository, from its own root.
The Q&A logged inline here is the record of *why* the plugins changed; the plugin repos'
histories are the record of *what* changed.

Markers are `/hitl-step`'s: `[ ]` not started, `[~]` in progress, `[x]` done, `[!]`
blocked, `[-]` descoped.

## The three roots

| Root | Remote / tip at plan creation | Commit convention | Notes |
|---|---|---|---|
| `tmp/tokalign-dev/` | `PanosMavromatis/tokalign-dev`, `main` at `5a78334`, clean | imperative subject, no prefix, body explains why (from its history; its `CLAUDE.md` is silent) | to be renamed; version `0.1.0`; own `CLAUDE.md` (template-derived, `@README.md`); `dev/smoke-test.sh` wraps `claude plugin validate` |
| `tmp/workflow-claude/` | `PanosMavromatis/workflow-claude`, `main` at `59661bc`, clean | conventional commits (`feat(scope):`, `chore(plan):`, `docs:`) | dogfoods its own commands: `docs/plan/DO.md` master plan, revisions 02–04 closed, lands work via PRs; strict `allowed-tools` and script-proposes/command-writes conventions in its `CLAUDE.md` |
| `.claude/skills/workflow-claude/` (pfsmgraph, untracked) | byte-identical to the clone at plan creation (`diff -rq` clean) | — | the copy this session actually runs; `settings.local.json` allow-lists its scripts by absolute path. Goes stale the moment the clone is edited (see E6) |

Neither plugin is marketplace-installed: `workflow-claude` loads via `--plugin-dir`, and
neither repo carries a `marketplace.json`. Both are destined for one organisation
marketplace later; nothing here builds it, but nothing here may make it harder (F4).

## Ground rules

- **Commit and push from each plugin's root, never from pfsmgraph's.** A `cd` inside one
  Bash call is fine (`cd tmp/workflow-claude && git …`); if the harness makes that awkward,
  ask the user to run it. pfsmgraph's `/smart-commit` rule (agent docs) does not apply
  inside the plugin repos — neither has a `docs/agents/` tree — but each repo's *own*
  commit convention does, and it overrides `/smart-commit`'s conventional-commit template
  where the two differ (`tokalign-dev`).
- **pfsmgraph is not edited by this plan** beyond this file. Everything the revised plugin
  needs *from* pfsmgraph — a manifest, a `docs/agents/claude.md` section, a `DEFERRED.md`
  closure, a recovered `FORMALIZATION.md` — is collected as a hand-back in F4 and lands on
  a pfsmgraph branch afterwards, through pfsmgraph's own process.
- **Known deviation, recorded rather than fixed:** both clones carry a root `CLAUDE.md`,
  and pfsmgraph's `docs/agents/claude.md` says an imported tree's `CLAUDE.md` is renamed
  to `.orig` on arrival. These are not imports — they are live sibling repositories whose
  `CLAUDE.md` is precisely the guidance wanted while editing them (component placement,
  `${CLAUDE_PLUGIN_ROOT}`, version bumps, `allowed-tools` discipline). **They stay, confirmed
  2026-09-07.** The accepted cost is that a pfsmgraph session reading any file under `tmp/`
  pulls that plugin's instructions into context; that is tolerable precisely because those
  instructions are about editing the plugin, which is what a session down there is doing.
  Whether this exception gets written into `docs/agents/claude.md` as a rule — imported tree
  renames, live sibling repository parked for editing keeps — was left open and is a
  candidate hand-back (F4), not a blocker.
- **Don't relitigate ADR 0016.** The lifecycle is five stages — formalization, then four
  implementation phases: Python, Cython, Numba CPU-parallel (`prange`), Numba CUDA. The
  plugin implements that; it does not reopen it. The one thing ADR 0016 leaves open that
  the plugin *must* take a position on is phase-3 file naming (A5).

## Findings at plan time (so they are not rediscovered)

- `tokalign-dev` is four real skills and four stubs. Real: `algorithm-formalize` (348
  lines), `algorithm-prototype` (218), `cython-translation` (255), `benchmarking` (188).
  Stubs, ten lines each, "TODO: implement": `gpu-parallelization`, `scoring-matrix`,
  `alignment-viz`, `package-release` — and **every reference and example file under the
  stubs is zero bytes** (`anti-diagonal-parallelism.md`, `numba-cuda-patterns.md`,
  `example-numba-impl.py`, …). So the old phase 3 was never implemented in the plugin, and
  both ADR 0016 phases 3 and 4 are written from scratch, not amended.
- Coupling to `tokalign`, counted by `grep -c tokalign` per file: `algorithm-formalize`
  21, `README.md` 14, `cython-translation` 8, `CLAUDE.md` 8,
  `pseudocode-conventions-DP.md` 6, `next-phase.md` 6, `validate-equivalence.py` 5,
  `cython-for-dp.md` 4, `algorithm-prototype` 4, `test-patterns.md` 4,
  `example-formalization.md` 3, `phase-check.md` 3, `new-algorithm.md` 3, and 1–2 in
  seven more. The word count understates it: the *structural* couplings are the
  `src/tokalign/algorithms/<name>/` layout, the `_types.py` trio
  (`Alphabet`/`ScoringMatrix`/`AlignmentResult`), `_registry.py`, `conftest.py`'s
  `get_available_backends()`, `setup.py build_ext --inplace`, `uv run --extra dev`,
  `benchmarks/run_benchmark.py`, and a **pre-ADR-0011 reserved block** (indices 0–3
  reserved, `GAP` = 3, user symbols from 4 — pfsmgraph is 0–5 and from 6, and `tokalign`
  itself was renumbered onto that on 2026-09-01).
- Two assumptions are alignment-specific and would be *wrong* for `hmm`, not merely
  narrow: "normalize to max (similarity)" — pfsmgraph's Viterbi is a **min-sum over
  bits**, and a port that reaches for `max` inverts every comparison (`core.md`); and the
  anti-diagonal wavefront as *the* parallel decomposition — an HMM recurrence is 1-D over
  time with dense state coupling and may have no anti-diagonals (ADR 0016 Open; master
  plan line 194).
- Phase-file naming disagrees three ways: the plugin says `_numba.py` for the GPU phase;
  ADR 0016 proposes `_cpu_parallel.py` and says the proof-of-concept used `_cuda.py`
  (it did not — `.scratch/align-poc/tokalign` has `_numba.py`; hand-back in F4); pfsmgraph
  `hmm` names algorithm-first and flat: `_viterbi.py`, `_viterbi_cython.pyx`.
- Staleness is mtime-based and transitive (`phase-detection.md`). mtime is reset by every
  `git checkout`, and a formalization *recovered from* code is necessarily newer than the
  code, which would mark phase 1 stale and have `/next-phase` regenerate the reference
  implementation from a document derived from it — exactly backwards. A3 exists for this.
- pfsmgraph's backend matrix is a registry, not discovery: one `Backend(name, module,
  hardware)` row per implemented phase in the repo-root `_backends.py`, with ADR 0003's
  loud-skip/hard-fail semantics; and the algorithm suites are **not yet parameterised
  over it** (no backend-selection API until `align`), so a phase skill has to know both
  states of that world.
- `DEFERRED.md` carries, under "No trigger yet": *how the Claude Code development plugin
  fits the multi-package family — one family-wide dev plugin, or one per package.* This
  plan answers it (one generic plugin, per-repo manifest). Closing the entry is a hand-back.

## Section A — Decisions to settle before editing

Each goal here is Q&A. Recommendations are stated so the answer can be "yes"; the point of
logging them is that the plugin's README can later cite *why*, not just *what*.

- [x] Settle the plugin's name, version and rename mechanics.
  > **Q:** Name — `dp-compile`, `dp-lifecycle`, or `dp-phases`?
  > **A:** `dp-compile`.
  > **Q:** Version, and when — `0.2.0` at the rename commit, `0.2.0` at the end, or `1.0.0`?
  > **A:** `0.2.0`, at the rename commit.
  > **Q:** Does the GitHub rename land before the content work or after it?
  > **A:** First, before any content work.
  > **Q:** Keep imperative-no-prefix commits, or adopt conventional commits to match
  > `workflow-claude`?
  > **A:** Keep imperative, and write it into the plugin's `CLAUDE.md`.
  > **Done:** All four settled on the stated recommendation, 2026-09-07. Two findings from
  > the check that preceded them. All three candidate names are free on GitHub, so
  > availability decided nothing and the name was chosen on what it says. And **no
  > `marketplace.json` exists in any of the user's repositories** — `claude-plugin-tools`
  > is unrelated (shell scripts for per-project plugin enabling, untouched since
  > 2026-05-12) — so the organisation marketplace is future work rather than an existing
  > structure these plugins must fit into.
  - [x] **Name: `dp-compile`.** The emphasis stays on dynamic-programming kernels, and
        "compile" reads as the progressive lowering from prose to pseudocode to Python to
        compiled and JIT phases. Its known weakness is recorded rather than argued away:
        "compile" understates phases 3 and 4, which JIT at call time rather than compiling
        ahead of it. `dp-lifecycle` named the ordered-lifecycle framing more exactly and
        `dp-phases` matched the plugin's own `/phase-check` vocabulary; both were free.
  - [x] **Version: `0.2.0`, set at the rename commit** rather than at the end. The version
        is effectively a cache key — Claude Code picks up a plugin's changes only when it
        changes — so bumping early keeps the cached `tokalign-dev@0.1.0` and the new plugin
        from ever looking like the same thing. Deliberately still pre-1.0: the manifest
        format (A2) is new, unexercised against any repository but this one, and will
        almost certainly change on contact with the second consumer.
  - [x] **The rename lands first, before any content work.** Decided here; *executed* in
        B1, which now carries the command sequence. Every commit of the revision then sits
        under the new name, and `gh repo rename` leaves a redirect so the old URL keeps
        resolving. Doing the four parts as one step — repo, remote, directory,
        `plugin.json` — also avoids a window where `plugin.json`'s `name` and the
        directory name disagree, which the plugin's own `CLAUDE.md` forbids.
  - [x] **Commit convention: keep imperative-no-prefix, and write it into the plugin's
        `CLAUDE.md`.** Its history already is that ("Implement benchmarking skill and
        benchmark command"), and the history is the authority — the same principle
        pfsmgraph applies to override the identical `/smart-commit` template. Writing it
        down is the load-bearing half: today the repo states no convention anywhere, so a
        fresh session has nothing to read and follows the command's conventional-commit
        template instead. Accepted cost: the two plugins will sit in one marketplace under
        two different conventions.
- [x] Settle how a consumer repository describes itself to the plugin (the manifest).
  > **Q:** Where does the manifest live — a root `dp-compile.toml`, a `[tool.dp-compile]`
  > table in `pyproject.toml`, or `.claude/`?
  > **A:** Root `dp-compile.toml`.
  > **Q:** How is file layout expressed — one path template per phase, named layout kinds,
  > or inference from a reference algorithm?
  > **A:** One path template per phase.
  > **Q:** Where does pfsmgraph keep a `FORMALIZATION.md`?
  > **A:** `docs/design/algorithms/<algorithm>/`.
  > **Done:** Settled 2026-09-07, all three on the recommendation, and the draft manifest
  > is in subgoal 3 below. **Two findings reshaped what the manifest has to carry.** First,
  > the plugin is stale against *its own* repository, not merely narrow for this one:
  > `tokalign`'s `_types.py` was renumbered onto ADR 0011 on 2026-09-01 (`GAP = 4`,
  > `USER_BASE = 6`, identical to pfsmgraph), while the plugin's skills still say "indices
  > 0-3 are reserved and user symbols start at 4". So that text is *wrong for both
  > consumers*, and the manifest needs no reserved-block field at all — only a pointer to
  > where the consumer documents its encoder. Second, **algorithms must be listed rather
  > than discovered**, and the layout settles that rather than taste: `tokalign` discovers
  > by listing `algorithms/*/`, but pfsmgraph's packages are flat, so globbing `_*.py`
  > under `hmm` returns `_numeric.py` and `_params.py` beside `_viterbi.py`.
      The plugin cannot know where a repo keeps its kernels, what its phase files are
      called, how it builds, how it tests, or how a backend is registered — and every one
      of those differs between `tokalign` and pfsmgraph.
  - [x] **Mechanism: a repo-root `dp-compile.toml`.** Self-contained, obvious to a reader
        opening the repository, and independent of the repo's packaging shape. When the
        file is absent the plugin falls back to defaults reproducing `tokalign`'s layout,
        so the proof-of-concept keeps working with no file at all.
        The rejected alternatives, and why — the reasoning here is not what this line
        first assumed. A `[tool.dp-compile]` table in the root `pyproject.toml` is the
        idiomatic Python home for tool config, and **"the root is virtual" is not an
        objection**: virtual means no `[project]` table, while `[tool.*]` tables work
        fine, as `[tool.uv]` and `[tool.pytest.ini_options]` already do there. It loses on
        the workspace ambiguity instead — whether a member's own table overrides the
        root's has to be answered, and answered the same way by every command.
        `.claude/dp-compile.toml` loses on something sharper: pfsmgraph's entire `.claude/`
        directory is untracked today (`git ls-files .claude` returns nothing), so a shared
        contract placed there would be invisible in a clone.
  - [x] **Contents.** Per repository: the invariant documents every phase skill reads
        before writing, an encoder *pointer* (no reserved-block values — see the parent's
        first finding), build and test commands, and the backend-registration recipe. Per
        phase: one path template, with `{algorithm}` and `{package}` substituted — which
        is the whole of the layout question, since any layout expressible as a path is
        expressible as a template, and **phase-file naming (A5) falls out for free**
        because the template carries the name. Per algorithm: only what the templates
        cannot supply — which package it belongs to, and any differential oracle (D3).
        Named layout kinds (`per-algorithm-dir`, `flat-suffix`) were rejected for costing
        generality: a third consumer matching neither would need a plugin change rather
        than a config change. Inferring the layout from a reference algorithm was rejected
        for being undefined exactly when the plugin is most useful — a repository adding
        its *first* algorithm — and for being invisible, so a wrong inference is found late.
  - [x] **Drafted** below. It lives here rather than in the plugin because A1 settled that
        the rename runs before any content work, so `tmp/tokalign-dev/` takes no writes
        yet; B2 copies it into the plugin's `examples/`, and F4 lands it at pfsmgraph's
        root. Keeping the draft in this tracked file rather than a scratch path is
        deliberate — it is the keystone artifact and everything in Section B reads from it.

        ```toml
        # dp-compile manifest -- pfsmgraph
        #
        # States what the plugin cannot infer. With no manifest at all, dp-compile falls
        # back to defaults reproducing tokalign's layout, so that repository needs no file.

        [project]
        name = "pfsmgraph"

        # Read before writing any phase. These carry invariants that no amount of reading
        # the kernel reveals: arc-emission (output_p[i, j, sym], never B[s, sym]), the N+1
        # state geometry, min-sum over bits rather than max-product, and ADR 0003's rule
        # that a tie-breaking rule is contract because two correct backends would
        # otherwise legitimately disagree.
        invariants = [
          "docs/agents/core.md",
          "docs/design/adr/0002-three-phase-algorithm-lifecycle.md",
          "docs/design/adr/0003-one-parameterized-test-suite-per-algorithm.md",
          "docs/design/adr/0015-arc-emission-mealy-formulation.md",
          "docs/design/adr/0016-numba-cpu-parallel-phase.md",
          "docs/design/adr/0017-frozen-parameter-object-for-hmm.md",
        ]

        # Where the symbol/code mapping is defined. dp-compile hardcodes no reserved block
        # and no user-symbol base; it reads these instead.
        encoder = [
          "docs/api/dataseq/encoder.md",
          "docs/design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md",
        ]

        [commands]
        build    = "uv sync"        # meson-python editable; compiled phases rebuild on import
        test     = "uv run pytest"
        test_one = "uv run pytest {path}"

        # One template per phase. tokalign's default is
        # "src/tokalign/algorithms/{algorithm}/_python.py" and needs no new concept.
        [phases]
        formalization = "docs/design/algorithms/{algorithm}/FORMALIZATION.md"
        python        = "packages/pfsmgraph-{package}/src/pfsmgraph/{package}/_{algorithm}.py"
        cython        = "packages/pfsmgraph-{package}/src/pfsmgraph/{package}/_{algorithm}_cython.pyx"
        cpu_parallel  = "packages/pfsmgraph-{package}/src/pfsmgraph/{package}/_{algorithm}_cpu_parallel.py"
        cuda          = "packages/pfsmgraph-{package}/src/pfsmgraph/{package}/_{algorithm}_cuda.py"

        # ADR 0003. A new backend is a row in the registry plus, for a compiled phase, an
        # entry in that member's meson.build -- install_sources for .py, extension_module
        # for .pyx. meson does not glob, so the second half is not optional.
        [backends]
        registry       = "_backends.py"
        build_manifest = "packages/pfsmgraph-{package}/meson.build"

        # Algorithms are listed, never discovered: the packages are flat, so globbing
        # _*.py under hmm returns _numeric.py and _params.py beside _viterbi.py.
        [algorithms.viterbi]
        package = "hmm"
        # Differential oracles (D3). Not a pure equality check: the delta-seeding defect is
        # fixed rather than reproduced, so m008_0001_008 differs at exactly one position.
        oracles = [
          ".scratch/hmm-lush/Training/set02a/set02a_200/m001_0001_001.vpath.xls",
          ".scratch/hmm-lush/Training/set02a/set02a_200/m001_0005_005.vpath.xls",
          ".scratch/hmm-lush/Training/set02a/set02a_200/m008_0001_008.vpath.xls",
        ]
        ```
- [x] Settle the staleness mechanism (replaces mtime).
  > **Q:** What replaces mtime — a provenance header with a content hash, mtime plus a
  > touch discipline, or git commit order?
  > **A:** Provenance header with a content hash.
  > **Q:** After a formalization is recovered from code and approved, which way does the
  > dependency edge point?
  > **A:** It stays reverse — the doc derives from the code.
  > **Q:** How is the header written in a Markdown formalization?
  > **A:** A row in the Metadata table the template already mandates.
  > **Done:** Settled 2026-09-08. Git commit order was rejected on something the earlier
  > framing missed: it is blind to *uncommitted* work, so a kernel edited and not yet
  > committed reads as fresh — and that is exactly the window the plugin exists to guide.
  > It shares mtime's direction blindness on top of that. **The consequence that reaches
  > furthest is not the hash but the recorded edge**: once each file names its own source,
  > the dependency structure is *read from the files* rather than assumed, and it stops
  > being a chain. For a recovered algorithm `_viterbi.py` is a root with two dependents —
  > the formalization and the `.pyx` — so editing the kernel marks both stale, which a
  > linear chain cannot express. C1 carries what that does to effective-phase detection.
  - [x] **Mechanism: a provenance header recording the source path and its sha256.** Stale
        iff the recorded hash differs from the named source's current hash. Survives every
        `git checkout`, is visible in a diff, and encodes direction explicitly.
        **The direction is the point, not the hash.** A recovered formalization records
        that it derives from the kernel, so the kernel is not stale relative to it — the
        case mtime cannot represent at all, since the recovered document is necessarily
        newer than its own source.
  - [x] **Format.** In `.py` and `.pyx`, a comment above the module docstring:
        `# dp-compile: derived-from <path> sha256:<hash>`. In a Markdown formalization, a
        row in the `| Field | Value |` Metadata table the template already mandates — which
        invents no syntax and puts the claim where a human reviewing the document actually
        reads it. YAML front-matter was rejected because **no tracked `.md` in pfsmgraph
        has any** (checked 2026-09-08), so it would introduce a second document
        convention; an HTML comment was rejected for the opposite reason to its appeal —
        invisible when rendered means a wrong provenance line survives review unseen.

        **The hash is computed over the file with its own `dp-compile` header removed**,
        and that is load-bearing rather than tidy. A file's header records its *source's*
        hash, so if the header counted toward its own file's hash, re-stamping any file
        would spuriously invalidate everything downstream of it — and stamping an existing
        chain would take one pass per link instead of one pass. A dependent depends on its
        source's *content*, never on its source's provenance bookkeeping.

        Accepted cost, stated so it is not a surprise: the hash is over the whole file, so
        a docstring typo fix in `_viterbi.py` marks its formalization and its `.pyx` stale.
        That is exactly what mtime does today, so it is no regression — but unlike mtime,
        reverting the typo restores the hash and clears the staleness.
  - [x] **Legacy fallback: a file with no header falls back to mtime, with a warning that
        names the file.** Every existing `tokalign` artifact is headerless, so without this
        the mechanism would break the one repository the plugin currently works in. The
        warning is what keeps the fallback from becoming permanent by inattention — a
        silent fallback would leave a repo on mtime forever with nothing saying so.
- [ ] Settle the fate of the `tokalign`-only components.
  - [ ] The three domain stubs — `scoring-matrix`, `alignment-viz`, `package-release`.
        Recommend delete: they are ten lines each over zero-byte references; scoring
        matrices and alignment visualisation belong to an alignment-domain plugin or the
        consumer, and `package-release` duplicates the consumer's own release path
        (pfsmgraph has `just release` and a runbook).
  - [ ] `validate-equivalence.py` (hardwired to `tokalign._types` and the `align()`
        signature). Recommend script out, guidance in: the Cython/parallel skills end by
        writing a fuzz/property test *into the consumer's suite*, using the consumer's own
        types and encoder, where ADR 0003's parameterisation can pick it up.
  - [ ] `run-benchmark.sh` (walks up to `pyproject.toml`, calls
        `benchmarks/run_benchmark.py`). Recommend keep the skill, take the command from
        the manifest, drop the wrapper.
  - [ ] The pre-commit hook (`setup.py build_ext --inplace`, `--extra dev`, `pytest
        tests/ -x`). Fires on *every* `git commit` in every repo where the plugin is
        loaded, including inside `/smart-commit` (hooks stack). Recommend: keep only if
        it becomes manifest-driven and no-ops with a message when no manifest exists;
        otherwise drop it and make "tests green before commit" a step the phase skills
        state. Decide here; E3 implements.
- [ ] Settle phase-file naming for phases 3 and 4, where ADR 0016 left it open.
      Recommend the plugin's canonical phase names be `python`, `cython`, `cpu_parallel`,
      `cuda`, and the manifest's layout pattern decide whether they are filenames
      (`_cython.pyx`) or suffixes (`_viterbi_cython.pyx`). Backend-matrix names follow
      (`python · cython · cpu-parallel · cuda`). `_numba.py` is retired: it names the
      library, and two phases now share the library.
- [ ] Settle how `dp-compile` detects `workflow-claude`, and what absence means.
      Installed-plugins JSON cannot be trusted (`--plugin-dir` loads never appear there).
      Recommend the precedent `/smart-merge` already uses for MCP: *you can see your own
      tools, so judge from that* — each `dp-compile` command opens by checking its tool
      list for `workflow-claude:` skills and, if none, stops with the install line
      (`claude --plugin-dir <path>/workflow-claude` today; the marketplace line later).
      Never degrade silently.
- [ ] Settle the branch and plan workflow *inside* each plugin repository.
  - [ ] `workflow-claude`: it has a master plan and closed revisions, and lands work via
        PRs. Recommend `/open-revision 05-dp-compile-interop` (label to taste) and one
        branch, since its edits are small and its convention is its own.
  - [ ] `dp-compile`: no `docs/plan/`, history is straight to `main`. Recommend a
        sequence of focused commits to `main`, not bootstrapping plan machinery into a
        repository that is being rewritten end to end — the record of *why* is this file.
  - [x] pfsmgraph side: **done 2026-09-07 — branch `chore/revise-plugins` opened from
        `beb0637`, with a pointer plan.** The name took the `chore/` prefix every prior
        branch in this repository carries, so the plan directory is
        `docs/plan/chore-revise-plugins/` rather than the flat `revise-plugins` this line
        first assumed; `chore` because pfsmgraph receives almost nothing from this work —
        the substantive diff lands in two other repositories. `/new-branch` created
        `docs/plan/chore-revise-plugins/TODO.md` as usual,
        and its body is then replaced with a one-line pointer to `tmp/TODO.md`. Rung 2 still
        resolves, but to something that says immediately where the real plan is, which beats
        both alternatives: declining the plan makes rung 2 miss and fall through to the
        pfsmgraph *master* plan (a different wrong file, merely a more obvious one), and a
        duplicated real plan would drift from this file the moment work starts while
        describing commits made in other repositories. Keep the `**Status**:` stamp so
        `/smart-merge` has something to mark `merged`, and keep the master-plan backlink so
        `/file-plans` can place it later.

## Section B — Rename and generalise `tokalign-dev` → `dp-compile`

- [ ] Rename everything that carries the old identity.
  - [ ] `plugin.json` name, description, version (per A1); local directory; `origin`.
        **A1 settled that this runs first, before any content work.** `gh repo rename` is
        the user's to run — it is outward-facing and touches the GitHub account, not the
        working copy:

        ```bash
        cd tmp/tokalign-dev
        gh repo rename dp-compile   # leaves a redirect, so the old URL keeps resolving
        git remote -v               # gh normally rewrites origin itself — verify, don't assume
        cd .. && mv tokalign-dev dp-compile
        ```

        Then `plugin.json` in the same commit: `name` to `dp-compile` (it **must** match
        the directory name), `version` to `0.2.0`, and a `description` that no longer says
        "the tokalign package". Everything after this point happens under
        `tmp/dp-compile/`, so update the pfsmgraph-side references in this plan too — the
        three-roots table at the top still says `tmp/tokalign-dev/`.
  - [ ] Every `tokalign-dev:` namespace reference — the `Skill` tool calls in
        `new-algorithm.md`, `next-phase.md`, `cython-translation/SKILL.md`; the smoke
        test; `README.md`; `CLAUDE.md`.
  - [ ] Acceptance: `grep -rn 'tokalign-dev' --exclude-dir=.git .` returns nothing.
- [ ] Introduce the manifest (A2) and a shared `commands/references/manifest.md` that
      every command and skill reads first, replacing `phase-detection.md`'s hardcoded
      `src/tokalign/algorithms/<name>/`. Defaults reproduce `tokalign`. Include the
      pfsmgraph example.
- [ ] Strip `tokalign` from the four real skills and the four commands — the structural
      couplings, not just the word.
  - [ ] Paths and layout → manifest (formalize, prototype, cython, benchmarking, all four
        commands, `phase-detection.md`).
  - [ ] `_types.py` trio and `_registry.py` → manifest-named encoder, result type and
        registration recipe. The wrapper/kernel split stays as an invariant — pfsmgraph
        sharpens it: the kernel neither validates nor raises, the wrapper turns
        `inf` into `ImpossibleSequenceError` — and the skill should state the generic
        form (kernel purely numeric so phases 2–4 are transliterations).
  - [ ] Objective and semiring stated explicitly in the formalization (max-product,
        min-sum, max-plus, …) and **never normalised** by the skill; delete "normalize to
        max (similarity)". Cite the `hmm` bits trap as the reason.
  - [ ] Reserved block never hardcoded: delete "indices 0–3", "GAP = 3", "user symbols
        from 4"; the manifest points at the consumer's encoder documentation (pfsmgraph:
        ADR 0011 and `docs/api/dataseq/`).
  - [ ] Build and test commands → manifest (pfsmgraph: meson-python rebuild-on-import,
        `uv run pytest`, `meson.build` listing; no `setup.py`, no extras).
  - [ ] `test-patterns.md` → both worlds: import auto-discovery (`tokalign`) and a
        registry row with `hardware` and loud-skip semantics (pfsmgraph, ADR 0003), plus
        the not-yet-parameterised state where a backend-reaching test lives in a labelled
        non-shared section (`test_viterbi.py` precedent).
  - [ ] Affine gaps, alignment types, `gap_open`/`gap_extend` → an alignment-family
        example inside a generic skill, not the skill's assumption.
- [ ] Generalise the formalization conventions and template.
  - [ ] `pseudocode-conventions-DP.md` header and voice; keep its DP-only scope and the
        "do not force a non-DP algorithm into this template" rule.
  - [ ] Template metadata: replace the alignment-specific rows (Alignment type, Gap model)
        with generic ones plus an algorithm-family block; add **Objective** (semiring and
        direction) and **Parallel decomposition** (consumed by C2) as mandatory sections;
        keep the TC-XX test-specification discipline and its three precision levels.
  - [ ] Keep the Needleman-Wunsch example as the example, labelled as one family's
        instance.
- [ ] The mechanical tail: `dev/smoke-test.sh` `expected` array, `hooks/` per A4,
      `CLAUDE.md` (drop the `{{template}}` residue, add the commit convention per A1),
      `plugin.json` version; `claude plugin validate` green.

## Section C — Implement the ADR 0016 chain

- [ ] `phase-detection.md` and its three consumers for five stages.
  - [ ] Nominal-phase table 0–4; filenames from the manifest; staleness per A3.
  - [ ] **Effective phase needs redefining, not extending — A3 changed its shape.** The
        current rule is "the phase just before the first stale file *in the chain*", which
        assumes one linear order that is both the phase order and the dependency order.
        With provenance edges read from the files those come apart: for an algorithm whose
        formalization was recovered, the doc is phase 0 by number but *downstream* of
        phase 1 by dependency, and a stale doc is a reason to regenerate the doc — never a
        reason to regenerate the kernel it was derived from. Replace it with: **the target
        is the first stale artifact in dependency (topological) order, regenerated from
        the source its own header names**; and when nothing is stale, the target is the
        next unwritten phase. That rule covers the forward chain unchanged and the
        recovered DAG correctly, where "the phase before the first stale file" cannot.
  - [ ] `/next-phase` routes 2 → 3 to the new CPU-parallel skill and 3 → 4 to GPU;
        `/phase-check` reports four backends; `/benchmark` counts them.
  - [ ] Every "do not create X yet" list in the formalize, prototype and Cython skills
        names the two new files.
- [ ] New skill `cpu-parallelization` (phase 3: `@njit(parallel=True)` with `prange`).
  - [ ] The decomposition comes from `FORMALIZATION.md`'s Parallel decomposition section,
        never invented here. Anti-diagonal for 2-D alignment DP; for a 1-D-over-time
        recurrence with dense state coupling (HMM) the candidates are
        states-within-a-timestep, batch over sequences, or an associative scan in the
        (min, +) semiring — and "no anti-diagonals" is a valid answer the skill must
        accept, because master plan line 194 settles that question against the Viterbi
        kernel, not here.
  - [ ] Race-freedom is checked against the phase-1 oracle (same discipline as every
        phase), and **the tie-breaking rule is contract** (`core.md`): the per-cell
        reduction over predecessors stays sequential so first-wins survives `prange`.
  - [ ] `numba` becomes a hard runtime dependency of the package (ADR 0016), declared via
        the manifest's dependency recipe; `numba-cuda` does not.
  - [ ] Registration: a backend row with `hardware=None` (a failed import escalates).
- [ ] Write `gpu-parallelization` (phase 4) for real — currently a stub over zero-byte
      references.
  - [ ] `numba-cuda` behind the consumer's GPU extra (ADR 0004); backend row with
        `hardware="CUDA device"`, so absence skips loudly; CI escalation is the
        consumer's (`PFSMGRAPH_REQUIRE_BACKENDS`-style) and comes from the manifest.
  - [ ] Reuse phase 3's decomposition verbatim; what is left to debug is
        hardware-specific — coalescing, occupancy, `cuda.jit` semantics — and the skill
        says so, so it does not re-derive the algorithm.
  - [ ] Fill `references/anti-diagonal-parallelism.md`, `numba-cuda-patterns.md` and the
        example.
- [ ] Benchmarking over four backends: JIT warm-up applies to phase 3 as to phase 4;
      crossover reporting names all three transitions; command from the manifest.
- [ ] Equivalence discipline per phase, written into the consumer: extend the
      parameterised suite where one exists, or add to the labelled non-shared section
      where it does not (pfsmgraph until `align`'s backend-selection API), plus the
      property test that replaces `validate-equivalence.py` (A4).

## Section D — Legacy-code entry: recovering a formalization from an implementation

The `hmm` case: the reference implementation was translated from Lush, no natural-language
or pseudocode statement of the algorithm ever existed, and the closest thing to a
specification is `HMMLIB-ACCOUNT.md` plus three `.vpath.xls` oracles the original wrote.
The plugin's chain assumes prose → pseudocode → Python; this section gives it a second
entrance so every algorithm ends up with a pseudocode reference regardless of where it
started.

- [ ] New skill (name to settle: `algorithm-recover`, `formalize-from-code`, or a mode
      of `algorithm-formalize`).
  - [ ] Input: an existing phase-1 implementation, optionally the legacy source it was
        translated from and any differential fixtures. Output: `FORMALIZATION.md` with a
        provenance header (A3) naming the code as its source.
  - [ ] Section mapping: "Source Pseudocode" becomes "Source implementation" (cited by
        path and commit); "Adaptations from Source" becomes "Deviations of the
        implementation from the legacy source" — for `hmm`: the δ-seeding fix, the `-1`
        log-zero sentinel not reproduced, `psi` as `int64`, `safe_divide`'s zero
        convention; Objective states the semiring the code actually uses (min-sum over
        bits); TC-XX cases are *extracted* from the existing tests and fixtures, not
        invented.
  - [ ] Same hard stop for human review as the forward skill; the recovered document
        becomes the living specification from then on, and later edge cases fold into it.
- [ ] `/new-algorithm` gains an entry-mode question: from source material (forward), from
      an existing implementation (reverse), or from existing pseudocode (skip to phase 1
      with the supplied `FORMALIZATION.md`). Phase detection treats a recovered
      formalization as derived, so phase 1 is not stale (A3).
- [ ] Differential oracles as a first-class input. Where a legacy implementation left
      outputs beside its inputs (the `.vpath.xls` files), the formalization records the
      oracle's location and the differential test as a relational TC, and every later
      phase inherits it — the plugin must not let a kernel validate against itself.
- [ ] Dry run on pfsmgraph's Viterbi: recover a `FORMALIZATION.md` from
      `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_viterbi.py` into the scratchpad (not
      into pfsmgraph — landing it is a hand-back), and check it against
      `HMMLIB-ACCOUNT.md` §3 and `core.md`'s arc-emission, N+1, min-sum and tie-break
      facts. This is the acceptance test for the three goals above.

## Section E — Interoperation with `workflow-claude`

- [ ] Presence check (A6) at the head of every `dp-compile` command, and a
      "Prerequisites" section in its README with the install line. Absence stops with
      instructions; it never degrades.
- [ ] Delegation, implemented rather than described. `dp-compile` owns the algorithm
      lifecycle and nothing else: commits go through `/smart-commit` (its end-of-phase
      text says so instead of prompting for a commit itself), branches through
      `/new-branch`, phase progression is recorded as branch-plan subgoals worked by
      `/hitl-step`, merges through `/smart-merge`, and documentation is never touched by
      `dp-compile` (`/agents-docs-update` owns it). Decide what, if anything,
      `/next-phase` writes into the branch plan; keep it minimal.
- [ ] Hook coexistence (A4's decision): if the pre-commit hook survives, it fires inside
      `/smart-commit`'s `git commit` (hooks stack — the same composition the conflict
      report describes for `security-guidance`) and must no-op with a message where no
      manifest exists. Document the interaction in `workflow-claude`'s
      `_meta/plugin-conflict-report.md` (new section) and in `dp-compile`'s README.
- [ ] `workflow-claude` edits, expected to be small: README "Companion plugins" pointer;
      conflict-report section; a `CLAUDE.md` line. Record any functional change that
      turns out to be needed (a documented detection contract, say) rather than slipping
      it in. Landed through its own revision (A7).
- [ ] `dp-compile` README rewrite: the combined workflow (`/open-revision` →
      `/new-branch` → `/new-algorithm` and `/next-phase` inside `/hitl-step` →
      `/smart-commit` → `/smart-merge`), the manifest, the five-stage table under ADR 0016
      numbering, the two entrances (D), prerequisites, and the removal of the `tokalign`
      "Decisions" and "Naming" sections.
- [ ] Replace the live copy with a symlink. **Settled 2026-09-07.**
      `.claude/skills/workflow-claude/` in pfsmgraph is an untracked byte-identical copy of
      the clone and goes stale at the first edit; a re-sync step fails *silently*, since a
      stale copy still loads and still works, just from the old text. Symlink it to
      `tmp/workflow-claude` so drift is unrepresentable and edits take effect with no sync
      at all. The accepted cost is that the clone can no longer move or be deleted without
      breaking the live plugin — note it in the plugin's own README, since a future clone
      elsewhere would hit it.
  - [ ] `settings.local.json`'s allow-list is **not** a cost here, contrary to the concern
        raised when this was posed. Permission patterns match the command *text*, not the
        resolved inode, and the two entries spell
        `.claude/skills/workflow-claude/scripts/…` — a path the symlink keeps valid. Verify
        once by running `check-agents-md.sh` through that path after the swap rather than
        assuming it.
  - [ ] Do the swap at E6 time, not earlier. The copies are byte-identical today, so there
        is nothing to gain from swapping mid-session and a live plugin directory is not
        worth disturbing for no benefit.

## Section F — Verification, commit, hand-back

- [ ] `./dev/smoke-test.sh` and `claude plugin validate` green in `dp-compile`.
      `workflow-claude` has no suite; if `/step` or `/hitl-step` were touched, run its
      byte-identical Step 1 check from its `CLAUDE.md`.
- [ ] Live-session check, run by the user: `claude --plugin-dir tmp/dp-compile
      --plugin-dir tmp/workflow-claude` in pfsmgraph with the example manifest in place;
  - [ ] **Ordering problem, found while drafting the manifest in A2.** This check needs
        `dp-compile.toml` at pfsmgraph's root, but F4 defers landing it to a later
        pfsmgraph branch, and the ground rules say this plan edits nothing in pfsmgraph
        beyond `tmp/TODO.md`. Decide which gives: land the manifest on *this* branch as
        the one exception (it is the artifact the whole revision is built around, and the
        check is worthless without it), or run the check against an untracked copy and
        land the tracked one later. Do not discover this at F2 with the plugin finished
        and nothing to point it at.
      `/dp-compile:phase-check viterbi` reports phase 1; the same session without
      `workflow-claude` trips the presence check. Record results as `> **Ran:**` lines.
- [ ] Commit and push each repository from its own root per A7; version and GitHub
      rename per A1.
- [ ] Hand-back list for pfsmgraph, executed on a pfsmgraph branch afterwards, not here:
  - [ ] Land the manifest at the pfsmgraph root.
  - [ ] Close `DEFERRED.md`'s "how the development plugin fits the multi-package family"
        entry: one generic plugin, per-repo manifest, this plan as the record.
  - [ ] A `dp-compile` section in `docs/agents/claude.md` (it names a Claude Code
        feature, so `claude.md` not `core.md`).
  - [ ] ADR 0016 Open: phase-3 naming settled per A5; and correct its claim that the
        proof-of-concept used `_cuda.py` (it used `_numba.py`).
  - [ ] Land the recovered Viterbi `FORMALIZATION.md` from D4 where the manifest says.
  - [ ] Master plan lines 192–196 (phases 2–4 of Viterbi): note they run under
        `dp-compile`, so the first real use of the revised plugin is on the kernel it was
        revised for.
  - [ ] Marketplace: nothing to build yet — **verified 2026-09-07, no `marketplace.json`
        exists in any of the user's repositories**, and `claude-plugin-tools` is unrelated.
        Note in both READMEs that the `--plugin-dir` line is interim.
