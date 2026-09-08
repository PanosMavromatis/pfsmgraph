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
| `tmp/dp-compile/` | `PanosMavromatis/dp-compile`, `main` at `3b7fc67` | imperative subject, no prefix, body explains why (settled in A1; to be *written into* its `CLAUDE.md` at B4, since the repo states it nowhere) | **renamed from `tokalign-dev` 2026-09-08**, old URL redirects; version `0.2.0`; own `CLAUDE.md` (template-derived, `@README.md`); `dev/smoke-test.sh` wraps `claude plugin validate` |
| `tmp/workflow-claude/` | `PanosMavromatis/workflow-claude`, `main` at `59661bc`, clean | conventional commits (`feat(scope):`, `chore(plan):`, `docs:`) | dogfoods its own commands: `docs/plan/DO.md` master plan, revisions 02–04 closed, lands work via PRs; strict `allowed-tools` and script-proposes/command-writes conventions in its `CLAUDE.md` |
| `.claude/skills/workflow-claude/` (pfsmgraph, untracked) | byte-identical to the clone at plan creation (`diff -rq` clean) | — | the copy this session actually runs; `settings.local.json` allow-lists its scripts by absolute path. Goes stale the moment the clone is edited (see E6) |

Neither plugin is marketplace-installed: `workflow-claude` loads via `--plugin-dir`, and
neither repo carries a `marketplace.json`. Both are destined for one organisation
marketplace later; nothing here builds it, but nothing here may make it harder (F4).

## Ground rules

**A finding is a `>` note, never a `- [ ]` line.** Learned 2026-09-08, after seven
findings and deferral notes in Section B were written as unchecked checkboxes and then
reported for two sections as outstanding work. Nothing was actually pending — every one had
been discharged by A4, C1 or C2 — but a checkbox is a promise that something will be done,
and a running `grep -c '^- \[ \]'` counts only top-level items, so an indented note was
invisible to the count *and* looked open to a reader. Where a note records a real deferral,
mark it `[-]` with the goal that will pick it up; where it records something observed,
it is a `>` blockquote.


- **Commit and push from each plugin's root, never from pfsmgraph's.** A `cd` inside one
  Bash call is fine (`cd tmp/workflow-claude && git …`); if the harness makes that awkward,
  ask the user to run it. pfsmgraph's `/smart-commit` rule (agent docs) does not apply
  inside the plugin repos — neither has a `docs/agents/` tree — but each repo's *own*
  commit convention does, and it overrides `/smart-commit`'s conventional-commit template
  where the two differ (`dp-compile`).
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
- Phase-file naming disagrees three ways: the plugin's docs say `_numba.py` for the GPU
  phase; ADR 0016 proposes `_cpu_parallel.py` "mirroring the existing `_python.py` /
  `_cython.pyx` / `_cuda.py` per-phase naming seen in `.scratch/align-poc/tokalign`";
  pfsmgraph `hmm` names algorithm-first and flat, `_viterbi.py` and `_viterbi_cython.pyx`.
  **The precedent ADR 0016 cites does not exist** *(corrected 2026-09-08; this line
  previously said the proof-of-concept "has `_numba.py`", which is also false and was
  written from a stale mypy-cache entry rather than from the tree)*. The PoC has exactly
  two phase files, `_python.py` and `_cython.pyx`. There is no `_cuda.py` and no
  `_numba.py`: `_numba` survives only as a string in that repo's `_backends.py` mapping,
  pointing at a module nobody ever wrote. So the third phase was never implemented there
  under any name, which is why the plugin's own `gpu-parallelization` skill is a stub.
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
        # `benchmark` is deliberately absent: pfsmgraph has no benchmark infrastructure at
        # all, where tokalign has benchmarks/run_benchmark.py. The benchmarking skill must
        # report an unconfigured command rather than assume one (A4).

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
- [x] Settle the fate of the `tokalign`-only components.
  > **Q:** What happens to the pre-commit hook?
  > **A:** Manifest-driven, still blocking, and scoped to commits that touch a kernel path.
  > **Q:** What happens to the three domain stub skills?
  > **A:** Delete all three.
  > **Q:** Where does the Cython/Python equivalence check live?
  > **A:** Not in a script — the phase skills write a property test into the consumer's suite.
  > **Done:** Settled 2026-09-08. **The hook is worse than "needs generalising", and this
  > was measured rather than reasoned.** All three of its commands fail in pfsmgraph:
  > `uv run --extra dev` errors outright (`Extra 'dev' is not defined in any project's
  > optional-dependencies table` — this family uses `[dependency-groups]`), there is no
  > `setup.py` because every member is meson-python, and `pytest tests/` collects **41**
  > tests where the suite is **280**, missing every `packages/*/tests/`. The script exits 2
  > on a failed pytest and exit 2 blocks, so loaded as-is here it would **block every
  > commit** — including the ones `/smart-commit` makes. Scoping it to kernel commits needs
  > no new manifest field: `[phases]` already knows which paths are kernels.
  - [x] **Delete `scoring-matrix`, `alignment-viz` and `package-release`.** Verified before
        deciding: each is a ten-line "TODO: implement this skill" over reference and
        example files that are *all zero bytes*, so nothing is lost. Scoring matrices and
        alignment visualisation are alignment-domain rather than DP-lifecycle, and
        `package-release` duplicates the consumer's own release path — pfsmgraph has
        `just release` and `docs/ops/release.md`. A named stub that never fills in is worse
        than no skill, because the skill list is the first thing a reader sees and it
        advertises a capability that does not exist. (`gpu-parallelization` is a stub of
        the same shape but is **not** deleted — C3 writes it.)
  - [x] **`validate-equivalence.py` is removed; the check becomes a property test the
        phase skills write into the consumer's own suite.** It then uses the consumer's
        types and encoder, lives beside the code it guards, and runs in CI rather than only
        when someone remembers a script — a check outside the suite is a check nobody runs.
        **The skill must know pfsmgraph's suite is not yet backend-parameterised**: ADR
        0003 also requires tests be written against the public API only, and
        `viterbi(params, record)` has nowhere to put a backend, so the two halves are
        jointly unsatisfiable until `align` brings a backend-selection API. Until then the
        test goes in a labelled, explicitly non-shared section, which is what
        `test_viterbi.py` already does. Generalising the script instead was rejected on
        what it would have to know: not paths and commands, but how to *construct valid
        inputs* for an arbitrary consumer's types — far more than a manifest can carry.
  - [x] **`run-benchmark.sh` is dropped; the benchmarking skill takes its command from the
        manifest.** The wrapper existed only to walk up to a project root and call
        `benchmarks/run_benchmark.py`, which `[commands]` now states directly.
        **`benchmark` is optional, and pfsmgraph has none** — no `benchmarks/` directory
        and nothing tracked matching it, against tokalign's `run_benchmark.py`. So the
        skill's absent-command path is not hypothetical: it is what the *only* consumer
        with a manifest will hit, and it must report that rather than assume a command.
  - [x] **The hook stays, but manifest-driven, still blocking, and scoped.** It takes
        `build` and `test` from `[commands]`, runs *only* when the staged set touches a
        path matching a `[phases]` template, and no-ops with a message when no manifest
        exists. Blocking is kept deliberately — the plugin's own `CLAUDE.md` argues hooks
        exist precisely because they always run where an instruction is advisory, and a
        wrong compiled kernel is what this one catches. Scoping is what makes blocking
        tolerable: today it runs the full suite before a docs-only commit. E3 implements,
        and must also settle the hook's coexistence with `/smart-commit` and
        `security-guidance`, whose hooks stack rather than override.
- [x] Settle phase-file naming for phases 3 and 4, where ADR 0016 left it open.
  > **Q:** Should one spelling serve the manifest key, the filename token and the
  > backend-matrix row?
  > **A:** Yes — `python`, `cython`, `cpu_parallel`, `cuda`, identical in all three.
  > **Q:** Should the hand-back close ADR 0016's Open section on naming, or leave it open?
  > **A:** Close it now.
  > **Done:** Settled 2026-09-08. **A2 had already done most of this work without saying
  > so**: once `[phases]` carries a path template per phase, the *filename* is data in the
  > manifest rather than a constant in the plugin, so "what do we call the phase-3 file"
  > stops being one question and becomes two — what the plugin calls the phase (a key), and
  > what a given consumer calls the file (a template). Only the first is global. One
  > spelling for the key means no mapping layer: a mapping with a single entry, existing
  > solely because `cpu-parallel` reads better in a session header than `cpu_parallel`, is
  > the kind nobody remembers when a fifth phase arrives.
  > **`_numba` is retired**, and the reason is worth keeping: it names the *library*, and
  > since ADR 0016 two phases share that library, so the name stopped distinguishing what
  > it was there to distinguish.
  > **ADR 0016's deferral is void rather than merely unmet.** It justified proposing
  > `_cpu_parallel.py` as "mirroring the existing `_python.py` / `_cython.pyx` / `_cuda.py`
  > per-phase naming" in the proof-of-concept, and defers settling until a real phase-3
  > kernel lands. Checked 2026-09-08: that repository has **two** phase files, `_python.py`
  > and `_cython.pyx`. No `_cuda.py`, no `_numba.py` — `_numba` exists only as a string in
  > its `_backends.py` mapping, pointing at a module nobody wrote. So the convention being
  > mirrored was never written, which is the same shape as ADR 0012's falsified premise:
  > not a deadline that has yet to arrive, but a reason that was never true. Hence closing
  > it rather than waiting. The settled form is `_<algorithm>_cpu_parallel.py` and
  > `_<algorithm>_cuda.py`, which matches what `packages/pfsmgraph-hmm/meson.build` already
  > does for `_viterbi_cython` and says it will do for `_baum_welch_cython`.
- [x] Settle how `dp-compile` detects `workflow-claude`, and what absence means.
  > **Q:** How is presence detected — the command reading its own tool list, a detection
  > contract between the plugins, or asking the user once per session?
  > **A:** The command reads its own tool list.
  > **Q:** What does absence mean?
  > **A:** Stop where the command delegates; note-and-continue where it does not.
  > **Done:** Settled 2026-09-08. Detection follows the precedent `/smart-merge` sets for
  > MCP — *there is no shell command that reports availability, you can see your own tools,
  > so judge from that* — so a command opens by checking for any `workflow-claude:` skill.
  > **Config inspection is ruled out empirically, not on principle:** `workflow-claude`
  > appears nowhere in `~/.claude/plugins/*.json`, and in pfsmgraph it is not an installed
  > plugin at all — it is a full plugin tree sitting in `.claude/skills/`, which is neither
  > a marketplace install nor the `--plugin-dir` its own README documents. **So the install
  > line a presence check prints must cover three load paths, not one**, and naming only
  > `--plugin-dir` would send a user to fix something that is not how they loaded it.
  > A detection *contract* — `workflow-claude`'s SessionStart hook writing a marker that
  > `dp-compile` reads — was rejected despite being the deterministic option. It assumes
  > hooks fire, which depends on the load path, and this session's load path is already the
  > non-standard one; it also couples two plugins forever for a check one of them can make
  > alone. The trick it would have generalised is real and worth recording, though:
  > `remind-disable-commit-commands.py` establishes that **a plugin's own hook firing is
  > itself proof that plugin is loaded** — but that proof is available only to the plugin
  > shipping the hook, which is exactly why `dp-compile` cannot borrow it.
  > **Absence is not uniform, and the earlier "never degrade" was too blunt.** No
  > `dp-compile` command strictly needs `workflow-claude`: `/phase-check` reports, and
  > writing a `.pyx` needs no branch or commit machinery. So the stop belongs where the
  > delegation is — `/next-phase` ending in `/smart-commit`, `/new-algorithm` opening a
  > branch — while read-only commands print one line and proceed. Refusing a report for a
  > dependency that report never uses reads as a bug the first time someone hits it.
  > Note for E2: **no current `dp-compile` command references `workflow-claude` anywhere**,
  > so this coupling is entirely new rather than an existing one being tightened.
- [x] Settle the branch and plan workflow *inside* each plugin repository.
  > **Q:** How should `workflow-claude`'s changes land — one standalone branch, a full
  > revision 05, or straight to `main`?
  > **A:** One branch via `/new-branch`, standalone, no revision.
  > **Q:** How should `dp-compile`'s rewrite land — focused commits to `main`, one branch
  > for the whole rewrite, or a branch per section?
  > **A:** Focused commits straight to `main`.
  > **Done:** Settled 2026-09-08, and the two repositories deliberately get *different*
  > answers because their conventions differ. Checked first: `workflow-claude`'s master
  > plan has **no open items** — revisions 02, 03 and 04 are all closed and archived — and
  > it states **no cleanup exemption**, so unlike pfsmgraph its written convention routes
  > everything through a PR. Straight-to-`main` was therefore rejected there specifically
  > because it would import pfsmgraph's rule into the repository that *defines* the
  > lifecycle for everyone else. A full revision was rejected on that repo's own
  > definition: a revision is a milestone "whose subgoals each spawn a branch", plural, and
  > this is one branch of documentation edits — the more so now that A6 rejected the
  > detection contract, which was the one change that might have needed code.
  - [x] **`workflow-claude`: one branch via `/new-branch`, `**Subgoal**: standalone`, then
        `/smart-merge`.** No revision opened. Accepted consequence, the same one this
        pfsmgraph branch carries: with no master-plan backlink, `/file-plans` will report
        `no backlink` and leave the branch plan flat. That is the designed safe outcome
        rather than a defect — and it is worth noticing that the repository defining that
        behaviour now exercises it on itself.
  - [x] **`dp-compile`: focused commits straight to `main`**, no branch and no plan
        machinery. Bootstrapping a two-tier plan convention into a repository being
        rewritten end to end would duplicate bookkeeping that `tmp/TODO.md` already holds —
        which is the entire reason that file exists. The cost is real and accepted: `main`
        is half-rewritten for the duration. It does not reach anyone, because
        `--plugin-dir` loads from the working tree rather than from a cached published
        version, so nothing stale is served while the rewrite is in flight.
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

- [x] Rename everything that carries the old identity.
  > **Done:** 2026-09-08, in two `dp-compile` commits — `3b7fc67` (repo, remote,
  > directory, `plugin.json`) and `8960b59` (27 namespaced references across seven files).
  > Acceptance met: `grep -rn 'tokalign-dev' --exclude-dir=.git .` returns nothing, and
  > `dev/smoke-test.sh` passes.
  - [x] **Done 2026-09-08 — `dp-compile` commit `3b7fc67`.** The user ran the rename
        sequence (`gh repo rename dp-compile`, `git remote -v`, directory move); verified
        afterwards: `tmp/dp-compile/` in place, `origin` at `dp-compile.git`, the clone
        clean, pfsmgraph still ignoring it, and **the old name redirects** — querying
        `PanosMavromatis/tokalign-dev` returns `dp-compile`, so nothing pinned to the old
        URL breaks. `plugin.json` landed in the same commit: `name` to `dp-compile` (it
        must match the directory name, asserted before committing), `version` to `0.2.0`,
        and a description that names no project and no longer says "prototype → Cython →
        GPU phases" — that was three phases, and the lifecycle has been four since ADR 0016
        inserted the CPU-parallel one. It also drops the packaging claim, whose skill is
        one of A4's three deletions.
        Not pushed yet. The `[commands]` and `[phases]` machinery the description implies
        does not exist yet either — B2 and B3 build it, and the description is a statement
        of the plugin's purpose rather than of what it can do today.
  - [x] **Done — 27 occurrences across seven files** (`dp-compile` commit `8960b59`).
        **This subgoal's own file list was wrong in both directions**, which is worth
        recording because it was written from a survey rather than a grep: `dev/smoke-test.sh`
        was named and carries *no* reference — it works in relative paths — while
        `skills/algorithm-prototype/SKILL.md` and
        `skills/cython-translation/scripts/validate-equivalence.py` were not named and do.
        Two places needed more than a substitution, because renaming alone would leave them
        self-contradicting: `CLAUDE.md`'s overview sentence would have read "dp-compile is
        a plugin ... for the `tokalign` package", and its `plugin.json` example block still
        carried the old description and `0.1.0` — an example disagreeing with the file it
        documents. Both now match the real manifest, asserted rather than eyeballed.
  - [x] **One pre-existing bug is preserved rather than fixed, because it dies with its
        file.** *(Resolved by A4's deletion; verified 2026-09-08 — the plugin-relative
        path appears nowhere.)* `cython-translation/SKILL.md` invokes the equivalence script as
        `dp-compile/skills/…`, a path assuming the plugin sits *inside* the consumer's
        repository rather than at `${CLAUDE_PLUGIN_ROOT}`. A4 removes that script, so
        fixing the path would be work on a file scheduled for deletion — but if that
        removal is ever reversed, the path is wrong and this note is where it is recorded.
  - [x] **Standing smoke-test warning, pre-existing and a false positive.** *(Settled at
        C1: live with it. Recorded in `CLAUDE.md` as expected-and-do-not-fix.)* Verified by
        stashing and re-running: `claude plugin validate` reports "No frontmatter block
        found" for `commands/references/phase-detection.md`, which is an `@`-included
        reference document rather than a command — but the validator scans every `.md`
        under `commands/`. Moving it would break the include that is its entire purpose.
        C1 rewrites that file; decide there whether to live with the warning permanently or
        restructure the include.
  - [x] **Acceptance met**: `grep -rn 'tokalign-dev' --exclude-dir=.git .` returns
        nothing, and `dev/smoke-test.sh` passes (one pre-existing warning, above).
- [x] Introduce the manifest (A2) and a shared `commands/references/manifest.md` that
      every command and skill reads first. **Done 2026-09-08 — `dp-compile` commit
      `88d231f`**, 215 lines.
  > **Q:** With no `dp-compile.toml`, should defaults reproduce `tokalign`'s layout as A2
  > settled, should the defaults be announced, or should the manifest be required?
  > **A:** Required, erroring with a template.
  > **Q:** Where does the worked pfsmgraph example live?
  > **A:** Embedded in `commands/references/manifest.md`.
  > **Done:** **This reverses A2's "defaults reproduce the `tokalign` layout", and the
  > reversal came out of implementing it.** The failure it would cause: in any repository
  > *other* than the one whose layout is baked in, the plugin resolves paths under a layout
  > that repository does not have, finds nothing, and reports "algorithm not found" —
  > misdiagnosing a missing manifest as a missing algorithm, which is the exact
  > silent-failure shape this plan keeps cataloguing. A2's supporting rationale had also
  > weakened since it was written: `.scratch/align-poc/tokalign` is *frozen reference
  > material* (deny-by-default, 11 files tracked, described in its own policy as the
  > proof-of-concept the ADRs derive from), not a project that would run the plugin, and
  > the live `tokalign` repo can gain a ten-line file trivially.
  > The example is embedded rather than filed under a new root `examples/` — the plugin has
  > no such directory, examples live only inside skills, and an example in the same file as
  > the prose describing it cannot drift from it. A second worked example (one directory
  > per algorithm) is included to demonstrate that the same four templates express both
  > layouts; that is the claim the whole design rests on, so it is shown rather than
  > asserted.
  > **Verified, not eyeballed:** all three TOML blocks parse, and every path in the
  > pfsmgraph example was resolved against this repository — the six invariant documents,
  > both encoder documents, the phase-1 kernel, `_backends.py`, `meson.build` and all three
  > oracles exist; the only four that do not are exactly the unwritten phases. The `cython`
  > template independently agrees with what `packages/pfsmgraph-hmm/meson.build` already
  > declares.
  - [x] **Discharged at C1** — the file was rewritten whole, and `src/tokalign` appears
        nowhere in it. *(Original note: deferred deliberately; `phase-detection.md` hardcodes
        `src/tokalign/algorithms/<name>/`.** Rewriting it here would mean rewriting it
        again at C1, which changes its phase table from three stages to five and replaces
        the effective-phase rule outright (A3). One rewrite, done once, at C1.
  - [x] **Second standing smoke-test warning.** *(Settled at C1 together with the first —
        and there turned out to be a third, the plugin-root `CLAUDE.md` notice. All three
        are recorded in `CLAUDE.md`.)* `manifest.md` joins `phase-detection.md` as
        a frontmatter-less `.md` under `commands/`, so the validator now warns twice. Adding
        frontmatter is *not* the fix — it would register these reference documents as
        commands. C1 decides between living with it permanently and restructuring where
        `@`-included references live.
- [x] Strip `tokalign` from the four real skills and the four commands — the structural
      couplings, not just the word. **Done 2026-09-08 in three `dp-compile` commits:
      `46faf7b` (commands), `78ff771` (algorithm-formalize), `bfcd03e` (prototype,
      test-patterns, cython, benchmarking).** Coupling in every B3-scope file is now zero,
      measured rather than asserted; the smoke test passes throughout.
  > **Q:** How does a skill learn the consumer's API shape — read it from the code, add
  > API fields to the manifest, or require the invariants documents to state it?
  > **A:** Read it from the code; ask when there is none.
  > **Q:** How far should the alignment-specific prose in `algorithm-formalize` be
  > generalised?
  > **A:** Generic principle, with alignment as one worked family.
  > **Done:** **Two instructions turned out to be wrong rather than narrow, and both are
  > reversed with the reason recorded in place.** "Normalize to max (similarity)" is
  > correct for one family and inverts every comparison in another — a decode accumulating
  > `-log2(p)` minimises a description length in bits, which *grows* as the probability
  > falls. And "the gap sentinel maps to integer 3" was **already false in the repository
  > the plugin was written for**, whose encoder was renumbered to 4 on 2026-09-01. Both
  > share a shape worth naming: a correct local observation promoted to a global rule.
  > Neither was wrong when written; each became wrong the moment a second consumer existed
  > — and the second became wrong without any second consumer at all.
  > Alignment examples are kept everywhere, relabelled as one family's instance rather
  > than deleted: an abstract template teaches nothing, and the *shapes* are what a reader
  > needs. `algorithm-prototype` now carries two illustrations chosen to sit as far apart
  > as two correct answers can — an alignment entry point and an HMM decode over a numeric
  > kernel — which makes "read the code rather than assume a shape" read as necessary
  > rather than pedantic.
  - [x] **Paths and layout → manifest**, in all four skills and all four commands.
        Algorithm-name resolution changed *shape*, not just its path: it looked names up
        as directories under a hardcoded root and now looks them up in `[algorithms]`,
        which is what lets both layouts share one code path.
        `phase-detection.md` is **excluded deliberately** — C1 rewrites it once, for five
        stages and the new effective-phase rule together, rather than twice.
  - [x] **Types and registration → the repository, read rather than declared.** The
        wrapper/kernel split is stated in its generic form and promoted from a convention
        to a rule, with the reason attached: the kernel neither validates nor raises,
        because the GPU phase implements the same kernel and a device function cannot
        raise a Python exception at all — so a kernel that validates is a kernel that
        cannot be transliterated. Backend registration likewise gained its second half,
        which fails *silently*: a build system that does not glob omits a source nobody
        listed, producing a module that imports in development and is absent from any
        built artifact.
  - [x] **Objective is a mandatory metadata row, and normalisation is now forbidden
        rather than required.** The gotcha that taught converting to max now teaches why
        converting is unsafe — it is not flipping an operator, since the recurrence
        structure, the base cases and the identity element all change with the semiring,
        and a silent conversion is indistinguishable from a bug.
  - [x] **Every hardcoded reserved code is gone**, from the formalize gotchas, the
        prototype gotchas and the TC-XX translation rules alike. Reserved symbols are
        named as sentinels with their integers deliberately unstated, and tests are told
        to ask the encoder which integer a sentinel is rather than write the number.
  - [x] **Build and test commands → `[commands]`.** Every `setup.py build_ext` and
        `--extra dev` is gone from the skills and their references.
  - [x] **`test-patterns.md` restructured around both models plus the third state**
        (204 → 267 lines). Writing it up sharpened the argument for the registry beyond
        "pfsmgraph does it this way": **import auto-discovery cannot distinguish "not
        written yet" from "written but broken"**, because a compiled phase whose extension
        failed to build raises `ImportError` exactly like one that was never written — so
        a stale build reads as an absent phase and the suite goes green having tested less
        than it did yesterday. That is the gap a registry row's `hardware` field closes.
        The tie-break-is-contract corollary is stated there too, since it is the rule a
        parameterised suite depends on and the one most easily mistaken for an
        implementation detail.
  - [x] **Alignment specifics are now labelled examples rather than assumptions**, in
        every place they appear — the formalize assumptions step, the prototype signature
        block, the Cython wrapper/kernel pair, and the whole `test-patterns.md` worked
        template. Deleting them was rejected: an abstract template teaches nothing, and
        the shapes are exactly what a reader needs.
  - [x] **Still coupled, and out of B3's scope by design** *(all five now resolved: both
        references rewritten at B4, both examples carry the one-family label — verified
        2026-09-08 — and `package-release/` was deleted by A4.)* — `pseudocode-conventions-DP.md`
        and `formalization-template.md` (the next goal rewrites both),
        `example-formalization.md` and `example-cython-impl.pyx` (kept as one family's
        instance, but their headers need the label), and `package-release/SKILL.md`
        (deleted by A4 in the mechanical tail).
- [x] Generalise the formalization conventions and template. **Done 2026-09-08 —
      `dp-compile` commit `14affb3`.**
  > **Q:** The metadata table is defined twice, in the skill's Step 5 and in
  > `formalization-template.md`. How should they relate?
  > **A:** The template is canonical; the skill points at it.
  > **Q:** What happens to the conventions doc's project-specific section?
  > **A:** Split it — universal rules stay, family notation becomes an example.
  > **Done:** **The duplication had already opened before it was closed**, which is the
  > argument for closing it: the skill's copy carried the new `Objective` and provenance
  > rows from B3 while the template still carried `Alignment type` and `Gap model`. The
  > split of the conventions doc turned on which rules are universal — the encode/decode
  > boundary note applies in every repository and to every family, because it is what
  > makes the later phases transliterations, so it was promoted rather than demoted with
  > the rest.
  > **One terminology change, made for a reason specific to this plugin:** generic prose
  > now says *backtrace* rather than *traceback*, because in a plugin whose subject is
  > Python code "traceback" already means an exception's stack trace, so "check the
  > traceback" is ambiguous exactly where precision matters. The alignment examples keep
  > their own field's word, with a note explaining the two.
  - [x] **Conventions doc de-coupled, DP-only scope kept**, including the rule that a
        non-DP algorithm is reported rather than forced into the template. Its
        "tokalign-Specific Conventions" section is now two: a universal encode/decode
        boundary requirement, and family-specific notation carrying alignment and HMM
        examples side by side — the HMM one making emission *arity* explicit, since an
        arc-emitting model's emission depends on both endpoints and so cannot be hoisted
        out of the inner loop, which changes the recurrence rather than the notation.
  - [x] **Template rewritten and made canonical** (108 → 149 lines). `Alignment type` and
        `Gap model` become `Family` and `Variant`; `Derived from`, `Objective` and
        `Parallel decomposition` are added. Parallel decomposition is a **section**, not
        just a row, because C2 needs the *argument* for why cells may be computed
        concurrently rather than the name of a decomposition — and "undetermined" is
        stated as legitimate, since a guess there becomes a race rather than an error.
        The TC-XX discipline and its three precision levels are kept, and gain one
        requirement: **at least one deliberate tie**, because learned parameters rarely
        tie and a suite drawn only from real data can pass while the tie-breaking rule is
        wrong.
  - [x] **Both worked examples kept and labelled** — the Needleman-Wunsch formalization
        and the Cython wrapper/kernel pair. The formalization example's header also
        records that **it predates the current template**, so a reader does not mistake
        its section list for the canonical one; that is cheaper than regenerating the
        example and more honest than leaving the discrepancy silent.
  - [x] **What still names the old project, and why each is fine:** *(now two files, not
        six: `README.md`, which E5 owns, and `manifest.md`, where it is intentional. The
        other four were deleted or rewritten.)* `README.md` (E5
        rewrites it), `phase-detection.md` (C1), `validate-equivalence.py`,
        `run-benchmark.sh` and `package-release/SKILL.md` (A4 deletes all three), and
        `manifest.md` — where it is *intentional*, since the second worked manifest is
        that repository's.
- [x] The mechanical tail. **Done 2026-09-08 — `dp-compile` commit `8430847`.** A4's five
      deletions executed (three stub skills, `validate-equivalence.py`,
      `run-benchmark.sh`); the smoke-test array rebuilt; `CLAUDE.md` de-templated and
      given the commit convention in writing; `claude plugin validate` green.
  > **Done:** **The `if` pattern question could not be answered, so the hook was rebuilt
      not to depend on it.** The two official plugins using that field spell their
      patterns incompatibly — `Bash(git commit:*)` against
      `Bash(python3 *scripts/*.py *)` — so at least one syntax matches nothing, and a
      matcher that matches nothing yields a hook that never fires *and never says so*. The
      deciding evidence: `security-guidance`, which uses the colon form, **re-checks the
      command with its own regex anyway** (`security_reminder_hook.py:663`, "Mirrors
      Claude Code's own commit…"). So the hook is now Python, reads the `PreToolUse` JSON,
      and decides for itself; the `if` is gone rather than trusted.
  > **The hook's logic is tested, and the test caught a defect before it shipped.**
      Deriving kernel paths by wildcarding `_{algorithm}.py` matched *every* private
      module in a flat package — `_numeric.py` and `_params.py` alongside `_viterbi.py` —
      so an edit to a helper would have run the whole suite. It now expands templates over
      the manifest's listed algorithms, which is the same reason A2 gave for listing them
      rather than globbing. That reason was stated as a resolution concern; it turns out
      to bite anywhere a phase template is turned into a matcher.
  > **Two things `CLAUDE.md` said were stale rather than merely thin**: its gotchas
      described mtime staleness, which A3 replaced, and it described no manifest at all.
      Both are now written up alongside the `commands/references/*.md` warning, so nobody
      "fixes" the validator complaint by adding frontmatter and turning two `@`-included
      fragments into invocable commands.
  - [x] **Discharged at C2**, and it gained the reference file too. *(Original note:
        deferred; the smoke-test array gains
        `skills/cpu-parallelization/SKILL.md` when that skill exists. Adding it now would
        make the smoke test fail on a file nobody has written.

## Section C — Implement the ADR 0016 chain

- [x] `phase-detection.md` and its three consumers for five stages.
  > **Done:** 2026-09-08. `phase-detection.md` rewritten whole (45 → 195 lines), and the
  > deferrals from B2 and B3 discharged in that one pass as intended.
  > **The rule was simulated before it was committed**, seven cases, and the seventh found
  > a hole the prose had: *an artifact whose named source is absent*. A `cuda` kernel
  > written straight from the `.pyx`, or a gap filled out of order, leaves a header naming
  > a file that does not exist — nothing to hash, so neither fresh nor stale, and not a
  > target either since it cannot be regenerated from a source that is not there. Folded
  > into `blocked`, whose definition is now "an ancestor is stale **or absent**". Six of
  > seven cases had passed; the rule was wrong only where nobody would have looked.
  > **Two claims are now measured rather than argued.** In a recovered graph, editing the
  > root kernel marks the formalization *and* the `.pyx` stale simultaneously — reported
  > together, doc first — and editing the recovered *document* marks nothing at all. The
  > second is the case the old rule got backwards, and it is now checked.
  > **A dangling reference is live and deliberate.** `/next-phase`'s routing table and the
  > README both name `dp-compile:cpu-parallelization`, which C2 writes. Naming it now was
  > the alternative to writing the table twice; C2 closes it, and until C2 lands a phase-3
  > target routes to a skill that does not exist.
  - [x] Nominal-phase table 0–4; filenames from the manifest; staleness per A3.
        > Also settled the validator question C1 inherited: **three** warnings, not two —
        > the third is `CLAUDE.md at the plugin root is not loaded as project context`.
        > Correct about an installed plugin, beside the point for a file that is context
        > for developing the plugin in the repository where plugin root *is* repo root.
        > All three are now recorded in `CLAUDE.md` as expected-and-do-not-fix.
  - [x] **Effective phase needs redefining, not extending — A3 changed its shape.** The
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
  - [x] `/next-phase` routes 2 → 3 to the new CPU-parallel skill and 3 → 4 to GPU;
        `/phase-check` reports four backends; `/benchmark` counts them.
        > `/phase-check` no longer reports a single phase number at all — under a graph
        > there is not always one, so it reports the nominal phase plus a per-artifact
        > state table. `/benchmark` gates on `blocked` as well as `stale` (an ancestor
        > moved, so the number describes code on its way out) but explicitly **not** on
        > the formalization, which compiles to nothing and contributes no timing.
        > Four files beyond the three named consumers had to change or they would have
        > contradicted the new reference: `skills/benchmarking/SKILL.md` still described
        > mtime staleness; `gpu-parallelization`'s trigger description still said
        > "Phase 3" and `_numba.py`, which would have misfired against the new routing;
        > `test-patterns.md` named a test file `_numba_specific.py`, a library name where
        > a phase name belongs; and `README.md` carried the whole superseded model.
  - [x] Every "do not create X yet" list in the formalize, prototype and Cython skills
        names the two new files.
        > `cython-translation` had no such list at all — added one. The lists now name
        > phases rather than filenames, since the filenames are the manifest's.
- [x] New skill `cpu-parallelization` (phase 3: `@njit(parallel=True)` with `prange`).
  > **Q:** Phase 3 makes `numba` a hard runtime dependency and phase 4's `numba-cuda` goes
  > behind an extra, but the manifest has no way to say where a dependency is declared.
  > Add an optional `[dependencies]` table, have the skill report-and-stop with no table,
  > or reuse `[backends].build_manifest`?
  > **A:** Add an optional `[dependencies]` table.
  > **Done:** 2026-09-08. `skills/cpu-parallelization/SKILL.md` (1790 words) plus
  > `references/decomposition-patterns.md` (1235). The dangling
  > `dp-compile:cpu-parallelization` reference C1 left is closed.
  > **The plan assumed a manifest feature that did not exist** — this subgoal says the
  > dependency is "declared via the manifest's dependency recipe", and there was no such
  > recipe. `[dependencies]` is now `declared_in` plus one inline table per parallel
  > phase, where the absence of an `extra` key *is* the hard/optional distinction. Absent
  > table ⇒ report and stop, on the `benchmark` precedent. Rejecting
  > `[backends].build_manifest` mattered concretely rather than aesthetically: in the one
  > repository with a manifest it is `meson.build`, which declares build *sources*, while
  > runtime dependencies live in `pyproject.toml` — same package, different file.
  > **Verifying C2 found a defect in the C1-era hook.** `kernel_paths()` expanded *every*
  > `[phases]` template, so the formalization counted as a kernel and staging a prose edit
  > rebuilt and ran the whole suite. Phase 0 compiles to nothing and is imported by
  > nothing, so it cannot break a backend; excluded, and re-checked at four kernels.
  > **`[dependencies]` is inert to the hook** — verified rather than assumed, since a new
  > top-level table is exactly the kind of addition that leaks into a `.values()` loop.
  - [x] The decomposition comes from `FORMALIZATION.md`'s Parallel decomposition section,
        never invented here. Anti-diagonal for 2-D alignment DP; for a 1-D-over-time
        recurrence with dense state coupling (HMM) the candidates are
        states-within-a-timestep, batch over sequences, or an associative scan in the
        (min, +) semiring — and "no anti-diagonals" is a valid answer the skill must
        accept, because master plan line 194 settles that question against the Viterbi
        kernel, not here.
        > The reference separates two outcomes a plan tends to collapse: **"undetermined"
        > blocks the phase** (the question is unanswered, so amend the formalization
        > first), while **"not usefully parallel" completes it** (a `prange` over four
        > states is slower than a `range` over four states). Without that split a
        > repository reaching the honest second answer would either have to overstate it
        > or appear to have skipped a step. Three families are written up — 2-D grid with
        > neighbour dependencies, 1-D over time with dense state coupling, 1-D with a
        > bounded window — and the second says plainly that the wavefront question does
        > not apply rather than that it is hard.
  - [x] Race-freedom is checked against the phase-1 oracle (same discipline as every
        phase), and **the tie-breaking rule is contract** (`core.md`): the per-cell
        reduction over predecessors stays sequential so first-wins survives `prange`.
        > Three checks, none optional: the suite; **thread-count invariance**
        > (`NUMBA_NUM_THREADS=1` against many — a disagreement is a race, never a
        > tolerance problem); and a **constructed tie**, since learned float parameters
        > essentially never tie and a green differential run is evidence about the corpus.
        > Written alongside it because it is the same failure wearing different clothes: a
        > float *sum* reduced by `prange` is reassociated and not bit-identical, so a
        > forward or posterior recurrence needs a tolerance where a min/max Viterbi does
        > not.
  - [x] `numba` becomes a hard runtime dependency of the package (ADR 0016), declared via
        the manifest's dependency recipe; `numba-cuda` does not.
  - [x] Registration: a backend row with `hardware=None` (a failed import escalates).
        > Stated with its reason rather than as a rule: the hard/optional split *follows*
        > from the registry. A registered backend that will not import is a hard failure,
        > so putting numba behind an extra would make every install without that extra a
        > broken one.
- [x] Write `gpu-parallelization` (phase 4) for real — currently a stub over zero-byte
      references.
  > **Done:** 2026-09-08. SKILL.md 10 lines → 1908 words, plus
  > `references/numba-cuda-patterns.md` (1307) and `examples/example-cuda-impl.py`.
  > `anti-diagonal-parallelism.md` deleted per C2's note, and `example-numba-impl.py`
  > renamed to `example-cuda-impl.py` — the same library-versus-phase naming rule that
  > retired `_numba.py`, applied to the last file still carrying it.
  > **The structurally important CUDA fact is a code shape, not a performance note.**
  > `cuda.syncthreads()` synchronises a block, not a grid, so there is no grid-wide
  > barrier inside a kernel and the decomposition's sequential axis has to move to the
  > host as a loop of launches. The failure mode is nasty in a specific way and the skill
  > says so: keeping that loop inside the kernel is correct *whenever the grid fits in one
  > block*, so it passes the small test and fails at exactly the size the GPU was for.
  > **A GPU is not needed to write or test this phase.** `NUMBA_ENABLE_CUDASIM=1` runs
  > `@cuda.jit` code on the CPU — evidence about indexing, bounds and barrier placement,
  > and about nothing else, since it models neither coalescing nor divergence nor races
  > between blocks. The skill requires saying that a result came from the simulator,
  > because a green simulator run reads exactly like a validated kernel.
  > **One more manifest key, my call rather than a question**, since it applies the
  > pattern C2 settled rather than introducing one: `[backends].require_env`, optional,
  > naming the variable that escalates a hardware skip to a CI failure. Without it the
  > registration step could only tell the user *that* the backend will skip, not how to
  > stop it — and guessing a variable name is worse than saying nothing, because an
  > instruction that does nothing looks like it was followed. Verified against the real
  > `_backends.py`: `REQUIRE_ENV` at line 41, the `hardware` escalation at 99–106.
  - [x] `numba-cuda` behind the consumer's GPU extra (ADR 0004); backend row with
        `hardware="CUDA device"`, so absence skips loudly; CI escalation is the
        consumer's (`PFSMGRAPH_REQUIRE_BACKENDS`-style) and comes from the manifest.
        > The hard/optional split is written as a *consequence* at both phases rather
        > than as two rules: phase 3 registers with no hardware requirement, so a failed
        > import is a hard failure and the dependency must be in every install; phase 4
        > requires a device, so its absence is a skip and a base install with no GPU has
        > to remain a working library. Stated that way the two are one decision seen from
        > opposite ends, and the skill can forbid "just add it to the hard list for now"
        > as the silent inversion it is.
  - [x] Reuse phase 3's decomposition verbatim; what is left to debug is
        hardware-specific — coalescing, occupancy, `cuda.jit` semantics — and the skill
        says so, so it does not re-derive the algorithm.
        > Written as the reason ADR 0016 inserted phase 3 at all: before it, a first GPU
        > attempt debugged the decomposition and the hardware at once, and both produce
        > the same symptom — an intermittently wrong answer. The skill routes a suspected
        > decomposition error *back to phase 3* rather than fixing it in place, since a
        > fix here leaves the two backends implementing different arguments with only the
        > equivalence suite holding them together.
        > One case the subgoal did not anticipate: a manifest that declares no
        > `cpu_parallel` phase. Then phase 4 derives from `cython` and the decomposition
        > has never been validated under concurrency — a real position for a repository
        > to take, and the skill says so plainly up front rather than proceeding as if
        > phase 3 had happened.
  - [x] Fill `references/numba-cuda-patterns.md` and the example. **`anti-diagonal-
        parallelism.md` should be deleted rather than filled** (note from C2, 2026-09-08):
        C2 wrote `skills/cpu-parallelization/references/decomposition-patterns.md`, which
        covers the wavefront *and* the families that have no anti-diagonals. Phase 3 is
        where a decomposition is decided and validated and phase 4 reuses it, so phase 4's
        reference should point there. Two copies of a decomposition argument is the exact
        doc-rot shape this plan has been cataloguing, and the zero-byte file makes the
        deletion free. What stays phase-4-only is genuinely hardware: coalescing,
        occupancy, `cuda.jit` semantics, and the host/device transfer boundary.
- [x] Benchmarking over four backends: JIT warm-up applies to phase 3 as to phase 4;
      crossover reporting names all three transitions; command from the manifest.
  > **Done:** 2026-09-08. All three landed, and the goal turned out to be mostly a
  > *removal*: C1 had already done the crossover reporting and the four-backend count, and
  > B5 the manifest command, but the bottom half of `skills/benchmarking/SKILL.md` still
  > described tokalign's own harness — `results.md`/`scaling.png`/`results.json` as
  > guaranteed outputs, a `--memory` flag, `alphabet.symbols`, an alignment-specific
  > scoring setup, and gotchas about matplotlib and `memory-profiler`. It contradicted the
  > "output layout belongs to the repository" principle stated at the top of the same
  > file. Rewritten from `### 2` to the end.
  > **The JIT subgoal hid a third case.** "Warm-up applies to phase 3 as to phase 4" is
  > true and incomplete: `cache=True` moves compilation to the first run *of the process
  > ever*, not of the session, so a harness measuring compilation overhead gets a different
  > answer depending on whether a cache file happens to exist on disk. The cache state is
  > therefore part of the measurement rather than of the environment. And a warm-up is per
  > *(backend, signature)* — a new input dtype triggers a fresh compile, so one warm-up at
  > the top of a size sweep does not cover the sweep.
  > **A crossover is a property of the machine, not of the algorithm**, and both commands
  > now say so. `cython → cpu_parallel` moves with core count and `cpu_parallel → cuda`
  > with the device, so the same kernels on a laptop and a workstation disagree about which
  > backend to ship and both are right. This is the doc-rot shape already catalogued twice
  > in this plan — a correct local observation promoted to a global rule — caught before it
  > was written rather than after: "Cython wins below 500" becomes false the moment it
  > reaches a README as a fact about the algorithm.
  > **One invented flag removed from `/benchmark`**: it appended `--memory` on request,
  > which assumes the consumer's harness accepts it. On a harness that errors that is
  > noise; on one that ignores unknown arguments it silently returns an ordinary timing run
  > labelled as a memory profile.
- [x] Equivalence discipline per phase, written into the consumer: extend the
      parameterised suite where one exists, or add to the labelled non-shared section
      where it does not (pfsmgraph until `align`'s backend-selection API), plus the
      property test that replaces `validate-equivalence.py` (A4).
  > **Done:** 2026-09-08. Written once, in
  > `skills/algorithm-prototype/references/test-patterns.md` under **Equivalence
  > discipline, phase by phase**, with the three phase skills pointing at it rather than
  > restating it. That file already owned the question of how a repository's suite sees
  > backends (Models A and B, and the third state), so the discipline belongs beside it.
  > **The organising idea made the section small: the discipline is cumulative, and each
  > phase adds exactly one new class of failure.** Phase 1 is the oracle; phase 2 adds
  > boundary and type errors from translation; phase 3 adds races and a tie-break lost to
  > concurrency; phase 4 adds barrier placement and device-only behaviour. Stated that way
  > it is a four-row table plus one paragraph each, and — the reason it is stated once —
  > three copies of a *cumulative* rule cannot stay in step, since each copy would have to
  > repeat everything above it.
  > This also resolves a phrasing that had crept into two skills: the phase-3 and phase-4
  > checks read as complete lists, when they are additions to what the parameterised suite
  > already runs against a newly registered backend.
  > **Two stale claims found in `test-patterns.md` and corrected**, both of the shape this
  > plan keeps cataloguing. Its TC-XX section still said "indices 0-3 are reserved and user
  > symbols start at index 4" — the `tokalign` numbering that was renumbered onto ADR 0011
  > in September 2026, and the same claim B4 removed from `algorithm-formalize` while this
  > copy survived. Replaced with the rule rather than the numbers: ask the encoder, never
  > hardcode, because a hardcoded code addresses a real position and returns a confident
  > answer computed from the wrong symbol. And the parameterisation `ids` example still
  > labelled backends `"numba"`.

## Section D — Legacy-code entry: recovering a formalization from an implementation

The `hmm` case: the reference implementation was translated from Lush, no natural-language
or pseudocode statement of the algorithm ever existed, and the closest thing to a
specification is `HMMLIB-ACCOUNT.md` plus three `.vpath.xls` oracles the original wrote.
The plugin's chain assumes prose → pseudocode → Python; this section gives it a second
entrance so every algorithm ends up with a pseudocode reference regardless of where it
started.

- [x] New skill (name to settle: `algorithm-recover`, `formalize-from-code`, or a mode
      of `algorithm-formalize`).
  > **Q:** Separate skill `algorithm-recover`, separate skill `formalize-from-code`, or a
  > mode of `algorithm-formalize`?
  > **A:** Separate skill `algorithm-recover`.
  > **Done:** 2026-09-08 — `dp-compile` commit `9617581`. **The name was the smaller half
  > of the question; the shape was settled by the frontmatter.** A skill's `description`
  > is a trigger, and `algorithm-formalize`'s ends "do NOT trigger if the phase-1 file
  > already exists" — which is the exact condition under which the reverse entrance must
  > fire. One block cannot carry both honestly, and the description is what decides
  > whether a skill runs at all, so a mode was ruled out on a mechanism rather than on
  > taste. Word budget was not the constraint: `algorithm-formalize` is 3127 words against
  > the 5000-word cap, so a mode would have fitted.
  > **The skill's spine is a hazard the plan had not named**: a document recovered from
  > code *describes* the code, and therefore cannot contradict it. Phases 2-4 translate
  > from the formalization, so a document translated from phase 1 adds a step and no
  > independent check, and a phase-1 defect ends up written down as the specification with
  > the document's authority behind it. Three things prevent the collapse — independent
  > evidence, stating intent rather than mechanism, and the review — and where none is
  > available the document must say so in `Notes` rather than sound confident.
  > **`[algorithms.<name>].oracles` already existed** (A2), including the line "an oracle
  > is not necessarily an equality check". So D1 consumes it rather than inventing it, and
  > D3 is left with making it first-class *downstream* rather than at the manifest.
  > **Two consequential edits outside the skill.** `/next-phase`'s routing table had one
  > phase-0 row; it now splits on whether the phase-1 file exists, because with a kernel
  > present the forward skill has no source material and would describe it. And the smoke
  > test's `expected` array never listed `formalization-template.md` or
  > `test-patterns.md` — both are referenced from several skills, so losing either breaks
  > skills whose own directory still looks complete. Both are listed now.
  - [x] Input: an existing phase-1 implementation, optionally the legacy source it was
        translated from and any differential fixtures. Output: `FORMALIZATION.md` with a
        provenance header (A3) naming the code as its source.
        > **Note:** the skill also names the three evidence cases explicitly — legacy
        > source present, only an account of it present, neither present — because which
        > one holds changes what the document may claim, and a recovery that does not say
        > which it was in is unauditable.
  - [x] Section mapping: "Source Pseudocode" becomes "Source implementation" (cited by
        path and commit); "Adaptations from Source" becomes "Deviations of the
        implementation from the legacy source" — for `hmm`: the δ-seeding fix, the `-1`
        log-zero sentinel not reproduced, `psi` as `int64`, `safe_divide`'s zero
        convention; Objective states the semiring the code actually uses (min-sum over
        bits); TC-XX cases are *extracted* from the existing tests and fixtures, not
        invented.
        > **Note:** the mapping is written **into the template**, in place, rather than
        > duplicated in the new skill. The template is canonical for structure, and two
        > entrances sharing one structure is the property that keeps them from drifting
        > into two different documents.
        > **The plan's four `hmm` examples are four different kinds**, which is what
        > `references/deviation-taxonomy.md` generalises: a defect fixed (δ-seeding), a
        > defect not reproduced but harmless (`psi`), a representation replaced (`-1` →
        > `+inf`), and a convention deliberately preserved (`safe_divide`). `core.md`
        > supplies two more — a primitive substituted with different failure behaviour
        > (`LU-solve` → `numpy.linalg.solve`) and a repository invariant imposed (the ADR
        > 0011 renumbering). **Only the first kind reaches the recurrence**, and the
        > recurrence then states the *fixed* behaviour; writing the original's in would
        > specify the bug and make every later phase reproduce it.
        > **The last kind is the one a diff cannot show.** A convention matched on purpose
        > leaves no trace in the code, the diff or the tests, so to a later reader it looks
        > like an oversight — which is precisely what a cleanup undoes. Recording only the
        > differences loses exactly the entries the section exists for.
  - [x] Same hard stop for human review as the forward skill; the recovered document
        becomes the living specification from then on, and later edge cases fold into it.
        > **Note:** the stop is stricter here than in the forward skill, and says so.
        > Beyond creating no implementation artifacts it forbids *editing* the kernel it
        > just read — including a docstring "correction". A recovery that finds a real
        > defect reports it; folding the fix into a documentation commit hides the one
        > finding the recovery was for. The user-facing line also says **why** the review
        > matters more on this entrance: a forward formalization can be checked against its
        > source paper, whereas this one was read off the very code it now governs.
- [x] `/new-algorithm` gains an entry-mode question: from source material (forward), from
      an existing implementation (reverse), or from existing pseudocode (skip to phase 1
      with the supplied `FORMALIZATION.md`). Phase detection treats a recovered
      formalization as derived, so phase 1 is not stale (A3).
  > **Done:** 2026-09-08 — `dp-compile` commit `a3dacdb`. **It is not a three-way
  > question, and that is the design rather than a shortcut.** One branch is already
  > settled by the filesystem: if the phase-1 file exists the entrance is reverse, since
  > `algorithm-formalize`'s trigger excludes that case outright and with a kernel present
  > it would end up describing it. The command states the inference and asks for
  > confirmation — which also catches a path template that resolved somewhere unexpected,
  > before a skill starts writing. A genuine question survives only where no phase file
  > exists, and it is between two options.
  > **Registration moved into the command.** It previously only *checked* whether
  > `[algorithms.<name>]` existed and never said who writes it — a gap in every mode, made
  > acute by the reverse one, where the code existing is exactly why nobody registered it.
  > Since algorithms are listed and never discovered, an unregistered algorithm is
  > invisible to `/phase-check` and `/next-phase` however many of its files exist, so this
  > is most of what "scaffold" means here. The write is proposed and confirmed rather than
  > silent: `dp-compile.toml` is the repository's contract with the plugin.
  > **The supplied mode skips the authoring of phase 0, not its gate.** `Derived from`
  > stays empty (it came from outside; nothing here can make it stale), the document is
  > checked against the template — read as canonical rather than restated in the command —
  > and against `[project].invariants`, and it is never reformatted into the template's
  > voice. Two absences are named because neither fails now: no `Parallel decomposition`
  > costs nothing until phase 3 stops on it, and no `Test Cases` leaves phase 1 inventing
  > the specification it was meant to translate.
  > **The parenthetical named a live defect, not a property to preserve.** The mtime
  > fallback said a headerless file's source is the previous declared phase. A recovered
  > algorithm's `_python.py` is correctly headerless — it is the root — and the
  > formalization's header names it, so the fallback added a *reversed* edge on top of a
  > recorded one: a cycle. And because a recovered document is necessarily newer than the
  > kernel it was written from, mtime would report the kernel stale the moment the recovery
  > landed, then regenerate it from a description of itself. Reachable as soon as D1
  > shipped. The fix is general rather than a recovery special case — **the fallback may
  > never invent an edge that reverses a recorded one**, and a file that no header names
  > and that carries none of its own is a root that nothing can make stale.
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

- [ ] Presence check (A6) at the head of every `dp-compile` command: check the tool list
      for any `workflow-claude:` skill. **Absence stops only the commands that delegate**;
      read-only ones print one line and continue (A6 corrected the earlier blanket "never
      degrades", which would have had `/phase-check` refuse for a dependency it never uses).
  - [ ] The README's "Prerequisites" must name **three** load paths, not one — a
        marketplace install, `claude --plugin-dir <path>/workflow-claude`, and a plugin
        tree placed in the project's `.claude/skills/`, which is how pfsmgraph itself loads
        it today. A message naming only `--plugin-dir` sends a user to fix something that
        is not how they loaded it.
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
- [~] `workflow-claude` edits, expected to be small: README "Companion plugins" pointer;
      conflict-report section; a `CLAUDE.md` line. Record any functional change that
      turns out to be needed (a documented detection contract, say) rather than slipping
      it in. Landed through its own revision (A7).
  > **Note:** branch `docs/plan-notes-and-interop` opened 2026-09-08 from `main` at
  > `59661bc` and pushed, per A7's "one standalone branch, no revision". Commit `d6b9318`
  > carries the first change; the three interop edits above are still to come on the same
  > branch, then `/smart-merge`.
  > **A functional change was needed, and this is the record E4 asked for rather than the
  > slipping-in it warned against.** Neither `/step` nor `/hitl-step` documents a way to
  > record a **finding**. Their whole annotation vocabulary — `Q`, `A`, `Blocked`,
  > `Deferred`, `Descoped`, `Done`, `Commit`, `Ran`, `Result` — describes the state of a
  > *task*, so an observation that is not a task has no category and gets written as
  > `- [ ]`. `> **Note:**` is already the convention in practice, including in
  > `workflow-claude`'s own plans, and was written down nowhere — so an agent following the
  > protocol cannot discover a form that exists only in files it is not reading.
  > Found by using the commands on this very plan: seven findings in Section B were written
  > as checkboxes and reported as outstanding work across two completed sections.
  > **`/hitl-step` would have prevented the misreport even without the fix.** Its Step 4
  > forbids flipping a parent to `[x]` while any subgoal is `[ ]`. The protocol was correct
  > and simply was not run — "go ahead with C1" bypassed it. So the gap and the incident
  > have different causes, and both are now addressed: the missing category, and two checks
  > that were stated but not made mechanical (Step 4 now says to re-read the subgoals
  > rather than recall them; Step 6 re-derives a section-completion claim at every indent).
  > **The live copy stays on `main` deliberately.** `.claude/skills/workflow-claude/` is a
  > second clone of the same repository, and pointing it at an unmerged branch would put
  > unreviewed work into the load path. It updates when the branch merges — and E6, which
  > replaces that copy with a symlink, is what stops this being a standing question.
- [ ] `dp-compile` README rewrite: the combined workflow (`/open-revision` →
      `/new-branch` → `/new-algorithm` and `/next-phase` inside `/hitl-step` →
      `/smart-commit` → `/smart-merge`), the manifest, the five-stage table under ADR 0016
      numbering, the two entrances (D), prerequisites, and the removal of the `tokalign`
      "Decisions" and "Naming" sections.
  > **Note from C1 (2026-09-08):** the README's *phase model* was corrected in place —
  > the five-stage table, provenance-not-mtime, the stale/blocked worked example, the
  > skills table (three deleted skills were still listed) and the directory tree (still
  > showed `pre-commit-check.sh`). Those were false claims rather than rewrite material.
  > What is deliberately left for this goal: the opening line and §"Key Architectural
  > Patterns", both still written about `tokalign` and its `Alphabet` class; the
  > "Decisions" section pointing at `../docs/decisions/adr/` in a repository layout that
  > no longer exists; and the "Naming" section, which describes a `tok` prefix rename via
  > the `package-release` skill that A4 deleted.
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
  - [ ] Master plan lines 192–196 (phases 2–4 of Viterbi): note they run under
        `dp-compile`, so the first real use of the revised plugin is on the kernel it was
        revised for.
  - [ ] Marketplace: nothing to build yet — **verified 2026-09-07, no `marketplace.json`
        exists in any of the user's repositories**, and `claude-plugin-tools` is unrelated.
        Note in both READMEs that the `--plugin-dir` line is interim.
