# Scratch — migration working area

**Retained across branches (changed 2026-08-31).** This directory was created as a
temporary `dataseq` working area and was to be deleted by the last goal of
`docs/plan/feat-dataseq-merge/TODO.md`. It is not: the same imports are the
migration source for `hmm` and `align` 0.1.0, so the tree stays for as long as the
migrations need it. What changes per package is not the *contents* but the
**`.gitignore` policies** — each import's rules are re-scoped to surface the files
relevant to the package being migrated, so the tracked set follows the work.

The four imported implementations are read side by side before anything is merged
into `packages/pfsmgraph-dataseq/`. Nothing in this directory is part of any
distribution, and nothing outside it may import from it.

## Why the leading dot

It is load-bearing, not cosmetic. `uv run pytest` has no `testpaths` configured, so
its rootdir walk reaches everywhere; a half-translated Lush file that fails at import
would surface as a *collection error* and fail the run before any test executes. The
dot matches pytest's default `norecursedirs` entry `.*`, so this tree is never walked
— with no configuration to add now or remove later. `_scratch/` would not work: the
default list contains `_darcs`, not `_*`.

The directory also sits outside `packages/`, which the workspace glob
`members = ["packages/*"]` would otherwise claim as a member and fail on for want of
a `pyproject.toml`. And `.scratch` is not a name `.gitignore` already swallows
(`lib/`, `build/`, `var/`, `share/`), so this code commits normally.

## Layout

| Directory | Implementation | Language |
|---|---|---|
| `dl/` | The `dl` version — the merge base, per PRD §3.5 | Python |
| `hmm-lush/` | The earlier `hmm` implementation, plus its Python translation | Lush → Python |
| `py-rudimentary/` | The rudimentary third implementation, plus the predecessor it was refactored from | Python |
| `align-poc/` | The proof-of-concept alignment library (`tokalign`) that PRD §1.2 describes and ADRs 0001–0004 derive from | Python + Cython |

**"`dl`" is a slot in this repository's package family, not the name of the source
project.** The merge base comes from **MelodyHPO**, a standalone and now defunct
project that built dataset handling and DL models together without packaging them;
the `dl` label originates in the `pfsmgraph` umbrella structure and was applied
retroactively. Anywhere the plan or the branch doc says "the `dl` implementation",
the artefact is `.scratch/dl/MelodyHPO/melody_hpo/data/`.

## Provenance

Filled in as each implementation is imported. Record where it came from and at what
revision, so a claim about "what the original did" stays checkable after this
directory is gone.

| Implementation | Source | Revision / date | Imported |
|---|---|---|---|
| `dl` | `github.com/PanosMavromatis/MelodyHPO`, branch `main` | `5f423118fa7cda0d7ca347ef0f112a326cca819c` — 2026-03-23, "Automate doc_paths construction in minicorps layer definitions" | 2026-08-31 |
| `hmm-lush` | A personal project of the repository owner's, not under version control | No VCS. Source mtimes span 2008-01-24 – 2011-02-01; the tree was last reorganised 2022-08-26 | 2026-08-31 |
| `py-rudimentary` (`segalign/`) | `github.com/PanosMavromatis/segalign`, branch `main` | `ca9780916fcf24e0c2293e6f6b3bc960f02239dc` — 2025-09-05, "Finished cleaning up existing Mode 8 tract corpus" | 2026-08-31 |
| `py-rudimentary` (`SegAlign-Draft/`) | `github.com/PanosMavromatis/SegAlign-Draft`, branch `main` | `9dc37b9afbd4f29f2222580f31cbedc761bf8070` — 2025-06-30, "Finished TISMIR submission" | 2026-08-31 |
| `align-poc` (`tokalign/`) | `github.com/PanosMavromatis/tokalign`, branch **`feat/docker-vertex-ai`** (not `main`) | `6d279368d2082acd22132a4fa674701fcd4d1ede` — 2026-05-12, "chore(dev): retire dev/agents in favor of workflow-claude plugin"; 71 commits | 2026-08-31 |
| ↳ nested `tokalign/tokalign-dev/` | `github.com/PanosMavromatis/tokalign-dev`, branch `main` | `5a783347179152458b8c531031b973925445c90f` — 2026-05-12; 20 commits | 2026-08-31 |
| ↳ nested `tokalign/dev/plugins/workflow-claude/` | `github.com/PanosMavromatis/workflow-claude`, branch `main` | `f0ee581b98ec8ccbe3c85b10d3e9c51af594369a` — 2026-05-12; 7 commits | 2026-08-31 |

The MelodyHPO working tree was copied whole, so it also carries a checkout of a
*second* repository at `MelodyHPO/data/MelodyData` —
`github.com/PanosMavromatis/MelodyData` at `abe762513eeeba0745425b76b3eb5f6409eadf65`
(2026-03-11), cloned with `--filter=blob:none`. That is ingested corpus data, not
MelodyHPO's own content, and is not tracked here; it is recorded because `data/` is
where `melody_hpo/data/` reads from, so a claim about the container's input format is
checkable against it while the copy is on disk.

The copy was `main` with one uncommitted file, `notebooks/explore/gpt2.ipynb`. That
is outside the imported scope, so nothing tracked here differs from `5f42311`.

## Two edits made to the imported tree

Both are renames, both reversible, and both were necessary rather than tidying.

- **`MelodyHPO/.git` → `MelodyHPO/.git-disabled`.** A directory containing `.git` is an
  *embedded repository*: git declines to descend into it and offers a gitlink instead.
  The failure mode is silent — `git add .scratch/dl/MelodyHPO/melody_hpo/data/data.py`
  staged nothing and exited `0`, and `git status` showed one collapsed
  `?? .scratch/dl/MelodyHPO/` line — so the branch would have merged with none of the
  merge base actually in it. The history is still readable with
  `git --git-dir=.git-disabled log`, which is how the revision above was captured.
- **`MelodyHPO/CLAUDE.md` → `MelodyHPO/CLAUDE.md.orig`.** Claude Code loads a nested
  `CLAUDE.md` when it reads files beside it, so a defunct project's agent instructions
  would have entered sessions working in this repository. Being untracked does not
  prevent that; presence on disk is what does it. The suffix keeps it readable while
  making it inert.

## What is tracked, and why so little

`.scratch/dl/.gitignore` is deny-by-default: the MelodyHPO tree is 2.2 GB on disk and
33 files (≈62 KB) of it are tracked. The exclusions carry their reasons inline in that
file rather than here. Two are worth knowing without opening it: `.env` stays ignored
by the repo-root rules, and `data/` (5.4 MB of ingested CSVs) is deliberately not
vendored, since it belongs to MelodyData rather than to MelodyHPO.

Both `.venv` trees, the caches, and `LLMsFS/` account for essentially all of the 2.2 GB.
None of it is needed to read the merge base; **do not walk the tree indiscriminately.**

## The `hmm-lush` import

**No renames were needed.** Both of the edits recorded above for MelodyHPO were
checked for before anything in this tree was read — it carries no `.git`, so there is
no embedded-repository trap, and no nested `CLAUDE.md` or `AGENTS.md`, so no foreign
agent instructions enter a session here. It does carry `CVS/` control directories
under `Code/_Old Lisp Code/`, which git does not treat specially the way it treats
`.git`, and which the deny-by-default rules exclude in any case.

`.scratch/hmm-lush/.gitignore` admits 135 files (≈129 KB) out of 929 MB. Nearly all of
what it turns away is `Training/`: saved checkpoints from training runs between 2008
and 2011, which are *outputs* of the algorithm being translated rather than inputs to
it. The exclusions carry their reasons inline in that file.

Two things about the tracked set are worth knowing without opening it.

- **`Code/SeqData/C/` is not tracked, and its absence is not an oversight.** Every `.c`
  there opens `WARNING: Automatically generated code ... by the DH compiler`: Lush's
  compiler emitting C from the `.lsh` beside it. It is a build artefact, and the `.o`
  files are built for two dead architectures (`powerpc-apple-darwin8`,
  `x86_64-unknown-linux-gnu`).
- **Two `.sds` directories are tracked whole**, `set11a_dInt.sds` and
  `set01z0_100.sds`. A `.sds` is a *directory*, not a file, and it is the only
  specification of the on-disk sequence format that exists — nothing in the tree
  documents it. Tracking one partially would be worse than not tracking it at all,
  since `_size` is a count the loader trusts: a directory missing some of its `.seq`
  files is not a smaller example but a corrupt one. That is why one specimen accounts
  for 107 of the 135 tracked files.

## The `py-rudimentary` import

**Two trees, one implementation.** `SegAlign-Draft` is the predecessor and
`segalign` is a refactoring of it into a `src/` layout with a `seq/` subpackage;
they are separate GitHub repositories, not two branches of one. Only `segalign`
contains anything that counts as a `dataseq` implementation, and it is the one
goal 4 compares against. `SegAlign-Draft` earns its place by *lacking* one: it has
no sequence abstraction at all, and both of its alignment entry points take bare
`List[Any]`, so sequences in it are unadorned lists of Python strings.

**Both renames were needed this time**, unlike the `hmm-lush` import where neither
was. Each tree carried a live `.git` (`SegAlign-Draft/.git`, `segalign/.git`), and
`segalign` additionally carried its own `CLAUDE.md`. All three are renamed as
recorded above — `.git-disabled` and `CLAUDE.md.orig`. **The revisions in the table
above were captured before disabling**, which is the only convenient moment: after
the rename every query needs an explicit `--git-dir`, as in
`git --git-dir=segalign/.git-disabled log`.

One thing the table cannot show: `segalign`'s working copy is **not** clean at
`ca97809`. `src/segalign/glob/needleman_wunsch.py` is modified and uncommitted, so
it is the one tracked file here that will not match GitHub. It is tracked at all
only because `src/segalign/__init__.py` imports `glob`, which imports it — without
it the merge target does not import and the tests do not run.

`.scratch/py-rudimentary/.gitignore` admits 72 files (≈356 KB) out of 1.7 GB, and
the exclusions carry their reasons inline in that file. Two things about the
tracked set are worth knowing without opening it.

- **The bar for tracking is deliberately higher here than for `hmm-lush`.** That
  import had no version control and was irreplaceable, so tracking it was
  preservation. These two are clean checkouts of live repositories at recorded
  revisions, so anything turned away costs one `git clone` to recover. The
  `tcoffee/` package — a T-Coffee multiple-alignment implementation, six modules —
  is the largest thing this reasoning turns away. It is real work, but it is
  `pfsmgraph-align`'s scope rather than `dataseq`'s. *(This originally added "and
  since `.scratch/` is deleted by the last goal of this branch, tracking it here
  would not survive to reach `align` anyway". That premise is void — the tree is
  retained. The exclusion stands on a better footing now: `align-poc/tokalign` is
  the actual ancestor of `pfsmgraph-align`, and supersedes this tree's alignment
  code as a migration source. When the `align` phase opens, widen `align-poc`'s
  policy rather than this one.)*
- **The 50 `All.csv` files are tracked so the tests are runnable**, not for their
  own sake. `Dataset.from_directories` walks that tree and builds its vocabulary
  from a chosen column, and `tests/seq/test_dataset.py` calls it against the real
  corpus rather than fixtures. Unlike the `.sds` directories, a partial copy would
  be *safe* here — nothing records an expected count, so a missing verse yields a
  smaller dataset rather than a corrupt one. All 50 are kept because they are
  200 KB.

## The `align-poc` import

**This is the library PRD §1.2 describes, and it was very nearly written off.** Before
it was found, four claims made about "the proof-of-concept" — an `Alphabet` type, a
`ScoringMatrix`, an `AlignmentResult`, and a Needleman-Wunsch `.pyx` — could not be
matched to anything in `.scratch/`, and the working hypothesis was that they were an
unverified recollection to be trimmed out of the PRD and ADRs 0001–0004. They were
not: all four are here, in `tokalign/src/tokalign/_types.py` and
`tokalign/src/tokalign/algorithms/needleman_wunsch/_cython.pyx`, and
`Alphabet.RESERVED_INDICES = 3` matches PRD §11's "padding/BOS/EOS low, gap
immediately after, user symbols from 4" exactly. **Nothing was trimmed**, and the
episode is recorded because the near-miss is the point: a documentation claim that
cannot be matched to code is not thereby false, and six documents were an hour away
from being rewritten to say a real library never existed.

`Alphabet` is also the only one of the four implementations that already agrees with
[ADR 0011](../docs/design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md)
on the substance: `encode` raises `KeyError` on an unseen symbol (strict by default),
`gap_index` is real, and `decode` exists. It needs renumbering — ADR 0011 inserts
`UNK` and `MSK`, moving user symbols from 4 to 6 — but that is a shift, not a repair,
and it is the reason the reserved block is worth getting right rather than merely
declaring.

**Eight renames were needed on import, the most of any so far.** Three nested
repositories (`.git` → `.git-disabled`) and five agent-instruction files. Two details
are worth knowing before re-running this import:

- **`AGENTS.md` and `AGENTS.override.md` were present at `tokalign/`'s root**, because
  that project uses the same agent-docs toolchain as pfsmgraph. They are not merely
  another project's instructions but the *generated artefacts* of one, and
  `protect-agent-docs.py` matches on filename, so under their original names they
  would have been treated as this repository's own.
- **`tokalign/dev/plugins/workflow-claude/` is mode 555**, a vendored clone of the
  same plugin this repository installs. Renaming a child needs write permission on the
  parent, so those two renames required `chmod u+w` first and the mode was restored
  afterwards. The failure presents as a bare `Permission denied` that reads like a
  sandbox restriction rather than a file mode.

The policy sits at `.scratch/align-poc/.gitignore` — one level *above* the imported
repository, as in `.scratch/dl/` — so `align-poc/` can hold our own analysis files
beside `tokalign/`. That placement is load-bearing rather than cosmetic: with the
policy one level down, its `!/*.md` rule ("our writing") also matched tokalign's own
`TODO.md` and `TODO.human.md`, silently tracking another project's planning docs as
though they were ours.

**The policy is phased**, which is new and follows from the retention change above.
Phase 1 (`dataseq`, active) admits only the `Alphabet` encoder and what makes its test
run; Phase 2 (`hmm`) is empty by design; Phase 3 (`align`) is written out but
commented, covering `scoring.py`, the algorithm/backend registry, the pure-Python
kernel and the `.pyx`. Advancing a phase is an uncomment, not a re-derivation — the
reasoning was done while the tree was in front of us.

## If this directory is ever deleted

Not planned any more, but the hazard does not disappear, so the reasoning is kept. A
squash merge collapses the commit that adds this code and the commit that deletes it
into nothing, losing it from `main` entirely. Retention would need a merge commit, a
tag on the pre-deletion SHA, or a branch left unmerged.
