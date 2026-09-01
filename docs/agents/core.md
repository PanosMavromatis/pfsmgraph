# pfsmgraph

Shared project knowledge for any coding agent working in this repository.

## Current state

**The workspace is scaffolded and `uv sync`s cleanly; no algorithms are implemented.** In place: the uv workspace root `pyproject.toml` (virtual — no `[project]` table), `uv.lock`, all five `packages/*` members with their own `pyproject.toml`, the (currently dormant) `meson.build` files for `align` and `hmm`, and one empty `pfsmgraph/<pkg>/__init__.py` per member (plus `dl/rnn/` and `dl/transformer/`). Not yet present: any implementation code, any tests. The proof-of-concept alignment/HMM code has not been moved in. The initial ADR set (twelve records) is authored in `docs/design/adr/` and is authoritative for the decisions it covers; the PRD remains the narrative design document.

**`align` and `hmm` are temporarily on hatchling, not meson-python.** meson-python's editable-install import hook injects a `sys.meta_path` finder that claims the entire `pfsmgraph` PEP 420 namespace and shadows the other distributions, so `import pfsmgraph.dataseq` (and `hseg`, `dl`) fails after `uv sync`. Neither package has compiled code yet — the `meson.build` extension blocks are dormant `if fs.exists()` guards — so the switch to meson-python is deferred to when the first `.pyx` lands, at which point the namespace/editable interaction must be solved (non-editable install of the compiled members, a single combined compiled distribution, or an upstream fix). The `meson.build` files and the revert recipe are kept in `packages/pfsmgraph-{align,hmm}/pyproject.toml`. This qualifies the PRD §6.1 note, whose "namespace is fine" evidence was gathered with *setuptools* editable, which composes; meson-python's finder does not.

**`.scratch/` holds imported source that is not ours, together with our own writing about
it.** It is where the three existing `dataseq` implementations are read side by side before
anything is merged into `packages/pfsmgraph-dataseq/`, together with the proof-of-concept
alignment library whose `Alphabet` is the *encoder* ancestor. That distinction is why ADR
0010 still says "three implementations" while also requiring the `Alphabet` reconciliation:
four imported sources, three of them containers. So "no implementation code, no tests" above is a
claim about `pfsmgraph`: `.scratch/` does contain Python, Cython and `tests/` directories
belonging to other projects, and now also Python of our own — a runnable transliteration of
the Lush original under `.scratch/hmm-lush/translation/`, written as a reading aid for the
merge.

**It is retained across branches, and that changed on 2026-08-31.** It was created as a
temporary `dataseq` working area to be deleted by the last goal of the `feat/dataseq-merge`
plan; it is not, because the same imports are the migration source for `hmm` and `align`
0.1.0. What is re-scoped per package is not the contents but the **`.gitignore` policies**:
each import's rules surface the files relevant to the package being migrated, so the tracked
set follows the work. `.scratch/align-poc/.gitignore` is the first written in explicit
phases (`dataseq` active, `hmm` empty by design, `align` written but commented), and
advancing it is an uncomment rather than a re-derivation.

**Four imported directories, six source trees**: `.scratch/py-rudimentary/` holds two
repositories — `segalign/` (the implementation) and `SegAlign-Draft/` (the predecessor it was
refactored from, tracked at one file because what it contributes is the *absence* of a
sequence abstraction) — and `.scratch/align-poc/tokalign/` carries two more nested inside it.
`.scratch/align-poc/` is the proof-of-concept alignment library that PRD §1.2 describes and
that ADRs 0001–0004 derive from, so unlike the other three it is a *source* of this project's
invariants rather than only evidence about them. Nothing there is part of any distribution
and nothing outside it may import from it. The leading dot is
load-bearing — it matches pytest's default `norecursedirs` entry `.*`, which is why `uv run
pytest` still collects zero items with those files present; and the directory sits outside
`packages/`, so the workspace glob never claims it. `.scratch/README.md` states the rest,
including why an imported repository's `.git` must be renamed before its contents can be
committed here.

Still to do, in PRD order (§11): implement `dataseq` (merge of three existing implementations, `dl` version as base — §3.5); then `hmm` (Lush translation); then `align`, then `hseg`. The `dataseq` merge also promotes ADR 0010 from `Proposed` to `Accepted` by settling the encoder API.

## Commands

Toolchain: **uv** (workspace) + **pytest**. Requires `uv` and Python ≥ 3.10.

- `uv sync` — create/refresh the venv; installs all five members editable (plain `.pth`) plus the `dev` group (`pytest`).
- `uv run pytest` — run the suite (no tests exist yet).
- `uv build --package pfsmgraph-<pkg>` — build one member's sdist + wheel.
- `uv lock` — refresh `uv.lock` (committed; one lockfile for the whole family).

When `align`/`hmm` move back to meson-python: re-add `meson-python`, `cython`, and `ninja` to the root `dev` group (`ninja` must be on `PATH` for rebuild-on-import), and the compiled members will additionally need a C compiler.

## Architecture

`pfsmgraph` is a family of five independently publishable Python packages sharing one PEP 420 namespace, developed in a single repo as a **uv workspace** under `packages/`.

```
                  dataseq          (base — no intra-family dependencies)
                     │
                   align
                  ╱  │  ╲
              hseg  hmm  dl
```

| Distribution | Import | Role |
|---|---|---|
| `pfsmgraph-dataseq` | `pfsmgraph.dataseq` | Data sequence container + symbol↔code encoder; base layer, PyTorch `Dataset`-compatible |
| `pfsmgraph-align` | `pfsmgraph.align` | Sequence alignment (DP-heavy, compiled) |
| `pfsmgraph-hseg` | `pfsmgraph.hseg` | Hierarchical segmentation |
| `pfsmgraph-hmm` | `pfsmgraph.hmm` | Baum-Welch, topology search via state merge/split; translated from an existing Lush implementation |
| `pfsmgraph-dl` | `pfsmgraph.dl` | PyTorch models; `rnn` and `transformer` are plain submodules of one distribution |

The family coheres because `align`/`hseg` are **interpretability instruments for the outputs of `hmm`/`dl`**, and because alignment is a training accelerant for HMM topology search — not merely because the topics are adjacent.

**Implementation order** (`dataseq` → `hmm` → `align` → `hseg`) deliberately differs from **release order**, which must follow the dependency graph since a package cannot publish before its dependencies exist on PyPI.

## Invariants

These constrain any code written here. They are inherited from the proof-of-concept and are non-negotiable without amending the PRD.

- **No `pfsmgraph/__init__.py` anywhere.** The `pfsmgraph` level is a PEP 420 implicit namespace; each distribution contributes exactly one regular subpackage beneath it. An `__init__.py` at that level breaks every other package's imports.
- **Encode at the boundary.** Multi-character string symbols are mapped to integers at the entry point of every public call; all inner computation is integer-only; results decode back to strings at exit. This is what makes Cython and CUDA backends mechanical to write — they never touch string types.
- **Fixed reserved symbol block** in `dataseq`, not configurable: `PAD`=0, `UNK`=1, `BOS`=2, `EOS`=3, `GAP`=4, `MSK`=5; user symbols from 6. `PAD` must be 0 because PyTorch's zero-fill idioms (`pad_sequence`, `torch.zeros()` buffers) would otherwise silently mean something other than "absent". Encoding is **strict by default** — unseen symbols raise; `UNK` fallback is explicit opt-in.
- **Three-phase algorithm lifecycle**, applied in order wherever dynamic programming appears: pure Python (correctness) → Cython (performance) → Numba CUDA anti-diagonal wavefront (scale).
- **One parameterized test suite per algorithm**, run automatically against every available backend, so backend equivalence is enforced rather than assumed. Absent hardware (no CUDA device) skips, but *loudly* — the session header names every backend excluded and why; a backend that is implemented but not importable (missing or stale Cython build) is a hard failure, never a skip; a lifecycle phase not yet reached contributes no parameter at all. `PFSMGRAPH_REQUIRE_BACKENDS` escalates skips to failures for CI. See ADR 0003.
- **Build backends are per-package, not family-wide.** meson-python for compiled members (`align`, the Baum-Welch core of `hmm`); hatchling for pure-Python members. meson-python editable installs need `ninja` present for rebuild-on-import. *(Currently `align`/`hmm` are on hatchling too, pending their first `.pyx` — see "Current state".)*
- **"GPU" means two unrelated things.** `numba-cuda` for the DP packages, `torch` for `dl`. Do not unify these into one `[gpu]` extra.
- **The ADRs outrank the imported implementations.** The `dataseq` merge takes the `dl`
  (MelodyHPO) version as its *base* because it is the most mature of the three, but base
  means starting point, not authority. Where any of the three disagrees with an Accepted
  ADR, the ADR wins and the implementation is changed, unless that ADR says otherwise in
  its own text. The imports are evidence about what has been tried; they are not a source
  of decisions that have already been made. This bites hardest on the reserved block:
  four sources use three different offsets for user symbols — `dl` at 3, the Lush original
  at 2, the rudimentary `segalign` at 2 (with `PAD` at **1**), the proof-of-concept
  `tokalign` at 4. The two that collide are both containers, and they mean different things
  by the same integers: `0` is `begin` in Lush and `:EOS` in `segalign`. **None of the three
  containers has a `GAP` code**; `tokalign` does, at index 3, which is why its user symbols
  start at 4 — the one source with a gap code being the one written to align sequences.
  *(Corrected 2026-08-31 by the goal-4 measurement in
  `.scratch/RESERVED-BLOCK.md` §2, which is authoritative for this table. The earlier wording
  counted the proof-of-concept as one of the three containers and denied it a `GAP` code;
  both were written from recollection before `segalign` and `tokalign` had been read.)*
  [ADR 0011](../design/adr/0011-fixed-reserved-symbol-block-and-strict-encoding.md)
  settles this: `PAD`=0 … `MSK`=5, user symbols from 6, and the renumbering lands **as part
  of** the merge rather than after it. So a merge note reading "the base must be overridden
  here" is about the points the ADRs leave open, never about reopening the ones they close.

## Workspace footgun

During development a `{ workspace = true }` path source satisfies *any* version constraint, so a missing or wrong bound in `[project.dependencies]` never fails locally — it only breaks a pip user after publish. Keep published lower bounds honest and review them on every breaking change.

The `0.0.0` placeholder releases already on PyPI are intentionally dependency-free; do not add dependency declarations to them.

A live instance of the footgun, worth recognising: all five members declare `0.1.0.dev0`, and `0.1.0.dev0` does **not** satisfy `>=0.1` under PEP 440 — a `.devN` release sorts strictly before the final, and is excluded even with `prereleases=True`. So `align`'s declared `pfsmgraph-dataseq>=0.1` is satisfiable by nothing that exists today: PyPI has only `0.0.0`, and local is `0.1.0.dev0`. It never fails because the workspace source satisfies any constraint. This resolves itself when a real `0.1.0` publishes.

## Versioning

**Versions are per-package, and there is deliberately no `VERSION` file at the repo root.** Release order is forced by the dependency graph — `dataseq` must publish before `align` can — so the five members can never share a version, and a repo-wide version number would be a claim about nothing. Each member owns the `version` field in its own `pyproject.toml`; all five currently read `0.1.0.dev0`.

Release tags are per-package too: `pfsmgraph-<pkg>-v<version>`, e.g. `pfsmgraph-dataseq-v0.1.0`. Hyphen rather than slash, because git refs are paths and a `pfsmgraph-dataseq/v0.1.0` tag cannot coexist with a plain `pfsmgraph-dataseq` one. No tags exist yet; the first is created by the release commit (`docs/plan/DEFERRED.md`, trigger "the first real release").

The `.dev0` suffix stays until that release commit. `uv build` stamps whatever `pyproject.toml` declares onto the wheel, so a bare `0.1.0` on an incomplete package means one accidental publish burns `0.1.0` on PyPI permanently — versions are immutable, and yanking or deleting a release does not free the number. A burnt `0.1.0.dev0` costs nothing by comparison, and pip will not install a pre-release by default.

## Design docs

- `docs/design/PRD.md` — packaging, naming, and distribution architecture; the source for the initial ADR set (§9).
- `docs/plan/DEFERRED.md` — decided-but-not-yet-actionable work, indexed by the trigger that unblocks it (the `dataseq` merge, the first `.pyx`, CI existing, the first real release). Check it when starting any of those; several items must land *as part of* their trigger rather than after it.
- `docs/design/adr/` — the twelve initial ADRs, authoritative for the decisions they cover; [`adr/README.md`](docs/design/adr/README.md) indexes them. Add new records with the next unused number and a row in that index; numbers are never reused.
