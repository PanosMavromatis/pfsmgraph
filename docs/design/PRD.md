# PRD: Package Naming & Distribution Architecture

**Status:** Draft — naming, distribution, and build-backend decisions settled; dependency graph partially open
**Date:** 2026-06-29 (updated 2026-08-21: `pfsmgraph-dataseq` added; PyPI names secured)
**Intended location:** `docs/design/PRD.md` (repo root, alongside `docs/design/adr/`)
**ADRs:** none yet — this document is the source from which the initial ADR set is authored (§9)

> A composable ecosystem of Python packages for modeling symbolic data sequences. Probabilistic finite-state models (PFSMs) are the unifying core, bridging sequence alignment, hierarchical segmentation, HMMs (Baum-Welch), deep learning (RNNs, Transformers), interpretability, and graph operations.

This document captures the design of `pfsmgraph` as a family of related, independently publishable packages: how the namespace is structured, how the repository and build system are organized, and how the packages depend on one another. It is a refactoring of existing proof-of-concept work (§1.2) into a layered ecosystem, not a greenfield design. It records what was decided, the evidence behind each decision, and the threads still open at the time of writing.

---

## 1. Context

### 1.1 What this ecosystem is

The ecosystem's durable conceptual core is **sequence modeling using probabilistic hierarchical finite-state techniques**, where "hierarchical" refers to modeling at multiple levels of abstraction (tokens at each level being, by analogy, "letters", "words", "phrases", etc.; the primary modeling domain is not language syntax).

Rather than ship as a single monolithic package, the project adopts a **prefix system** that groups the related packages under one namespace. Each package is independently installable and versioned, so a consumer takes only the layers they need.

The alignment library is intentionally **not** positioned to compete with general-purpose alignment tools. Its value is being purpose-built as a component of this ecosystem. This reframes what would otherwise be a discoverability cost (an opaque umbrella name) into an asset: the name correctly signals "component of a larger system" rather than "standalone tool."

### 1.2 The inherited foundation

This is not a greenfield design. It builds on an existing proof-of-concept sequence alignment library, which established the patterns the whole family now depends on:

- **A shared type foundation** — `Alphabet` (bidirectional string ↔ integer mapping over arbitrary multi-character symbols), `ScoringMatrix` (a 2-D array indexed by integer symbol IDs, with affine gap penalties), and `AlignmentResult`.
- **Encode at the boundary** — because multi-character string symbols cannot serve as array indices, `Alphabet` encodes them to integers at the entry point of every `align()` call; all inner computation is integer-only; results decode back to strings at exit. This is what makes compiled and GPU backends mechanical to write, since they never touch string types.
- **A three-phase algorithm lifecycle** — pure Python for correctness, Cython for performance, Numba CUDA (anti-diagonal wavefront) for scale, applied strictly in sequence wherever dynamic programming is involved.
- **A single parameterized test suite** — one test file per algorithm, automatically run against every available backend, so backend equivalence is enforced rather than assumed.

These decisions stand. They are the reason the restructuring below is tractable: the encode-at-the-boundary discipline and relative-import hygiene mean the code moves cleanly into a multi-package layout.

### 1.3 Origins

The components were not conceived together. They were written over two decades, in different languages, for different immediate purposes, and only recently recognized as one system.

| When | What |
|---|---|
| 2004 | First attempts at analyzing the *output* of black-box models (HMMs at the time), rather than inspecting their parameters. |
| Late 2000s | HMM component written in **Lush** (the Lisp dialect used in Yann LeCun's early research): Baum-Welch training, state merging and splitting. Included its own data sequence class. |
| ~2010–2024 | Hiatus. |
| 2025 | Sequence alignment component outlined, with hierarchical segmentation projected as its only follow-up. |
| Early 2026 | Deep-learning component begun, exploring interpretable neural models of symbolic sequences — **with its own, third data sequence class**. |
| June 2026 | The components recognized as one ecosystem; namespace and packaging strategy settled. |
| August 2026 | Common base extracted: `dataseq` added as a fifth package beneath the other four. |

Two observations from this history shape the present design.

**The duplication was not hypothetical.** Three independent data sequence representations had already been written — one in the Lush HMM code, one in the alignment component's type foundation, one in the deep-learning component — before anyone decided they should be shared. The `dataseq` extraction is a response to something that had already happened twice, not a precaution against something that might.

**The unifying insight predates the packaging.** The 2004 output-analysis work and the Lush HMM component were pursuing the same question the deep-learning component now pursues with neural models. The through-line was there long before the packages were named.

### 1.4 Why these components belong together

The ecosystem is held together by two substantive dependencies, not merely by shared subject matter:

**Alignment as an interpretability instrument.** Sequence alignment and hierarchical segmentation are tools for analyzing the *output* of black-box sequence models — complementary to activation-based and weight-based interpretability, which inspect a model's internals. This is the thread running from the 2004 experiments through to the current deep-learning work: comparing and segmenting what a model *emits* is a way to characterize what it has learned. It makes `align` and `hseg` instruments for interrogating `dl` and `hmm`, not merely neighbors of them.

**Alignment as a training accelerant.** Alignment offers a way to jump-start HMM training that is more streamlined than raw search through topology space via state merging and splitting. Aligned sequences suggest structure that would otherwise have to be discovered by search, making `align` a direct input to `hmm`'s training procedure rather than a parallel capability.

These two dependencies are why the family is a family. The packaging decisions in the rest of this document are the mechanism for expressing them; the layer boundaries follow from them.

### 1.5 Implementation order

**Implementation begins with `dataseq`.** This aligns implementation order with the dependency graph at the base, then diverges from it above.

The three existing data sequence implementations have been inspected and will be merged into one (§3.5). Building the foundational layer first, rather than deferring it, removes the risk that a fourth representation hardens during the Lush translation, and it gives `hmm` a defined interface to target rather than one to invent.

Expected order thereafter:

- **`dataseq` first** — a merge of three known implementations rather than a design to be invented, which makes it far more tractable than "foundational layer" usually implies.
- **`hmm` next**, despite sitting high in the dependency graph. A complete, working implementation already exists in Lush; translation to Python/Cython/Numba is well-scoped and agent-assistable, and the original design is expected to survive largely intact. The translation is expected to be straightforward provided `dataseq` exposes the interface `hmm` needs — which is not radically different from what the `dl` implementation already provides.
- **`align`** has an outlined design and a proof-of-concept type foundation.
- **`hseg` last.** Its algorithms are the least specified — the intent is that segmentation optimizes sequence alignments across a corpus, but the specific algorithms are not yet chosen. It has the largest design gap of the five.

The dependency graph (§3.4) still governs *release* order, which is a separate question: a package cannot publish before its dependencies exist on PyPI. Since `hmm` is expected to be implemented ahead of `align`, it may be finished well before it can be released.

---

## 2. Summary of decisions

| #   | Decision                                                                                                                       | Status                            |
| --- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| D1  | Adopt the `pfsmgraph` prefix                                                                                                   | Decided — names claimed (§4)      |
| D2  | PEP 420 namespace packages: `pfsmgraph-<pkg>` distributes, `import pfsmgraph.<pkg>`                                            | Decided (§3)                      |
| D3  | Package roster: `dataseq`, `align`, `hseg`, `hmm`, `dl`                                                                        | Decided (§1, §3)                  |
| D4  | Reserve the bare `pfsmgraph` name on PyPI as a placeholder                                                                     | Done — all six names secured (§4) |
| D5  | Single repository as a **uv workspace** rather than N repos                                                                    | Decided (§5)                      |
| D6  | `dl` is a **single distribution** with `rnn`/`transformer` as submodules — no third namespace tier                             | Decided (§7)                      |
| D7  | Each package declares its **own build backend**; the family is build-heterogeneous                                             | Principle (§6)                    |
| D8  | Build backend: **meson-python** for compiled members (`align`, `hmm` core); hatchling for pure-Python members                  | Decided (§6.1)                    |
| D9  | `dataseq` is the **base layer** — a common dependency of all four other packages, with no intra-family dependencies of its own | Decided (§3.4)                    |
| D10 | **Implementation begins with `dataseq`**, merging the three existing implementations with the `dl` version as the base         | Decided (§1.5, §3.5)              |
| D11 | **Fixed reserved symbol block** in `dataseq`: `PAD`=0, `UNK`=1, `BOS`=2, `EOS`=3, `GAP`=4, `MSK`=5; user symbols from 6        | Decided (§3.6)                    |

---

## 3. Naming and namespace strategy

### 3.1 Prefix

The prefix is `pfsmgraph` (probabilistic finite-state machine + graph). A finite-state machine *is* a graph, so `graph` is conceptually redundant — but it is load-bearing in practice, because the bare `pfsm` name is already taken on PyPI by an unrelated, thematically adjacent project (see §4). The suffix is therefore the disambiguator that makes the prefix available at all.

Accepted tradeoff: the prefix foregrounds the *substrate* (finite-state graphs) rather than the *differentiator* (the hierarchy). The "hierarchical" axis is carried by individual package names (e.g. `hseg`) rather than the family name. This was considered and accepted; alternatives that encode "hierarchical" in the prefix (`phfsm`, `hpfsm`) were rejected as unpronounceable.

### 3.2 Namespace package pattern

Packages use the implicit-namespace-package pattern (PEP 420), the same approach used by `azure-*`, `google-cloud-*`, and the legacy `zope.*` ecosystems:

```
uv add pfsmgraph-align          # distribution name
import pfsmgraph.align as align # import name
```

This gives family identity, independent versioning and release per package, and install-what-you-need granularity. The `as align` import alias neutralizes the verbosity of the prefix at call sites.

**Mechanical requirement:** no `pfsmgraph/__init__.py` may exist in any package — the `pfsmgraph` level is a namespace, not a regular package. Each distribution contributes exactly one regular subpackage beneath it (`pfsmgraph/align/`, `pfsmgraph/hseg/`, …).

### 3.3 Package roster

| Distribution | Import | Role |
|---|---|---|
| `pfsmgraph-dataseq` | `pfsmgraph.dataseq` | Data sequence library; base layer, used by all four others. Designed for PyTorch `Dataset`/`DataLoader` interoperability |
| `pfsmgraph-align` | `pfsmgraph.align` | Sequence alignment; depends on `dataseq` |
| `pfsmgraph-hseg` | `pfsmgraph.hseg` | Hierarchical segmentation; depends on `dataseq`, `align` |
| `pfsmgraph-hmm` | `pfsmgraph.hmm` | HMM topology search and Baum-Welch, incl. state merging/splitting; translated from an existing Lush implementation; depends on `dataseq`, `align` |
| `pfsmgraph-dl` | `pfsmgraph.dl` | Deep-learning components (PyTorch); `rnn` and `transformer` submodules; depends on `dataseq`, `align` |

### 3.4 Dependency graph

```
                  dataseq          (base — no intra-family dependencies)
                     │
                   align
                  ╱  │  ╲
              hseg  hmm  dl
```

`dataseq` sits beneath `align`, which was previously assumed to be the family's common base. Keeping `dataseq` free of intra-family dependencies is what prevents cycles and makes the release order unambiguous (§11).

`align` remains a common dependency of `hseg`, `hmm`, and `dl`. Whether those three have dependencies on *each other* is still open (§8).

### 3.5 `dataseq` composition — merging three implementations

`dataseq` is not designed from scratch. Three data sequence implementations already exist (§1.3), and they are being merged into one:

| Source | Maturity | Contribution |
|---|---|---|
| `dl` | Most mature; PyTorch-compatible by design | **The merge base.** Container semantics and the encoder/decoder (tokens ↔ numeric codes) |
| `hmm` (Lush) | Working but more primitive encoding | Design elements from a proven implementation; defines the interface the translation will need |
| `align` | Primitive proof-of-concept — a list of strings | No new design constraints |

**Why the `dl` implementation is the base.** It is the most mature of the three and was built for PyTorch interoperability from the outset. Notably, it required **no `DataLoader` subclassing** — stock PyTorch `DataLoader` was usable directly, which means the interoperability requirement is satisfied by conforming to the existing `Dataset` protocol rather than by extending PyTorch's machinery.

**Why one implementation serves all consumers.** The token ↔ integer-code mapping has different *purposes* across the family — indexing transition and emission matrices in Baum-Welch, versus looking up rows in a vector embedding table — but the *structure* is the same in both cases: a bidirectional mapping between symbols and contiguous integer codes. This is the encode-at-the-boundary principle (§1.2) generalized beyond alignment: the integer code is the shared currency, and each consumer interprets it according to its own needs. The differing requirements do not justify differing representations.

**Batching is not a complication.** The current `hmm` implementation does not use batch training, so questions about reconciling batched and unbatched access do not arise at this stage.

**Consequence for the Lush translation.** Building `dataseq` first means the `hmm` translation targets a defined interface rather than inventing one, and the interface it needs is not expected to differ substantially from what the `dl` implementation already provides. This dissolves the earlier concern that translating `hmm` first would harden a fourth sequence representation.

### 3.6 Reserved symbol block

The reserved index allocation is **fixed in `dataseq`** and is not configurable. Every package may assume this layout without negotiation; that is the point of placing it at the base layer.

| Index | Code | Purpose |
|---|---|---|
| 0 | `PAD` | Padding / absence |
| 1 | `UNK` | Unknown symbol (see strictness below) |
| 2 | `BOS` | Beginning of sequence |
| 3 | `EOS` | End of sequence |
| 4 | `GAP` | Alignment gap |
| 5 | `MSK` | Masking |

User symbols are numbered from **6**. All six codes are three characters, which keeps aligned display of multi-character symbols uniform.

**Why `PAD` is 0.** PyTorch's zero-fill idiom makes this close to mandatory: `pad_sequence` defaults to `padding_value=0`, batch buffers are naturally allocated with `torch.zeros()`, and collate functions pad with zeros unless told otherwise. If `PAD` were any other index, every collate path would need to pass the value explicitly, and any zero-initialized tensor would silently mean something other than "nothing here" — a quiet and hard-to-trace class of bug in the foundational layer. (`nn.Embedding(padding_idx=…)` accepts any index, so it does not constrain the choice; the zero-fill convention does.) That `hmm` does not currently batch is not a reason to relax this, since `dataseq` serves `dl` as well.

**`UNK` exists but encoding is strict by default.** An unknown symbol appearing in *training* data indicates an upstream error, not a condition to absorb silently. The encoder therefore raises on unseen symbols by default, with graceful fallback available as an explicit opt-in (e.g. `on_unknown="raise"` as the default, `"unk"` where resilient inference is genuinely wanted). This enforces corpus curation mechanically rather than by convention, while keeping the slot available for inference against real-world data. Omitting `UNK` entirely was rejected: adding it later would renumber every user symbol.

**`MSK` is reserved before it is needed.** Masked-objective training and masking-based interpretability probes (occlusion, feature ablation) are plausible in the `dl` component, and interpretability is a core motivation for the ecosystem (§1.4). Reserving the slot now costs one index; retrofitting it later would shift every user symbol and invalidate every persisted encoding and trained embedding table.

**Cost of reserved slots is negligible.** The alphabet grows by six, and a scoring matrix is (n+6)² rather than n² — immaterial for the curated alphabets this ecosystem targets, and a small price for never having to renumber.

---

## 4. PyPI names — secured

All six names — the five packages plus the bare `pfsmgraph` umbrella — have been **claimed** via module-free `0.0.0` placeholder releases (hatchling, `bypass-selection = true`). Ownership persists; real releases publish over the placeholders at a higher version.

| Name | Status |
|---|---|
| `pfsmgraph` (bare namespace placeholder) | Claimed |
| `pfsmgraph-dataseq` | Claimed |
| `pfsmgraph-align` | Claimed |
| `pfsmgraph-hseg` | Claimed |
| `pfsmgraph-hmm` | Claimed |
| `pfsmgraph-dl` | Claimed |
| `pfsm` (bare-prefix alternative, not used) | **Taken** — "Python Fast Strings Matching" (v0.1.3) |

The `pfsm` collision is in the same problem neighborhood (string matching sits next to alignment), so a bare-`pfsm` prefix would be both a namespace and a thematic clash — the worst kind. This is the decisive reason to keep `graph`.

Notes from the claiming process, worth retaining:

- **A placeholder upload is what holds a name.** A trusted-publishing "pending publisher" does *not* reserve anything — it is invalidated if someone else registers the name first. Trusted publishing is a publish-*security* mechanism, not a name-holding one. The correct order is: claim by placeholder upload, then attach a normal trusted publisher to the now-existing project when CI is ready.
- **PyPI rate-limits new project creation**, and the production limit is tighter than the documented defaults (which are 20/hour/user, 40/hour/IP). Creating four projects in one session triggered `429 Too many new projects created` on the fifth. Retrying the following day succeeded. Plan bulk claiming across more than one day.
- Name normalization means claiming `pfsmgraph-align` also locks `pfsmgraph_align` and `pfsmgraph.align`.
- TestPyPI is a separate instance — a name claimed there reserves nothing on real PyPI.
- Placeholders should be replaced with real releases within a reasonable window; PEP 541 treats content-free projects as somewhat more reclaimable. Keep the account email reachable.

---

## 5. Repository and distribution: uv workspace

### 5.1 Decision

Adopt a **single repository structured as a uv workspace**, with each package an independently publishable workspace member, rather than one repository per package.

### 5.2 Why not separate repos

A prior two-repo split in this project (the library versus its Claude Code development plugin) was driven by **audience and access-control** needs: a public library versus internal tooling, with different permissions. That rationale does not transfer here. The `dataseq`/`align`/`hseg`/`hmm`/`dl` packages serve a **single audience** (the project author, building one ecosystem) and share a **tight internal dependency graph**. That is the textbook case *for* a workspace: independently publishable packages, path dependencies during development, and atomic cross-package commits when an API change in `align` must be propagated to its consumers in the same breath. With N repos, every early API change to `align` becomes a release-and-re-pin dance across repositories.

Splitting a package into its own repo later remains available if and when its lifecycle genuinely diverges.

### 5.3 Pip compatibility (the decisive resolution)

A uv workspace is a **development-time construct only**. It lives in the `[tool.uv.workspace]` and `[tool.uv.sources]` tables, which build backends ignore. Published wheels are built from the standard `[project]` table, so a consumer's metadata carries only an ordinary dependency such as `pfsmgraph-align>=0.1`. A `pip install pfsmgraph-hseg` reads that metadata and pulls `pfsmgraph-align` from PyPI exactly as normal. **Pip users never know a workspace existed.** The workspace changes how the author resolves and builds locally, not what consumers receive.

### 5.4 Layout

```
pfsmgraph/                          # one repo
├── pyproject.toml                  # [tool.uv.workspace] members = ["packages/*"]
├── uv.lock                         # single lockfile for the whole family
├── docs/
│   └── design/
│       ├── PRD.md                  # this document
│       └── adr/
└── packages/
    ├── pfsmgraph-dataseq/
    │   ├── pyproject.toml          # base layer — no intra-family deps
    │   └── src/pfsmgraph/dataseq/
    ├── pfsmgraph-align/
    │   ├── pyproject.toml          # depends on pfsmgraph-dataseq
    │   └── src/pfsmgraph/align/    # NO __init__.py at pfsmgraph/ (namespace)
    ├── pfsmgraph-hseg/
    │   ├── pyproject.toml          # depends on pfsmgraph-dataseq, -align
    │   └── src/pfsmgraph/hseg/
    ├── pfsmgraph-hmm/
    │   └── src/pfsmgraph/hmm/
    └── pfsmgraph-dl/
        └── src/pfsmgraph/dl/
            ├── rnn/                # regular submodules — one distribution
            └── transformer/
```

### 5.5 Dependency declaration pattern

Each consumer carries both the real published dependency and the dev-time redirect. For `align` (which depends only on the base layer):

```toml
[project]
dependencies = ["pfsmgraph-dataseq>=0.1"]     # what pip users receive

[tool.uv.sources]
pfsmgraph-dataseq = { workspace = true }       # what the author resolves locally
```

For `hseg`, `hmm`, and `dl`, which sit above both:

```toml
[project]
dependencies = [
  "pfsmgraph-dataseq>=0.1",
  "pfsmgraph-align>=0.1",
]

[tool.uv.sources]
pfsmgraph-dataseq = { workspace = true }
pfsmgraph-align = { workspace = true }
```

`dataseq` declares no intra-family dependencies.

**Do not add these declarations to the published `0.0.0` placeholders.** The stubs on PyPI are dependency-free, which is correct — a stub declaring `pfsmgraph-dataseq>=0.1` would fail to resolve, since no such version exists yet. The declarations live in the local workspace and first reach PyPI with the initial real release.

### 5.6 Disciplines and the one footgun

- The workspace shares a single lockfile, so all members resolve against one consistent set of transitive pins. This is a feature for a tightly coupled family; it becomes a constraint only if two members ever need conflicting transitive versions.
- **Footgun:** during development the path source satisfies *any* version constraint, so a missing or wrong bound in `[project.dependencies]` will not surface locally — it only bites a pip user after publish. Keep the published version bounds honest and reviewed.
- The workspace does **not** address the Cython-extension editable-install friction; that remains per-package build config (see §6.1).

---

## 6. Build heterogeneity across the family

The Python → Cython → anti-diagonal-wavefront parallelization lifecycle applies **only where dynamic programming is involved**. This makes the family heterogeneous in build needs. A key benefit of the workspace: each member declares its own build backend in its own `pyproject.toml` — there is no need to pick one backend for everyone.

| Package | DP / wavefront | Compiled extensions | Build backend | GPU mechanism |
|---|---|---|---|---|
| `dataseq` | None expected | None expected | hatchling (`dl`-derived base is presumed pure-Python; confirm at merge — §8) | Stock PyTorch `DataLoader`; no subclassing |
| `align` | Pervasive | Cython + Numba CUDA | **meson-python** (decided — §6.1) | `numba-cuda`, optional `[gpu]` |
| `hmm` | Baum-Welch core only | Cython (partial; topology search likely plain Python) | **meson-python** (matches `align`) | `numba-cuda`, optional `[gpu]` |
| `hseg` | Unknown — TBD | Possibly none | hatchling if pure-Python; meson-python if DP is found | n/a |
| `dl` | None | None | hatchling (pure-Python) | PyTorch (`torch`) |

Two consequences:

- The build-backend decision applies to the *compiled* members (`align`, and the Baum-Welch slice of `hmm`), **not the family**. Pure-Python members use a lightweight backend (hatchling).
- "GPU support" means two unrelated things across the family: `numba-cuda` for the DP packages versus `torch` for `dl`. Do **not** unify these into a single `[gpu]` extra or dependency story.

### 6.1 Build-backend decision (D8 — decided)

The compiled members use **meson-python**; pure-Python members use **hatchling**. The existing proof-of-concept uses `setuptools.build_meta` (`setuptools>=75`, `cython>=3.0`, `numpy>=2.1` at build time, `language_level="3"`, directives as function-scoped `@cython` decorators in the `.pyx`). The switch to meson-python happens when that code is brought into the workspace.

Rationale and evidence:

- Now is the cheapest possible switch point: the entire compiled surface is one example Needleman-Wunsch extension with zero shipped algorithms, so there is almost nothing to port.
- meson-python's editable installs auto-rebuild compiled code on import. Under setuptools editable, a `.pyx` edit does **not** propagate until a manual `build_ext --inplace` — verified empirically: a stale `.so` persisted across a fresh process until rebuilt. The auto-rebuild advantage recurs for every algorithm cythonized (NW, SW, Hirschberg, banded, the `hmm` Baum-Welch core) and for the wavefront passes.
- meson-python is where the scientific-Python compiled ecosystem has standardized (NumPy, SciPy, scikit-*).
- A namespace concern was considered and **withdrawn**: setuptools was empirically shown to editable-install a Cython extension and a pure-Python sibling across two namespace workspace members with no shadowing (no `pfsmgraph/__init__.py`; `[tool.setuptools.packages.find]` with `where=["src"], namespaces=true`). setuptools was therefore workable; meson-python is chosen for the dev-loop and ecosystem reasons above, not out of necessity.

Operational note: meson-python editable installs need `ninja` available in the dev environment for the rebuild-on-import hook.

---

## 7. The `dl` package: single distribution, no third tier

`rnn` and `transformer` will co-exist within the same model (e.g. an RNN at the shortest time-scale, transformers higher in the hierarchy), so their usage is tightly coupled. They will be shipped as **one distribution** (`pfsmgraph-dl`) with `pfsmgraph.dl.rnn` and `pfsmgraph.dl.transformer` as plain submodules.

This keeps `dl` a regular package and `pfsmgraph` the **only** namespace level. Splitting `rnn`/`transformer` into separate distributions was rejected: it was not well-motivated given their coupling, and it would have introduced a *multi-level* namespace (`pfsmgraph.dl.*` spanning distributions), compounding the namespace + src-layout + Cython editable-install friction for no benefit. Separate distributions would only be justified by genuinely divergent heavy dependencies or independent release cadence, which is not anticipated.

---

## 8. Open questions

- **Remaining dependency graph.** `dataseq` as the base and `align` as the common mid-layer are now established (§3.4). Still open: whether `hseg`, `hmm`, and `dl` depend on *each other*, and how tightly the family co-evolves in the first year. This affects how much the workspace's atomic-commit benefit is worth.
- **Placement of the inherited alignment types.** `dataseq`'s container and encoder/decoder scope is settled (§3.5), as is the reserved index layout (§3.6 — `GAP` is a `dataseq` concern, not an `align` convention). `Alphabet` overlaps directly with the encoder/decoder being merged into `dataseq` — the two express the same symbol ↔ integer-code mapping — so reconciling them is part of the merge rather than a separate question. `ScoringMatrix` and `AlignmentResult` are alignment-specific and stay in `align`. What remains is the exact shape of the reconciled encoder API: constructor signature, the strictness switch, and how `align` consumes the mapping at its boundary.
- **`dataseq` build backend.** Whether `dataseq` is pure-Python (hatchling) or carries performance-critical inner loops warranting compilation (meson-python) follows from the merge. The `dl`-derived base is presumed pure-Python; confirm once merged.
- **`hseg` DP.** Whether hierarchical segmentation has its own DP recurrence (and therefore its own Cython/wavefront path) is unconfirmed. If it is pure orchestration over `align`, it stays a pure-Python package.

---

## 9. ADRs seeded by this document

This PRD is the source for the repository's initial ADR set. Numbering starts fresh; no ADRs are carried over from earlier iterations of the project. Each decision below should be promoted to a standalone ADR at scaffolding time, and the ADR — not this document — becomes the authoritative record thereafter.

| Decision | Proposed ADR topic | Source |
|---|---|---|
| D1–D2 | Namespace prefix and PEP 420 package layout | §3.1, §3.2 |
| D5 | Single repository as a uv workspace | §5 |
| D6 | `dl` as a single distribution; no third namespace tier | §7 |
| D7–D8 | Build-backend heterogeneity; meson-python for compiled members | §6, §6.1 |
| D9 | `dataseq` as the dependency-graph base layer | §3.4 |
| D10 | `dataseq` composition — merging three implementations, `dl` as base | §3.5 |
| D11 | Fixed reserved symbol block and strict-by-default encoding | §3.6 |

The inherited decisions introduced in §1.2 — the three-phase algorithm lifecycle, encode-at-the-boundary, the single parameterized test suite, and the earlier GPU-backend and dependency-strategy choices — also need ADRs in the new numbering. Their content stands unchanged; only package names and cross-references need updating. They are constraints on the codebase regardless of whether they are written down, so leaving them unrecorded would give the new repository no account of why its most pervasive patterns exist.

Prerequisite: the remaining open questions in §8 are narrower than they were — the `dataseq` merge (§3.5) settles the base layer's scope. The encoder API reconciliation should be resolved during the merge itself, before the ADR covering D9–D10 is finalized.

---

## 10. Out of scope for this document

- How the Claude Code development plugin fits the multi-package family (one family-wide dev plugin vs. per-package) was not discussed.
- The full mechanics of bringing the proof-of-concept code into the workspace; §11 captures only the items that must not be lost (backend swap, extension paths, the deferred `.pyx` fix).

---

## 11. Scaffolding notes

- The existing codebase is a minimal proof-of-concept being brought into this structure; the module moves are mechanical given the project's relative-import discipline.
- PyPI names are secured (§4). Placeholders should be replaced by real releases within a reasonable window.
- Scaffold the workspace with `dataseq` first (the base layer), then `align` on **meson-python** (§6.1), then `hseg`, `hmm`, `dl`. For `align`, `setup.py` is replaced by a `meson.build`; the extension target is `pfsmgraph.align.algorithms.needleman_wunsch._cython`, with the source path package-relative under `packages/pfsmgraph-align/`.
- **Release order follows the dependency graph:** `dataseq` → `align` → {`hseg`, `hmm`, `dl`}. A package cannot be published while a declared dependency version does not yet exist on PyPI. Note that this is *release* order, not *implementation* order — `hmm` is expected to mature first (§1.5).
- **Version bounds need deliberate attention.** In the workspace, a `workspace = true` source satisfies any constraint, so a wrong or missing bound in `[project.dependencies]` never surfaces locally — it only breaks a pip user after publish. Set real lower bounds when `dataseq` and `align` get their first actual releases, and revisit on every breaking change.
- Author the initial ADR set from this document (§9). Start numbering at 0001; do not import earlier ADR files or their numbering.
- **Implementation starts with `dataseq`** (§1.5, §3.5): merge the three existing implementations, taking the `dl` version as the base, before beginning the Lush `hmm` translation.
- **The reserved block renumbers existing code.** The proof-of-concept alignment types use a different allocation (padding/BOS/EOS low, gap immediately after, user symbols from 4). Adopting D11 (§3.6) shifts every user symbol and changes the gap index. Nothing is persisted yet, so this is a code-only change — but it must land as part of the merge, not after, and any hard-coded index assumptions in the proof-of-concept alignment code need auditing.

### 11.1 Deferred code fix — apply while moving the file, not before

Fold this into the file move so it does not become a separate edit-and-commit errand:

- **`_cython.pyx` (Needleman-Wunsch) — 2-D memoryview indexing.** Change every double-bracket access on the typed 2-D memoryviews `M`, `X`, `Y`, `T` from `A[i][j]` to `A[i, j]` (e.g. `M[i - 1][j - 1]` → `M[i - 1, j - 1]`, `T[i][j]` → `T[i, j]`, `M[m][n]` → `M[m, n]`). The `A[i][j]` form materializes an intermediate 1-D view per access; `A[i, j]` compiles to direct strided pointer arithmetic under the existing `@cython.boundscheck(False)` / `@cython.wraparound(False)`. This is a pure performance/idiom fix with **no behavioral change**, so it is safe to apply blind and will not affect test equivalence. Leave the already-correct comma-form access (`scores[seq_a[i - 1], seq_b[j - 1]]`) and the 1-D memoryviews (`seq_a[...]`, `aligned_a_buf[k]`) untouched. This file is the reference template for future kernels and the wavefront passes, so fix it before it is copied.
