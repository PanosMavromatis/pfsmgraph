# API documentation

One subdirectory per distribution. Three of the five members have no code yet and so
have no subdirectory; each gets one when it gets an implementation.

| Distribution | Import | Docs | Covers |
|---|---|---|---|
| `pfsmgraph-dataseq` | `pfsmgraph.dataseq` | [dataseq/](dataseq/) | the ragged record container, `pad_collate`, the symbol encoder |
| `pfsmgraph-align` | `pfsmgraph.align` | — | no code yet |
| `pfsmgraph-hseg` | `pfsmgraph.hseg` | — | no code yet |
| `pfsmgraph-hmm` | `pfsmgraph.hmm` | [hmm/](hmm/) | model parameters, the Viterbi decode and its batched form, Baum-Welch training (`baum_welch`), and runtime backend selection (`backend=`, `backends`) |
| `pfsmgraph-dl` | `pfsmgraph.dl` | — | no code yet |

These pages are hand-written Markdown. The layout, that choice, and the rules below are
[ADR 0013](../design/adr/0013-api-documentation-layout-and-tooling.md).

**Two things worth knowing before you read further.**

`docs/` is not shipped in any wheel. It is not in one under this layout and would not be
under a per-package one either, which is part of why the docs live at the repository root
rather than inside `packages/*/`. `pip install pfsmgraph-dataseq` gets you the code and its
docstrings; it does not get you these pages.

**Every code block here has been executed and its output pasted from the run**, never
transcribed from memory. Where a page shows a traceback or an error message, that is the
real text the library produced. This is the only thing standing between hand-written prose
and drift, so it is a rule rather than an aspiration, and `tests/test_api_docs.py`
enforces it by running every page on each test run.

**Some results depend on the machine, and those pages say so.** Which backends of
`pfsmgraph.hmm` can run differs between a laptop and a GPU server, so the backend examples
are chosen to print the same output everywhere and leave availability to `backends()`. The
compiled backends promise identical bits on one machine, not across machines.

## What is normative

The docstrings and these pages divide the work:

- **Docstrings are normative for signatures.** Names, parameters, defaults, and types are
  read from the source, which is where your editor and `help()` look too.
- **These pages are normative for contracts** — the invariants a caller may rely on, the
  reasons behind them, and the boundaries between packages. A contract stated here and
  nowhere else is still a contract.

Where a page and a docstring disagree, one of them is a bug. Both are in this repository;
fix whichever is wrong.
