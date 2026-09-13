# docs/hmm-api

**Status**: active
**Created**: 2026-09-13
**Subgoal**: Write the `/docs/api/` documents that pertain to this release —
`docs/plan/TODO.md`, revision 02-hmm-v0.1.0

## Goals

- [x] Settle what `docs/api/hmm/` contains before writing any of it
  > **Note:** The public `viterbi()` reaches only the phase-1 kernel: `__init__.py` imports
  > from `._viterbi`, and no public name reaches `_viterbi_cython`, `_viterbi_cpu_parallel`
  > or `_viterbi_cuda`. That is ADR 0003's open backend-selection question, routed to
  > `align`, not an oversight. But it means that in 0.1.0 the `gpu` and `cpu-parallel` extras
  > install modules no public call can use, which bounds what the docs may claim for them.
  > **Q:** How should `docs/api/hmm/` be split into pages?
  > **A:** README + params + viterbi, as `dataseq` does. `README.md` holds one worked
  > example, the numbered contracts (arc emission, min-sum over bits, `N + 1` states,
  > reserved fibres exactly zero, typed impossibility, smallest-index tie-break, frozen
  > value), the four-name surface table and the related ADRs. `params.md` owns `HMMParams`:
  > shapes, validation rules, the dead-arc exemption, zero-row rejection, cached derived
  > quantities, identity equality, and the reducible-chain error. `viterbi.md` owns
  > `viterbi`, `ViterbiPath` and `ImpossibleSequenceError`, the two `ValueError` cases,
  > deriving entropies yourself, and the seeding fix.
  > **Q:** In 0.1.0 the public `viterbi()` always runs the pure-Python kernel. How should the
  > docs present the other backends, the `gpu` / `cpu-parallel` extras and the `numpy<2.5`
  > cap?
  > **A:** State it plainly and briefly, in one short README section. `viterbi()` runs the
  > pure-Python reference kernel. The other three backends exist and are tested for
  > equivalence, but cannot be selected in 0.1.0. The extras change nothing a public call
  > does yet. The `numpy<2.5` cap binds only whoever installs `gpu`.
  > **Note:** Carried to the release subgoal, not settled here: whether `pfsmgraph-hmm`
  > 0.1.0 should ship the `gpu` and `cpu-parallel` extras at all, given the finding above.
  > The extras are consumer promises, and the published metadata of 0.1.0 is immutable.
  - [x] Decide the page split (by analogy with `dataseq/`'s `README.md` / `container.md` / `encoder.md`) and which contract each page owns
  - [x] Decide how the unselectable backends, the `gpu` / `cpu-parallel` extras and the `numpy<2.5` cap are presented to a consumer
- [x] Write the pages under ADR 0013's rule
  > **Done:** `docs/api/hmm/README.md`, `params.md` and `viterbi.md` were written, following
  > goal 1's split. They were drafted in the session scratchpad and run through
  > `test_api_docs._run` there before being placed, 5 + 28 + 24 = 57 examples with 0
  > mismatches. Only then were they copied into `docs/api/`.
  > **Q:** Write the drafts to `docs/api/hmm/` as they stand?
  > **A:** Yes, as drafted.
  > **Note:** Two statements on the pages come from the source and appear in no
  > docstring. First, `viterbi`'s code check is a *range* check only, so a record from
  > another vocabulary whose codes all fall in range decodes without error against the
  > wrong symbols. Second, recomputing `total_bits` by hand is compared within `1e-12`
  > rather than with `==`, because of the numpy/libm `log2` split found on
  > `feat/hmm-viterbi-cuda`. The claim that `viterbi` still decodes a reducible model,
  > whose `state_p` raises, was measured (`total_bits=3.0000`) rather than inferred.
  > Goal 3 reconciles both statements with the docstrings.
  - [x] Every code block executed, its output pasted from the run, tracebacks included
    > **Note:** Error outputs are pasted as the last line only, `SomeError: message`, the
    > convention `dataseq`'s pages and the verifier share. A full `Traceback` header
    > would fail that verifier's matching.
  - [x] `tests/test_api_docs.py` green over the new pages
    > **Note:** The test went from 5 to 8 cases: `hmm/README`, `hmm/params` and
    > `hmm/viterbi` were discovered by the existing glob with no change to the test. The
    > full suite went from 315 to 318 and passes. The check was also mutation-tested on scratch copies.
    > Changing a pasted return value (`viterbi.md:164`, `path.states`) and a pasted
    > error message (`params.md:109`) each produced exactly one reported mismatch, so the
    > new pages are genuinely under check rather than merely collected.
- [ ] Update the index and reconcile with the source
  - [ ] `docs/api/README.md`'s table lists `pfsmgraph-hmm` as documented
  - [ ] Each page checked against the docstrings of the names it documents; disagreements fixed on whichever side is wrong
