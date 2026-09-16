# feat/hmm-mdl

**Status**: active
**Created**: 2026-09-16
**Subgoal**: Write `_mdl.py` — `docs/plan/TODO.md`, revision 04-hmm-v0.3.0

## Goals

- [ ] Settle the seam before writing any of it: what `_mdl.py` owns, and what moves into it
  - [ ] Decide whether `_quantize` and `_data_description_length` **move** out of `_baum_welch.py`, stay where they are and are re-exported, or are called across the module boundary. The seam argues for moving; `tests/test_baum_welch.py`'s 94 tests argue for care, and the data half is the one piece with a measured oracle already passing
  - [ ] Fix the module's public shape: everything private to `pfsmgraph.hmm`, nothing added to `__init__.py`'s ten exported names in this branch, and the total as the one function a caller outside `_mdl.py` needs
- [ ] Port the two code-length primitives from `Code/Utility/util.lsh`
  - [ ] `int_code_length` — Rissanen's universal prior for the integers: `log₂ 2.865` plus the iterated logarithm while the term exceeds 1. Decide the domain (it is called on state counts and on `d`, so zero and one are the cases to pin) and check the terminating condition against the original rather than against the paper
  - [ ] `comb_code_length(sum, m)` = `log₂(sum + m) + log₂ C(sum + m − 1, m − 1)`. Use a log-gamma or a log-binomial that does not overflow; `d` reaches the hundreds and `m` is `1 + n_states`, so the binomial itself is not representable
  - [ ] Tests: exact values against hand-computed cases, monotonicity, and agreement with a `fractions`/`mpmath` reference where one is cheap
- [ ] The model description length (`update-model-dl`, `hmm-trainer.lsh:402-427`)
  - [ ] `int_code_length(n_states_r) + int_code_length(d) + (1 + n_states_r) · comb_code_length(d, 1 + n_states)`, plus `n_non_zero_transitions × comb_code_length(d, 1 + n_symbols)`. Establish what `n_states_r` is against `n_states`, and what counts as a non-zero transition — §8 says **after** quantization, which is what makes `d` structural rather than cosmetic
  - [ ] Decide how the reserved symbol block enters `n_symbols`. `HMMParams`'s symbol axis spans the whole vocabulary and requires the six reserved fibres to be exactly zero, so a literal `output_p.shape[-1]` charges the model for six symbols it can never emit
- [ ] The total, as the single named entry point the search calls
  - [ ] Data half + model half, with **no `1e100` sentinel** — `bits(0)` is `+inf`, which absorbs under addition and sorts last already (settled in the master plan)
  - [ ] Keep it substitutable: the PRD §8 question (two-part code vs. a refined one-part code) must later be answerable by replacing this function, not by rewriting the search
  - [ ] Validate the total against the three tracked models' `_total_dl` at their stored `d`. The data half already matches its logged `data-dl` to 0.009 bits, so a failure here localises to the model half
- [ ] Choose `d` (`suggest-d`)
  - [ ] Port it, and establish what it optimises. `d` is simultaneously the rounding grid and an argument of the code, and rounding at `1/d` is what drives small transitions to exactly zero — so a sparse topology is cheaper to describe than a dense one *because* of this choice
  - [ ] Oracle: the three tracked models' stored `d`. Record how close the port comes and what explains any gap
- [ ] Write the missing `DEFERRED.md` trigger for promoting `_mdl.py` to a shared home
  - [ ] The draft cited `## Trigger: hseg needing description lengths`; no such heading exists and the file has no occurrence of "description length" at all. Write it, or restate the deferral under `## Trigger: hseg design settling`, and say which and why
- [ ] Docs and the ADR question
  - [ ] Decide whether the criterion warrants an ADR. It is a model decision rather than a packaging one, which is the 0015/0017/0020 shape; the argument against is that the criterion is the original's and this branch ports rather than chooses it
  - [ ] `docs/api/hmm/` gains nothing if nothing is exported — confirm that, rather than assuming it, and note it here either way
