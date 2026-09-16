# feat/hmm-mdl

**Created**: 2026-09-16
**Base**: main at ca98f41
**Status**: active

## Purpose

Write `_mdl.py`, the minimum-description-length criterion revision 04's topology search scores every candidate move with. Revision 03 already built the data half — `_quantize` and `_data_description_length` are in `_baum_welch.py`, and `entropy` has been in `_numeric.py` since 0.1.0 — so what is owed is the two code-length primitives from `Code/Utility/util.lsh`, the model half (`update-model-dl`), the choice of `d` (`suggest-d`), and the **total as a single named entry point** that nothing else recomputes. That last point is the branch's real deliverable: PRD §8 leaves open whether this project should end up with a refined one-part code, and keeping the criterion a seam is what makes that answerable by substitution rather than by rewriting the search.

## Scope

- ~~Decide what moves into `_mdl.py` and what stays in `_baum_welch.py`~~ — settled 2026-09-16, see Notes
- `int_code_length` (Rissanen's universal prior) and `comb_code_length` (`log₂(sum+m) + log₂ C(sum+m−1, m−1)`), overflow-safe
- The model description length (`update-model-dl`, `hmm-trainer.lsh:402-427`), including what counts as a non-zero transition and how the reserved symbol block enters `n_symbols`
- The total, validated against the three tracked models' `_total_dl` at their stored `d`
- `suggest-d`, with those models' stored `d` as the oracle
- The `DEFERRED.md` trigger for promoting `_mdl.py` to a shared home, which the draft cited and which does not exist

## Context

- Master plan: revision 04-hmm-v0.3.0, first subgoal — `docs/plan/TODO.md`
- `.scratch/hmm-lush/HMMLIB-ACCOUNT.md` §8 is the reading of the MDL apparatus, and names `int-code-length` as Rissanen's universal prior; the master plan's settled block is authoritative where it and the 2026-09-03 draft disagree
- Settled and not to be relitigated: `_mdl.py` stays private; no `1e100` sentinel, because `bits(0)` is `+inf`; the criterion stays the original's two-part code
- ADR 0017 (`HMMParams` is a frozen value, derived quantities computed), PRD §8 (which description length scores the search)
- No DP kernel here, so the `dp-compile` gate does not arm and no ADR 0002 phases are owed

## Notes

**2026-09-16 — the seam.** `_mdl.py` owns `_quantize`, `_corpus_description_length` and `_data_description_length`; `_check_codes` moves to `_params.py`; `_baum_welch` imports the corpus length back for its convergence check, so every edge points downward from training to scoring. Nothing is exported.

The risk this scope line originally named was not the real one. `_data_description_length` has **no call site in `packages/`** — `baum_welch` watches the unrounded length, and the rounded one exists only for a scorer that does not exist yet — so the 94 tests were never exposed to the move, and the whole cost is one import block in `test_baum_welch.py`. What is genuinely shared is `_corpus_description_length` and `_check_codes`, which the plan had not named. Reasoning in `docs/plan/feat-hmm-mdl/TODO.md` under goal 1.
