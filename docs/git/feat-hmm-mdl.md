# feat/hmm-mdl

**Created**: 2026-09-16
**Base**: main at ca98f41
**Status**: active

## Purpose

Write `_mdl.py`, the minimum-description-length criterion revision 04's topology search scores every candidate move with. Revision 03 already built the data half — `_quantize` and `_data_description_length` are in `_baum_welch.py`, and `entropy` has been in `_numeric.py` since 0.1.0 — so what is owed is the two code-length primitives from `Code/Utility/util.lsh`, the model half (`update-model-dl`), the choice of `d` (`suggest-d`), and the **total as a single named entry point** that nothing else recomputes. That last point is the branch's real deliverable: PRD §8 leaves open whether this project should end up with a refined one-part code, and keeping the criterion a seam is what makes that answerable by substitution rather than by rewriting the search.

## Scope

- ~~Decide what moves into `_mdl.py` and what stays in `_baum_welch.py`~~ — settled and executed 2026-09-16, see Notes
- ~~`int_code_length` and `comb_code_length`, overflow-safe~~ — landed 2026-09-16, see Notes
- ~~The model description length~~ — landed 2026-09-16, see Notes
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

**2026-09-16 — the primitives.** `_mdl.py` and `tests/test_mdl.py` (60 tests). Two things the reading settled that `HMMLIB-ACCOUNT.md` §8 does not state correctly: the original codes `n + 1` rather than `n`, and its iterated logarithm runs while the *value* exceeds 1, so it adds every term above zero — one more than §8's "while the term exceeds 1". The DH compiler's generated C settled both, which is the first time this branch has needed it.

The overflow concern in the scope line above was already answered by the original, which never forms the binomial. Measured, none of the three candidate forms was distinguishable on fidelity (≤ 4e-11 bits apart), so the choice was idiom. Cost was a non-issue too: the model half calls `_comb_code_length` twice per score however many states there are.

**2026-09-16 — the move executed.** `_quantize`, `_corpus_description_length` and `_data_description_length` are in `_mdl.py`; `_check_codes` is in `_params.py`; `_baum_welch` imports downward. Suite unchanged at 1478, which is the whole check a relocation needs. `test_baum_welch.py` goes 94 → 83 tests and `test_mdl.py` 60 → 71.

Two things the move turned up. The five helpers the moved section needed (`SEED`, `ROW_TOL`, `_vocabulary`, `_random_params`, `_random_corpus`) were **copied, not extracted**: three test modules already define their own `_vocabulary`/`_random_params` and two disagree about the signature, so sharing them would be a refactor of four files rather than a move of one section. And `test_baum_welch.py`'s docstring said "Six sections" over seven bullets — an off-by-one dating from revision 03 — so removing the moved section made the header true rather than breaking it.

**2026-09-16 — the model half.** `_model_description_length(params, d)`, reproducing all three logged `model-dl` values (58.2207 / 142.024 / 439.154) to within the log's six-figure print. Both open questions were settled by that oracle rather than by argument: the symbol axis is the **user** symbols, and `comb_code_length(d, 1 + n_states)` is reproduced although the combinatorics call for `m = n_states`.

The reserved-block exclusion is worth 12, 25 and 105 bits on the three models. That is not a rounding detail — 105 bits is 24% of `m008`'s model cost, and the whole `m001` run moves 58 → 142 bits across four accepted splits. Two tests pin each decision against the variant a future reader would most plausibly "fix" it to, since excluding the reserved block looks like an oversight and `1 + n_states` looks like an off-by-one. Mutation check: four mutants, all caught (7, 4, 2, 4 failures).
