# chore/hmm-0.3.0-audit

**Status**: active
**Created**: 2026-09-18
**Subgoal**: Review the whole `hmm` codebase and audit `docs/api/` against what 0.3.0 ships (revision 04)

## Goals

- [x] Read every `hmm` module as a whole, and record what the read finds
  > **Q:** Goal 1 says "record what the read finds" and no goal covers acting on it. Should the fixes land on this branch?
  > **A:** Record and fix the docstrings now, in this goal — the six stale docstrings, the two off-by-one `util.lsh` citations, and an ADR 0025 reference in the four modules it governs. Leave `_suggest_split`/`_suggest_merge` as a note; removing or wiring them is a code change, not an audit.
  > **Done:** All 15 Python modules and both `.pyx` kernels read. Nine docstring/comment edits across seven modules: the ADR 0025 contradiction in `_topology.py`, ADR 0025 citations added to the four modules it governs, two off-by-one `util.lsh` line numbers, five stale tenses, and one Markdown link converted to reST. Nothing executable changed; the package imports and `__all__` is still ten names.
  - [x] Private helpers left without a consumer — confirm revision 04 removed `safe_divide` from that list
    > **Note:** Closed. `safe_divide` has 13 live call sites — `_forward_backward` (6), `_baum_welch` (3), `_topology` (3), `_mdl` (1) — and its docstring's "no call site in 0.1.0 reaches this yet" was corrected here.
    > **Note:** The same shape reopened twice, and is left alone deliberately. `_trials._suggest_split` and `_trials._suggest_merge` have no consumer in `src/`; only tests call them. `_suggest_move` bypasses both and calls `_split_moves`/`_merge_moves` directly, and it must: ranking merges and splits in *one* stable sort is what makes a merge win a tie, so it cannot compose two separate rankings. Both functions are faithful ports of Lush methods that exist (`suggest-split`, `suggest-merge`), so deleting them would lose the correspondence the port rests on. Resolving this is a code change, not an audit — see the Q&A above.
    > **Note:** `_forward_backward._state_posteriors` also has no `src/` consumer, but 13 test call sites including the `hmmlearn` oracle, so it is checked by something. Not the same shape.
  - [x] Docstrings still describing a subgoal's intermediate state, such as `_topology.py`'s "state merge next"
    > **Note:** The example this subgoal names is already gone — it was fixed when the merge landed on `feat/hmm-merge-states`. Six others were found and all six fixed here.
    > **Note:** One was substantive rather than stale: `_topology.py`'s module docstring read "The public surface for topology search arrives with the search loop", which is not a past intermediate state but a *promise about 0.3.0* that ADR 0025 overturned four days later. A reader of the module learned the opposite of the decision, and no module cited 0025 to correct them.
    > **Note:** The other five were tense: `_numeric.py` (`safe_divide` has no consumer), `_viterbi.py` ("will decode" → does), `_baum_welch.py` ("belongs with the model description length in revision 04" → it does, in `_mdl`), `_params.py` ("revision 04 has to decide what an unreachable state means" → it did; `_merge_states` refuses a two-transient pair), and both CUDA modules' "a placeholder until the suite is green" — a condition since met, with the tuning still never done.
  - [x] Every ADR, `HMMLIB-ACCOUNT.md` and Lush line reference points at what it names; no module presents new work as a port
    > **Note:** Verified against the tree, not read. All ADR paths cited resolve; 11 distinct ADRs are cited and **none is superseded** (0008 and 0012 appear nowhere in `src/`). All six `HMMLIB-ACCOUNT.md` sections cited (3, 4, 5, 7, 8, 9) exist. Every Lush range is within its file, and every anchor spot-checked lands on what it names: `hmm-param.lsh:142`→`split-state`, `:172`→the `state-p` seeding defect, `:218`→`merge-states`, `:262`→its twin, `hmm-trainer.lsh:405`→`n-states-r`, `:462`→`training-log-line`, `:677`→`keep-model`, `:749`→`try-split`, `:952`→`suggest-move`, `util.lsh:119`→`minimize`, `:467`→the `n+1`, `:523`→`rand-p-vector`.
    > **Note:** Two were off by one and are fixed: `_mdl.py` cited `util.lsh:124` for `CGOLD` (actual 125) and `util.lsh:125` for `ZEPS` (actual 126). A citation one line off is the kind that rots without ever failing.
    > **Note:** The significant finding was an *absence*: **no module cited ADR 0025**, including the four it governs. `_search`, `_trials`, `_topology` and `_mdl` now each cite it in their "Private to" paragraph, each naming the cost 0025 accepts for that module.
    > **Note:** Markup drift, fixed: `_search.py` was the only module using a Markdown link (`[ADR 0023](../../../../../docs/...)`) in a reST docstring, where it renders as literal text under `help()`. Six modules use the `.. [ADR NNNN]` footnote form; the four revision-04 modules carry no footnote block at all and cite inline instead. That divergence is left as it is — it is a convention question for the whole package, not a defect.
    > **Note:** "No module presents new work as a port" passes outright. `_topology.py` states "**The split is not a transliteration of the original**"; `_search.py` names its four departures and closes "A reviewer checking this against the original should expect exactly those differences and no others."
  - [x] The draw-order and `Generator` contracts end to end, from the public search call down to `_split_state`
    > **Note:** Intact. `_search` keys the start `spawn_key + (0,)` and round `r` `+ (1, r)`; `_trials._split_moves` appends `(s, t)`, giving the `(1, r, s, t)` its own docstring claims; `_try_split` forwards `rng=`; `_split_state` requires it as a keyword with no default and no module state. All draws happen before any array is built, so a dead arc never shifts a later draw.
    > **Note:** The draw count was verified by running it, not by reading it — a counting wrapper over `Generator.uniform` gave 13, 20, 27 draws at `S` = 1, 2, 3 with `n_user = 3`, matching the docstring's `S + 2*(S+1)*(n_symbols - USER_BASE)` exactly.
    > **Note:** This subgoal's own wording says "the public search call". There is none — ADR 0025 keeps the search private, so the chain starts at `_search`. Worth catching here rather than in the release notes.
- [ ] Verify the counts and the backend header against the tree rather than against the prose
  - [ ] `uv run pytest` — the suite count, and the ADR 0003 header unchanged from 0.2.0
  - [ ] Module count and ADR count in `core.md`
- [ ] Audit `docs/api/` against the ten names 0.3.0 ships
  - [ ] Every `__all__` name against its page: parameters, return fields, raised errors
  - [ ] `docs/api/hmm/README.md` — index, contracts and example, against a package that now searches topology as well as trains
  - [ ] `docs/api/README.md`
  - [ ] `packages/pfsmgraph-hmm/README.md`, which becomes the PyPI long description under an immutable version
  - [ ] Every code block executed with its output pasted; `tests/test_api_docs.py` green
