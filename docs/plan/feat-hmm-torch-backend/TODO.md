# feat/hmm-torch-backend

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Add the `torch` backend behind the `[torch]` extra (revision 03-hmm-v0.2.0)

## Goals

- [x] Decide the public surface: which call takes `backend="torch"`, and the `[torch]` extra
  > **Q:** Which call should `backend="torch"` attach to: a public trainer now, a private `e_step` row with no extra yet, or a public E-step only?
  > **A:** Public trainer now: export the EM loop as a public call taking `backend=`, which selects the E-step, with a public result type; the `[torch]` extra ships.
  > **Q:** What is the public trainer called (also its `backends()` and `_TABLE` key)?
  > **A:** `baum_welch`, naming the algorithm as `viterbi` does and leaving `fit`/`train` free for revision 04's topology search.
  > **Q:** What are the torch row's `needs` value and extra?
  > **A:** `needs="torch"`, extra `torch`: a new `Needs` literal with its own `_reason` branch, a reported skip under the test policy (`ESCALATED_NEEDS` unchanged), separate from `gpu`.
  > **Q:** How does ADR 0021 absorb torch?
  > **A:** A dated in-place amendment: `"torch"` joins §1's names and §3's table, stating that `BackendStatus` carries no tolerance and a torch result may differ by the recorded one.
  > **Q:** Is torch declared in the root `dev` group?
  > **A:** Yes, for the reason numba is: the repository exercises the backend on its own promise.
  - [x] Settle whether forward-backward/EM gains a public call here or torch reaches `_TABLE` privately
    > **Note:** torch cannot be a row under `forward_backward`: every row of a table must share one kernel signature, `_forward_backward` returns `(alpha, beta, scale)`, and the autograd route builds no β. The shared operation is the per-record E-step, `(init_state_p, transition_p, output_p, codes) -> (counts, bits)`, so the table gains a key for it, whose `python` row composes `_forward_backward`, `_description_length` and `_expected_counts`.
  - [x] Decide the `torch` row's `needs` value and extra name, and whether ADR 0021 needs an amendment
    > **Note:** torch already imports in this workspace (2.13.0+cu130, CUDA available) without being declared anywhere in the root: it arrives as `pfsmgraph-dl`'s hard dependency. The dev-group declaration replaces that accident with a promise.
  > **Done:** `baum_welch(params, records, *, backend="python")` becomes public, keyed `baum_welch` in `_TABLE` with `python` and `torch` rows over a per-record E-step signature; `[torch]` extra and dev-group entry land with goal 2, the ADR 0021 amendment with goal 4.
- [x] Implement the torch forward pass and autograd E-step
  > **Q:** Which tensors receive the gradients: the probabilities (counts `p * p.grad`) or log-parameters (counts `θ.grad`)?
  > **A:** The probabilities. `p · ∂log P/∂p` is `∂log P/∂log p` by the chain rule, so the identity is the plan's, and the forward pass starts from numpy's exact inputs rather than `exp(log p)`, which is not bit-exactly `p`; dead arcs need no `log(0)` leaf.
  > **Q:** Which dtype and device does `backend="torch"` use?
  > **A:** float64 on CPU only. ADR 0021 lets no environment choose where a call runs, and device placement is the batching subgoal's (ADR 0021 Open); float32 stays ADR 0020's Open item.
  > **Q:** What floor do the `[torch]` extra and the dev-group entry declare?
  > **A:** `torch>=2.13`, the version the backend is measured on, as `cpu-parallel`'s floor is numba's measured version.
  > **Q:** Which stopping-rule keywords does `baum_welch` make public?
  > **A:** All four, `batch_cycles`, `change_bits`, `patience`, `max_cycles`, defaulting to `run-converge`'s constants.
  > **Q:** Does the private loop become public by renaming or by a wrapper?
  > **A:** Renamed in place: `_em` → `baum_welch`, `_EMResult` → `BaumWelchResult`, both exported, and the EM tests switch to the public names.
  - [x] Forward pass in ADR 0020's scaled domain, matching numpy's description length
    > **Note:** measured in a scratchpad script against `_forward_backward._e_step`: 240 random models (S ≤ 32, N ≤ 400, 30% dead arcs) agree in description length to at most 1.0 eps relative, and the three Lush fixtures over the renumbered corpus (N = 1268) to at most 3.1 eps (m008: `-1.1e-12` bits of 1630.6). The empty record gives 0 bits and the constructed impossible record `+inf` on both. **The first fixture run was a false pass**: fed raw Lush codes rather than `load_corpus_record()`'s `+ USER_BASE` ones, every fixture was impossible, both backends returned `inf` with all-zero counts, and "0.0 difference" meant nothing. Goal 3's test must assert the fixture's description length is finite.
  - [x] E-step counts from `.grad` on the log-parameters
    > **Note:** as `p * p.grad` on probability leaves (goal 2's first Q&A), which is the log-parameter gradient by the chain rule. No `nan` anywhere, and the zero pattern of all three arrays matches numpy's exactly, dead arcs and reserved fibres included. The largest absolute difference is `3.3e-15` (init), `4.0e-13` (transition), `2.3e-13` (emission) at N ≤ 400, and `1.2e-12` on the fixtures, whose counts reach 1268. **Relative error is the wrong measure**: on counts near underflow it reached `5e57` for init counts, and `4e-7` even above a `1e-12` floor, while the absolute difference stayed at rounding. Scaled by `N·eps` the worst was 0.5 (init), 28.7 (transition, S = 2, N = 2000) and 6.3 (emission), so the error grows with count size as well as `N`, and goal 3 should set an absolute or mixed tolerance, not a relative one.
  - [x] Wire `baum_welch`: the `_TABLE` key and `torch` row, `_em`'s E-step through `_resolve`, the public call and result type, the `[torch]` extra and dev-group entry
    > **Note:** the lock diff was pure insertion (the dev entry, the `torch` extra, `provides-extras`), so no resolution moved. The header now ends `| baum_welch python ✓ · torch ✓`. Deriving error text from the table has a reach worth knowing: adding a row changed the pasted output of `viterbi(..., backend="numba")` in `docs/api/hmm/backends.md`, a call that never touches the new row, and ADR 0013's executed-docs test caught it along with `__all__` in the README.
  > **Done:** `_baum_welch_torch._e_step` (probability leaves, counts `p * p.grad`, float64 CPU) and `_forward_backward._e_step` are `baum_welch`'s two rows; `_em` is now the exported `baum_welch(params, records, *, backend="python", batch_cycles, change_bits, patience, max_cycles)` returning `BaumWelchResult`, resolving `backend` before any work. `[torch]` extra and dev-group `torch>=2.13` declared; suite green at 710 (the new torch-reason test plus the header/table pins updated). Pasted outputs in `docs/api/hmm/` synced; the call's own documentation is goal 4.
- [ ] Assert the count identity as an ADR 0003 cross-backend test
  - [ ] numpy ξ/γ counts vs torch gradients over generated and constructed models
  - [ ] Measure and record the tolerance actually needed
- [ ] Document and sync
  - [ ] `docs/api/hmm/`, ADR notes, `core.md`, DEFERRED/master plan
