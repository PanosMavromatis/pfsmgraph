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
- [ ] Implement the torch forward pass and autograd E-step
  - [ ] Forward pass in ADR 0020's scaled domain, matching numpy's description length
  - [ ] E-step counts from `.grad` on the log-parameters
- [ ] Assert the count identity as an ADR 0003 cross-backend test
  - [ ] numpy ξ/γ counts vs torch gradients over generated and constructed models
  - [ ] Measure and record the tolerance actually needed
- [ ] Document and sync
  - [ ] `docs/api/hmm/`, ADR notes, `core.md`, DEFERRED/master plan
