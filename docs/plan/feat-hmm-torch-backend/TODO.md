# feat/hmm-torch-backend

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Add the `torch` backend behind the `[torch]` extra (revision 03-hmm-v0.2.0)

## Goals

- [ ] Decide the public surface: which call takes `backend="torch"`, and the `[torch]` extra
  - [ ] Settle whether forward-backward/EM gains a public call here or torch reaches `_TABLE` privately
  - [ ] Decide the `torch` row's `needs` value and extra name, and whether ADR 0021 needs an amendment
- [ ] Implement the torch forward pass and autograd E-step
  - [ ] Forward pass in ADR 0020's scaled domain, matching numpy's description length
  - [ ] E-step counts from `.grad` on the log-parameters
- [ ] Assert the count identity as an ADR 0003 cross-backend test
  - [ ] numpy ξ/γ counts vs torch gradients over generated and constructed models
  - [ ] Measure and record the tolerance actually needed
- [ ] Document and sync
  - [ ] `docs/api/hmm/`, ADR notes, `core.md`, DEFERRED/master plan
