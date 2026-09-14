# feat/hmm-forward-backward

**Status**: active
**Created**: 2026-09-14
**Subgoal**: Implement forward-backward in numpy as the reference — `docs/plan/TODO.md`,
revision 03-hmm-v0.2.0

## Goals

- [x] Settle the log base, 2 or *e*, before writing any code
  > **Q:** Should the numpy reference use (a) the scaled probability domain, as the original does, (b) log space in base *e* with a conversion at the DL boundary, or (c) log space in base 2?
  > **A:** Before deciding, the user asked for the implications for accuracy, speed, compatibility with backends and host architectures, and PyTorch. Measured, then answered: "Go with (a), fixed-order loop, and record it. The reasoning and evidence you presented in the last two turns deserve the next ADR."
  > **Q:** Within (a), should the reference sum in a fixed ascending order, which keeps phases 2-4 bit-exact with it, or use numpy's `@` with a tolerance in every cross-backend test?
  > **A:** Fixed-order loop.
  > **Done:** [ADR 0020](../../design/adr/0020-scaled-probability-domain-forward-backward.md) records the decision and the measurements. The kernel does `+ * /` over a host-built `(S, S, U)` table of arc probabilities. Summation is ascending over `i`, then `j`. The only logarithms are `bits(Q_t)` on the host, and no backend fuses multiply-add. `torch` is held to a stated tolerance of a few ulps.
  - [x] Read what `update-data-dl` (`hmm-trainer.lsh:346-402`) and `run-add` (`483-652`) accumulate, and in which base
    > **Note:** The original never works in log space. Both methods run a *scaled probability-domain* forward pass (Rabiner scaling): each α column is divided by its own sum `Q-t[i]`, `Q-t[0] = 1.0`, and `run-add`'s β is divided by the same `Q-t[i]`, starting at `1/Q-t[N]`. So `P-t = α*·a·o·β*` is ξ already normalised by the likelihood, and no division by P(seq) appears anywhere. The logarithm is taken only of the `N` scale factors, in base 2 via `safe-add--log2`: data DL = `Σ -log₂ Q-t[i]`, plus a final term for the last column's sum, which is 1 after scaling and so adds 0 bits unless the sequence is impossible. The master plan's "in log space" describes the draft's intent, not the original.
    > **Note:** The three copies of the forward recurrence disagree at a zero. `update-data-p` (`175`) and `update-data-dl` scale with `safe-/`, so an impossible sequence sets that column to 0 and the sentinel absorbs. `run-add` scales with a bare `/` (`547`, and in β at `558` and `576`), so the same sequence makes `0/0 = NaN`, which propagates into every re-estimated parameter. `HMMLIB-ACCOUNT.md` §13 calls `run-add` stage 1 "the same recurrence a third time" and does not record this difference.
  - [x] State what each option does to `bits`, to log-sum-exp inside the recurrence, and to the one-host numeric contract settled by `fix/hmm-viterbi-log2`
    > **Note:** An earlier statement in this goal's discussion was wrong, and is corrected here rather than left in the transcript. "Only `+ * /` in the kernel" is **not** enough for bit-exactness. Viterbi's reduction is `min`, which gives the same result in any order, but a sum does not: numpy's `@` and an ascending loop disagreed by 1 ulp in 48-77% of scale factors. torch cannot be bit-exact in any domain, differing from numpy in 62% of positions on CPU and 40% on CUDA. The contract therefore also has to fix summation order and forbid fused multiply-add.
    > **Note:** Accuracy does not separate the options: all three were within about 2 ulps of exact rational enumeration. What separates them is the gradient. On a state unreachable at every position, log-space `torch` gave `NaN` in all 9 transition gradients, and the scaled domain in none. That is the falsifier the revision preamble named, and `split-state` in revision 04 builds such states.
  - [x] Record the decision, and the semiring the master plan's phases 2-4 subgoal should name
    > **Note:** The semiring is sum-product with per-step renormalisation, neither max-plus nor log-sum-exp. It is recorded in ADR 0020 and pointed to from both master-plan subgoals, whose wording is left as written, since the draft's "max-plus/logsumexp" was kept on purpose.
- [ ] α and β in the scaled probability domain as a private, purely numeric kernel, following ADR 0020's evaluation order
  - [ ] It never raises: an impossible sequence comes back as an infinite cost, as in `_viterbi`
  - [ ] The arc-emission factor stays inside the inner loop (ADR 0015)
- [ ] ξ, γ and the sequence likelihood
  - [ ] γ sums to 1 over states at every position, ξ marginalises to γ, and α and β give the same total
- [ ] Tests against an oracle inside the repository
  - [ ] Brute-force enumeration over every state path on small models
  - [ ] Constructed cases: an impossible sequence, an exactly uniform model, zero-probability arcs
