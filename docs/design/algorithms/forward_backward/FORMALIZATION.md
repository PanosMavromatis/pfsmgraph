# Forward-backward and the Baum-Welch E-step (arc-emission HMM)

## Metadata

| Field | Value |
|-------|-------|
| Source | **Recovered from the phase-1 implementation.** `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_forward_backward.py` at commit `e9593db`, which was ported from three methods in `.scratch/hmm-lush/Code/HMMlib/hmm-trainer.lsh`: `update-data-p` (`:126-184`), its clone `update-data-dl` (`:346-399`), and stages 1-4 of `run-add` (`:483-617`). The written account of those methods is `.scratch/hmm-lush/HMMLIB-ACCOUNT.md` §3 (the bit domain), §6 (the forward pass), §8 (the MDL apparatus), §9 (`run-add`) and §10 (the flat stream). The numeric contract is [ADR 0020](../../adr/0020-scaled-probability-domain-forward-backward.md). |
| Derived from | `packages/pfsmgraph-hmm/src/pfsmgraph/hmm/_forward_backward.py` `sha256:54735d9bfb7261d95506a64bc64ec67cd4b52daa4b17eca393829825fa9343d2` |
| Family | Hidden Markov model, **arc-emission (Mealy)**, [ADR 0015](../../adr/0015-arc-emission-mealy-formulation.md) |
| Variant | Forward-backward: the total probability of a record over every state path, reported as a description length in bits, and the expected arc and emission counts that Baum-Welch re-estimates from. It is not the decode, and it returns no path. |
| Objective | **Nothing is optimised.** The recurrences evaluate a sum over all `S^(N+1)` paths in the **sum-product semiring over the non-negative reals, with per-step renormalisation**. The additive identity is `0.0`, the multiplicative identity `1.0`, and `0.0` absorbs under multiplication, which is how an impossible arc removes every path through it. The description length sums `bits` of the scale factors, where `+∞` absorbs under addition. |
| Parallel decomposition | **Each timestep's `(record, state)` cells**, with `t` serial and every reduction serial and ascending. This recurrence has no anti-diagonals. See the section below. |
| Time complexity | `O(N·S²)` per record, for the recurrences and the counts alike |
| Space complexity | `O(N·S)` for `α`, `β` and the scale factors, plus `O(S²·A)` for the counts. Never `O(N·S²)`: ξ is formed one position at a time. The batched form is `O(B·L·S + B·S²·A)`. |
| Optimality | Exact. No approximation, and float64 rounding is taken in one fixed order. |
| Algorithm-specific params | None in the recurrence. The batched E-step takes `(init_p, trans_p, out_p, codes, lengths, device)`, and `device` is validated by the caller and never reaches the recurrence. |

**Two facts are lost first in translation, and neither is the direction of an objective.**
The first is that **emission is on the arc**: every term reads `out_p[i, j, k]` with both
endpoints, so no factor can be hoisted out of the inner reduction. The second is that
**the evaluation order is part of the specification**. Viterbi's `min` gives the same
result in any order; a sum does not. Two correct implementations that sum in different
orders disagree by an ulp in roughly half of all scale factors (48-77% measured in ADR
0020's evidence), and a suite meant to prove them equivalent then fails on a difference
neither got wrong. **The order stated below is contract for the same reason Viterbi's
tie-breaking rule is.**

## Deviations from the legacy source

Fifteen entries: four places the port behaves differently on purpose, five differences
judged harmless or forced, and six conventions it matched **on purpose**. The last group
leaves no trace in a diff, in the code or in the tests, which is why it is recorded at all.

**The recurrence below states this repository's behaviour, not the original's.**

### Different by design: the recurrence specifies the port

1. **An impossible sequence is zeros and `+∞`, never `NaN`.** `run-add` scales with a bare
   `/` (`hmm-trainer.lsh:547`, `:558`, `:576`), so a sequence whose scale factor reaches
   zero makes `α` `0/0`, and every re-estimated parameter becomes `NaN`. `update-data-p`
   and `update-data-dl` use `safe-/` for the same division. The port follows those two,
   taking `x / 0 = 0` everywhere it scales (ADR 0020 §3): the dead column and every one
   after it are all zeros, `β` is all zeros, every count is zero, and the description
   length is `+∞`. Constructed in TC-12 under `np.errstate(all="raise")`, so no `0/0` or
   `x/0` happens anywhere on the path, not merely in the output.
2. **ξ is never stored.** `run-add` materialises `P-t`, `(N+1)·S²` floats, only to sum it
   into `C-in`, `C-t` and `C-t-y` (`:579-617`; `HMMLIB-ACCOUNT.md` §9 measures 3.6 million
   entries for 50 states at 1,449 symbols). Here each position's `S × S` slice is formed,
   added and discarded. **No result changes**: every count cell still accumulates its ξ in
   ascending `t`, as stage 4's loop does, so this is memory, not arithmetic.
3. **Records, not one flat stream.** The original trains on the whole corpus concatenated
   (§10), so a transition from the last state of one sequence to the first state of the
   next is an ordinary modelled arc. Here the kernel runs per record, returns counts **per
   record**, and the caller sums them in record order. The original's computation is the
   one-record case, which is how the tracked fixtures are compared. A batch of `B` records is
   **not** the original's stream of `B` sequences, since no arc joins them.
4. **The batched E-step has no counterpart in the original.** It is an addition rather than
   a deviation, specified under *Batched recurrence*. It is listed here because its whole
   design is that no result depends on it.

### Different and judged harmless, or forced; none reaches the recurrence

5. **The trailing term is not reproduced.** Both `update-data-p` and `update-data-dl` finish
   with `safe-add--log2 result (Σ_j α*[N, j])` (`:179-182`, `:394-397`). Two cases:
   - **`N ≥ 1`: harmless.** The last scaled column sums to 1 within rounding, so the term
     is rounding noise. Measured 2026-09-15 on 300 random records: nonzero in 185, worst
     `9.61e-16` bits.
   - **`N = 0`: not noise, and specified as the port has it.** With no symbols the column
     is the unnormalised seed, so the term is `-log2(Σ init_p)`. That is exactly the value
     enumeration gives. The port reports `0` bits. Under `HMMParams`'s `SUM_TOL = 1e-5` the
     difference is at most `1.44e-5` bits per empty record; measured `-1.44e-6` bits at a
     seed summing to `1 + 1e-6`. **Decided 2026-09-15 on `feat/hmm-forward-phases` to
     specify `0` and record the difference rather than change the kernel** (TC-20). The
     kernel's docstring claims the opposite; see Notes.
6. **`*factor*` is not reproduced.** `data-p` and `data-dl` are multiplied by `(*factor*)`,
   fixed at `1.0` with four alternatives commented out (`:10-15`). Whatever experiment that
   was, it is switched off, so omitting it changes nothing.
7. **The `-1` log-zero sentinel is replaced by `+∞`.** `safe-add--log2` returns `-1` once
   either argument is the sentinel or a zero probability (`util.lsh:431-436`). Here
   `bits(0) = +∞`, which absorbs under addition and needs no guard. As for the decode,
   faithfulness to `-1` is **uncheckable**: there is no Lush runtime here, and the sentinel
   reaches no persisted value except through `update-total-dl`'s `1e100`.
8. **Each term associates differently.** The original multiplies `((α*·t)·o)` inside the
   accumulation, and for ξ `(((α*·t)·o)·β*)`. The port builds `w = t·o` once on the host,
   so its terms are `α·(t·o)`, and ξ is `(α·w)·β`. **This is forced by ADR 0020 §1, which
   puts the arc table on the host so every backend reads the same bits.** It is not a
   defect in either. Measured on uniform random factors, the two groupings differ in
   348,530 of 1,000,000 products. The consequence is that no bit-level agreement with the
   original is claimable, only agreement in value, and the only oracle in reach (TC-01) is
   a printed value anyway.
9. **Symbols are renumbered onto ADR 0011's block.** Lush's user symbols start at 2. Here
   `out_p`'s third axis spans the whole vocabulary, reserved codes included, and the arc
   table is built over the symbols the record uses. Every value is invariant under the
   widening, since a reserved fibre is exactly zero and so contributes exact zeros. The
   formalization names no reserved code; the reserved fibres being zero is what makes a
   `PAD` or `UNK` in a record impossible by arithmetic rather than by check.

### Matched on purpose: nothing but this section records these

10. **The seed is not normalised.** `α[0] = init_p` as given, and `Q[0] = 1` is set rather
    than computed, as in all three methods. The alternative, dividing row 0 by its sum, was
    not taken: with the seed left alone, `Σ bits(Q)` is `-log2 P(codes)` under the
    parameters **as given** for every `N ≥ 1`, including a seed summing to `1 + 1e-6`
    (TC-08). Renormalising would be `1.4e-6` bits off there.
11. **`β` is divided by the same scale factors as `α`, starting from `1 / Q[N]`.** This is
    `run-add`'s `β*`. It makes `α[t, i]·w[i, j, k]·β[t+1, j]` the pairwise posterior with no
    division by the likelihood, and `α[t]·β[t]·Q[t]` the state posterior. An unscaled `β`,
    the textbook convention, would need both divisions and change every count's rounding.
12. **The summation order is the original's loop order.** `α` accumulates over the source
    in ascending order for each destination, `Q` over the destination in ascending order
    as each column entry completes, and `β` over the destination in ascending order for
    each source. It was an accident of how the loops were written there; **here it is
    contract** (ADR 0020 §2).
13. **`C_in` is ξ₀ summed over the destination, not γ₀.** `run-add` forms `C-in` from
    `P-t[0]` (`:599-605`). The two agree within rounding and are not bit-identical, so
    choosing γ₀ would have been a silent change to every re-estimated `init_p`.
14. **Counts accumulate one position at a time, in ascending `t`, per cell.** Stage 4's loop
    nests `t` innermost; the port adds a whole ξ slice per position. Each count cell sees
    the same additions in the same order either way.
15. **The kernel neither validates nor raises.** Nothing in it checks shapes, ranges or
    stochasticity, and impossibility is a zero scale factor. Checking is the caller's
    (`baum_welch` refuses an impossible record before training). This split is kept for the
    decode's reason: a CUDA device function cannot raise, so a kernel that raised would
    have to be redesigned at every later phase rather than translated.

## Source Implementation

The tree is present in this repository, so this rendering was checked against a second
text rather than written from an account. `HMMLIB-ACCOUNT.md` is authoritative for what the
original *did*; this document is authoritative for what this repository does. From
`run-add` in `.scratch/hmm-lush/Code/HMMlib/hmm-trainer.lsh`:

Stage 1, the forward pass (`:533-548`, excerpt):

```lisp
(for (state-k 0 model-size-1)
     (alpha* position-i state-j
             (+ (alpha* position-i state-j)
                (* (alpha* (1- position-i) state-k)
                   (transition-p state-k state-j)
                   (output-p state-k state-j
                             (d-seq (1- position-i)))))))
; Update scaling factor
(Q-t position-i
     (+ (Q-t position-i)
        (alpha* position-i state-j)))
...
(alpha* position-i state-j
        (/ (alpha* position-i state-j)
           (Q-t position-i)))
```

Stage 2, the backward pass (`:557-577`, excerpt):

```lisp
(beta* data-seq-size state-j
       (/ 1.0 (Q-t data-seq-size)))
...
(for (state-k 0 model-size-1)
     (beta* position-i state-j
            (+ (beta* position-i state-j)
               (* (beta* (1+ position-i) state-k)
                  (transition-p state-j state-k)
                  (output-p state-j state-k
                            (d-seq position-i))))))
(beta* position-i state-j
       (/ (beta* position-i state-j)
          (Q-t position-i)))
```

Stages 3 and 4, ξ and the counts (`:587-617`, excerpt):

```lisp
(P-t position-i state-j state-k
     (* (alpha* position-i state-j)
        (transition-p state-j state-k)
        (output-p state-j state-k (d-seq position-i))
        (beta* (1+ position-i) state-k)))
...
(C-in state-j (+ (C-in state-j) (P-t 0 state-j state-k)))
...
(for (position-i 0 (1- data-seq-size))
     (C-t state-j state-k
          (+ (C-t state-j state-k) (P-t position-i state-j state-k)))
     (C-t-y state-j state-k (d-seq position-i)
            (+ (C-t-y state-j state-k (d-seq position-i))
               (P-t position-i state-j state-k))))
```

The description length is `update-data-p`'s, in its scaled loop and trailing term
(`:178-182`):

```lisp
(setq result-p (safe-add--log2 result-p (Q-t position-i))))
(let ((sum-alpha* 0.0))
  (for (state-j 0 model-size-1)
       (setq+ sum-alpha* (alpha* data-seq-size state-j)))
  (setq result-p (safe-add--log2 result-p sum-alpha*)))
```

## Definitions

| Name | Shape | Meaning |
|------|-------|---------|
| `S` | scalar | number of states |
| `A` | scalar | vocabulary size: the **whole** vocabulary, reserved codes included |
| `N` | scalar | number of symbols in the record. **A path over `N` symbols visits `N + 1` states**, so every state-indexed array below has `N + 1` rows and every symbol-indexed one has `N`. |
| `init_p` | `(S,)` | initial state distribution, used **as given**, not renormalised |
| `trans_p` | `(S, S)` | `trans_p[i, j]` = probability of the arc `i → j` |
| `out_p` | `(S, S, A)` | `out_p[i, j, k]` = probability of emitting `k` while crossing `i → j`. **Indexed by both endpoints.** Reserved fibres are exactly zero in a valid model. |
| `codes` | `(N,)` | integer codes, `0 ≤ codes[t] < A`, pre-encoded by the caller |
| `w` | `(S, S, A)` | `w[i, j, k] = trans_p[i, j] × out_p[i, j, k]`, one product per arc and symbol. An implementation may build it only over the symbols the record uses; each entry is the same product either way. |
| `Q` | `(N+1,)` | `Q[t]` = the scale factor of column `t`: the sum of the unscaled forward column for `t ≥ 1`, and `Q[0] = 1` |
| `α` | `(N+1, S)` | `α[t, i]` = the forward variable after `t` symbols, divided by `Q[1] × … × Q[t]`. Each row from 1 on sums to 1 within rounding, or is all zeros once the record is impossible. |
| `β` | `(N+1, S)` | `β[t, i]` = the backward variable from state `i` before symbol `t`, divided by `Q[t] × … × Q[N]` |
| `ξ` | `(N, S, S)` | `ξ[t, i, j]` = the posterior probability of crossing `i → j` while emitting symbol `t`. **Formed per position and never stored.** |
| `γ` | `(N+1, S)` | `γ[t, i]` = the posterior probability of occupying `i` before symbol `t`. A helper, not part of any backend's signature. |
| `C_in`, `C_t`, `C_e` | `(S,)`, `(S, S)`, `(S, S, A)` | expected initial, arc and arc-emission counts: `run-add`'s `C-in`, `C-t`, `C-t-y` |
| `DL` | scalar | the record's description length in bits, `-log2 P(codes)` |
| `bits(p)` | — | `-log2(p)`, with `bits(0) = +∞`. The only place a probability becomes a logarithm, evaluated on the host. |
| `x ⊘ y` | — | `x / y`, except `0` when `y = 0`, whatever `x` is |

**Notation for an ordered sum.** `Σ↑_{i=0..S-1} f(i)` means the left fold
`(…((f(0) + f(1)) + f(2)) + …) + f(S-1)`. Only its order is contract. Starting the fold from
`0.0` instead of from `f(0)` gives the same bits for non-negative terms, except for the sign
of an all-zero sum, which no comparison here can observe. **Every sum in this document is
one of these**, and none may be replaced by a reduction whose order is unspecified.

## Recurrence

**Boundary note.** This pseudocode operates on pre-encoded integer inputs. The caller encodes
symbols to integers before invocation and decodes them back afterwards. No symbol operation
occurs within this pseudocode, and `codes[t]` indexes `out_p`'s third axis directly, with no
offset arithmetic anywhere.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   w[i, j, k] = trans_p[i, j] × out_p[i, j, k]                            │
│                                                                          │
│   Forward, for t ← 1 to N, with k = codes[t-1]:                          │
│     c[j]     = Σ↑_{i=0..S-1} ( α[t-1, i] × w[i, j, k] )     for each j   │
│     Q[t]     = Σ↑_{j=0..S-1} c[j]                                        │
│     α[t, j]  = c[j] ⊘ Q[t]                                               │
│                                                                          │
│   Backward, for t ← N-1 downto 0, with k = codes[t]:                     │
│     r[i]     = Σ↑_{j=0..S-1} ( w[i, j, k] × β[t+1, j] )     for each i   │
│     β[t, i]  = r[i] ⊘ Q[t]                                               │
│                                                                          │
│   Pairwise posterior, for t ← 0 to N-1, with k = codes[t]:               │
│     ξ[t, i, j] = ( α[t, i] × w[i, j, k] ) × β[t+1, j]                    │
│                                                                          │
│   Counts:                                                                │
│     C_in[i]       = Σ↑_{j=0..S-1} ξ[0, i, j]                             │
│     C_t[i, j]     = Σ↑_{t=0..N-1} ξ[t, i, j]                             │
│     C_e[i, j, k]  = Σ↑_{t : codes[t] = k} ξ[t, i, j]                     │
│                                                                          │
│   Description length:                                                    │
│     DL = Σ↑_{t=0..N} bits(Q[t])                                          │
│                                                                          │
│   State posterior (helper):                                              │
│     γ[t, i] = ( α[t, i] × β[t, i] ) × Q[t]                               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**The emission factor cannot be hoisted.** `w[i, j, k]` depends on both endpoints, so it
stays inside the reduction over `i` in the forward pass and over `j` in the backward one.
Building `w` on the host moves only a multiplication. The factor is still read per arc,
unlike the `B[state, symbol]` of the state-emission formulation.

**Every product associates as written, and that is contract.** Each forward and backward
term is a product of **two** factors, so commutativity is all it needs. ξ and γ each have
**three**, and the grouping shown fixes their rounding. `(α × w) × β` and `α × (w × β)`
differ at the last bit.

**Why ξ needs no division by the likelihood.** `α[t]` carries `1/(Q[1]⋯Q[t])` and `β[t+1]`
carries `1/(Q[t+1]⋯Q[N])`, so their product carries `1/(Q[1]⋯Q[N]) = 1/P(codes)`, with
`Q[0] = 1`. This is deviation 11, and it is why `β`'s scaling is specified rather than free.

## Base Cases

```
α[0, i] ← init_p[i]                        for i ← 0 to S-1     (as given; deviation 10)
Q[0]    ← 1
β[N, i] ← 1 ⊘ Q[N]                          for i ← 0 to S-1
C_in    ← all zeros when N = 0              (there is no ξ[0])
DL      ← bits(Q[0]) = 0 when N = 0         (deviation 5; not -log2 Σ init_p)
```

**Impossibility is arithmetic.** If the first `t` at which `Q[t] = 0` exists, then `α[t']`
is all zeros for every `t' ≥ t`, `Q[t'] = 0` for every `t' ≥ t`, `β[N]` is all zeros and
hence so is every `β` row, every ξ and every count is `0`, and `DL = +∞`. No branch
produces this. It follows from `⊘` and `bits(0) = +∞`.

## Iteration Order

```
function FORWARD_BACKWARD(init_p, trans_p, out_p, codes)
    Input:  init_p  — real array of length S
            trans_p — real matrix S × S
            out_p   — real array S × S × A
            codes   — integer array of length N
    Output: (α, β, Q)

    build w                                   — elementwise; no order to fix
    set base cases
    for t ← 1 to N do                         — strictly sequential: α[t] reads α[t-1]
        for j ← 0 to S-1 do                   — independent for fixed t
            c[j] ← 0
            for i ← 0 to S-1 do               — the ordered reduction
                c[j] ← c[j] + α[t-1, i] × w[i, j, codes[t-1]]
            end for
        end for
        Q[t] ← Σ↑_{j} c[j]                    — serial, after every c[j] exists
        for j ← 0 to S-1 do  α[t, j] ← c[j] ⊘ Q[t]  end for
    end for
    for t ← N-1 downto 0 do                   — strictly sequential: β[t] reads β[t+1]
        for i ← 0 to S-1 do                   — independent for fixed t
            r ← 0
            for j ← 0 to S-1 do               — the ordered reduction
                r ← r + w[i, j, codes[t]] × β[t+1, j]
            end for
            β[t, i] ← r ⊘ Q[t]
        end for
    end for
    return (α, β, Q)
end function

function E_STEP(init_p, trans_p, out_p, codes)
    Input:  as FORWARD_BACKWARD
    Output: ((C_in, C_t, C_e), DL)

    (α, β, Q) ← FORWARD_BACKWARD(init_p, trans_p, out_p, codes)
    C_in ← zeros(S);  C_t ← zeros(S, S);  C_e ← zeros(S, S, A)
    for t ← 0 to N-1 do                       — ascending; one addition per cell per t
        k ← codes[t]
        for i ← 0 to S-1 do
            for j ← 0 to S-1 do               — cells independent for fixed t
                x ← (α[t, i] × w[i, j, k]) × β[t+1, j]
                C_t[i, j]    ← C_t[i, j] + x
                C_e[i, j, k] ← C_e[i, j, k] + x
            end for
            if t = 0 then  C_in[i] ← Σ↑_{j} ξ[0, i, j]  end if
        end for
    end for
    DL ← Σ↑_{t=0..N} bits(Q[t])               — logarithms on the host
    return ((C_in, C_t, C_e), DL)
end function
```

`Q[t]` is summed from the completed `c`, not interleaved with each `c[j]` as the original
interleaves it. The two orders are the same fold: `Q[t]`'s additions still happen in
ascending `j`, and nothing else reads the partial sums.

## Backtrace

This recurrence produces no backtrace. Its outputs are totals over every path, and no path is
chosen, so there is no predecessor array and no tie-breaking rule. **The evaluation order in
the recurrence takes the place a tie-breaking rule has in the decode**: it is the property
two correct backends could otherwise legitimately disagree on.

## Batched recurrence

The same recurrences over `B` records padded to a common width. Row `b` is `E_STEP` on
record `b` alone, **bit for bit**, which is what lets `baum_welch`'s result ignore
`batch_size` (TC-16, TC-17).

| Name | Shape | Meaning |
|------|-------|---------|
| `B`, `L` | scalars | records in the batch; the longest record's length |
| `codes` | `(B, L)` | `pad_collate`'s codes, padding past each record's end |
| `n` | `(B,)` | each record's length, `n[b] ≤ L` |
| `live(b, t)` | — | `t < n[b]`: symbol position `t` exists in record `b`. **Derived from `n`, never taken as a separate mask**, so the two cannot disagree. |
| `α`, `β` | `(B, L+1, S)` | as above, one row per record |
| `Q` | `(B, L+1)` | as above |

```
α[b, 0, i] ← init_p[i];   Q[b, 0] ← 1

for t ← 1 to L do
    if live(b, t-1) then
        forward step of the recurrence, over codes[b, t-1]
    else                                          — a padded step
        Q[b, t]    ← 1                            — exactly 1: bits(1) adds nothing
        α[b, t, i] ← α[b, t-1, i]                 — carried; never read (see below)
    end if
end for

for t ← L downto 0 do
    if t ≥ n[b] then
        β[b, t, i] ← 1 ⊘ Q[b, n[b]]               — each record's own end, at every padded row
    else
        backward step of the recurrence, over codes[b, t]
    end if
end for

ξ[b, t, i, j] ← (α[b, t, i] × w[i, j, codes[b, t]]) × β[b, t+1, j]   if live(b, t)
             ← 0.0 exactly                                          otherwise, by SELECTION

C_in[b], C_t[b], C_e[b] as above, from ξ[b, ·], in ascending t over all L positions
DL[b] ← Σ↑_{t=0..L} bits(Q[b, t])
```

**Counts are returned per record and never summed across `b`.** `(c₀ + c₁) + c₂` differs from
`c₀ + (c₁ + c₂)` in about a fifth of random triples, so a kernel that summed a batch would tie
the trained model to `batch_size`. The sum over records is the caller's, in record order.

**Why row `b` is the per-record E-step, bit for bit.** Every reduction runs over a state axis
within one record, and nothing combines values across `b`. A padded step contributes
`bits(1) = -0.0` to `DL` and an exact `0.0` to every count, and adding either changes no bit
of a non-negative sum. `β` resumes at `n[b] - 1` from exactly the value it starts from when
the record is alone. An implementation may build `w` over the symbols present anywhere in the
batch, since each entry is the same product of the same two floats.

**Three choices here are contract, and one is not.**

- *`Q[b, t] = 1` exactly at a padded step.* Any other value moves `DL[b]`.
- *`β`'s seed is read from column `n[b]`, not column `L`.* Column `L`'s `Q` is `1` for any
  padded record, so reading it would seed every short record's backward pass with `1`.
- *ξ at a padded position is selected to `0.0`, never multiplied by a mask.* A multiplication
  propagates a non-finite ξ as `NaN`, and `β = 1 ⊘ Q[n[b]]` overflows to `+∞` when `Q[n[b]]`
  is subnormal (`1 / 5e-324 = inf`). *Reasoned, not measured on a model.*
- *Not contract: the carry of `α` past `n[b]`.* Nothing reads `α[b, t]` for `t > n[b]`, since
  its ξ is selected away and its `Q` is forced to 1, so whether an implementation carries it,
  zeroes it or leaves it unwritten is unobservable, and no test claims otherwise. This is the
  inverse of the decode, whose carry is observable because its final argmin reads column `L`.

**Padding is inert only if the selection is real, and valid models cannot show that.** A valid
model's `PAD` fibre is zero, so padding already weighs 0 and dropping the selection is an
equivalent mutant on any `HMMParams`. TC-16's `pad-emittable` case passes raw arrays with
`PAD` emittable, which makes the selection the only thing between padding and the counts.

**Loop order is free.** Stepping every record through each `t`, or finishing one record before
starting the next, performs the same arithmetic on each row.

## Parallel decomposition

**Each timestep's `(record, state)` cells**, with `t` serial. Decided 2026-09-15 on
`feat/hmm-forward-phases`, before phase 3 exists, and for phase 3 to realise rather than
discover. The per-record kernel is the `B = 1` case.

- **Forward step `t`:** the `B·S` cells `(b, j)` are independent. Each reads only
  `α[b, t-1, ·]` and `w`, and writes only `c[b, j]`. The reduction over `i` inside a cell stays
  **serial and ascending**. Then `Q[b, t]` is a **serial ascending** sum over `j` per record,
  after the parallel region, and `α[b, t, j] = c[b, j] ⊘ Q[b, t]` is elementwise and may be
  parallel again.
- **Backward step `t`:** the cells `(b, i)` are independent, each a serial ascending reduction
  over `j` followed by `⊘ Q[b, t]`.
- **Counts at position `t`:** every count cell `C_t[b, i, j]` and `C_e[b, i, j, k]` is its own
  accumulator, receiving exactly one addition per `t`, so the cells `(b, i)` or `(b, i, j)`
  are independent. Positions stay serial and ascending. `C_in[b, i]` at `t = 0` is a serial
  ascending sum over `j`.
- **`DL[b]`:** a serial ascending sum over `t`, on the host, through `bits`.

**No reduction may become a parallel reduction, and the reason is contract rather than
performance.** A parallel sum combines partial sums in an order chosen by the scheduler, which
changes the last bit of about half of all scale factors. Every value stays within any
tolerance, so only a bit-exact cross-backend test notices (TC-15, TC-22).

**This recurrence has no anti-diagonals.** `α[t, j]` reads every `α[t-1, i]`, so the
anti-diagonal `{t + j = c}` holds `(t, j)` and `(t-1, j+1)`, and the first reads the second.
The argument is the decode's, unchanged, and ADR 0016's `Resolved` section already withdraws
the wavefront for this family.

**The two decompositions not taken:**

- **Whole records**, `prange` over `b` with each record's full loop serial. It is the fastest
  CPU axis: for the decode it was 1.4-4x faster wherever `B` filled the threads (4-vCPU Xeon,
  2026-09-15). It is declined for the decode's reason. A device thread cannot run a record's
  whole `L·S²` loop, so only the per-step cell grid is phase 4's launch geometry, and ADR 0016
  scopes phase 3 to rehearsing that.
- **An associative scan over time.** Sum-product with renormalisation is associative only up
  to the scale factors, so a parallel prefix is possible in principle. It is rejected. It
  re-associates exactly the sums ADR 0020 §2 fixes, and computes scale factors of prefix
  products rather than of single columns, so it forfeits bit-exactness with the reference.
  It costs `O(N·S³)` against `O(N·S²)`, and it would need its own equivalence argument rather
  than a differential test. This answers the item ADR 0020 leaves under `Open`.

**No implementation fuses multiply-add** (ADR 0020 §4). `c ← c + α × w` is exactly the shape a
contracting compiler turns into one fused instruction, which rounds differently. The Cython
phase on x86-64 is safe because `meson.build` passes no `-march`, so the baseline has no FMA.
aarch64 GCC and Apple clang contract by default and will need `-ffp-contract=off`. Numba
kernels never set `fastmath`. **Whether numba-cuda contracts in device code is unmeasured** and
is settled at phase 4 by TC-21.

**Logarithms stay on the host.** The device performs only `+`, `×` and `⊘` on a host-built
`w`, and returns `Q`. `DL` is `bits(Q)` summed on the host. A device `log2` rounds one ulp away
from numpy's in about a quarter of inputs, as measured for the decode on an NVIDIA L4.

**The parallelism is thin, and that is accepted.** `t` is strictly sequential in both passes,
so a parallel region opens and closes once per timestep around `O(B·S²)` work. The decode's
phase 3 measured a flat per-timestep fork/join cost that dominated below `S ≈ 64` on one host
and `S ≈ 250` on another. ADR 0016 scopes phase 3 to parallel correctness, so a phase-3 kernel
slower than phase 2 is within what the phase is for.

## Differential oracles

**The manifest declares none** (`[algorithms.forward_backward]` has no `oracles` key). Three
checks exist that are not downstream of this recurrence, and **only the first was produced by
the legacy implementation**.

1. **The original's logged data description length.**
   - *What produced it.* `update-data-dl` (`hmm-trainer.lsh:346-399`), during the original's
     own training runs. It is the last row's `data-dl` in each tracked model's
     `_training_log`, printed with `%g`, at the `d` stored beside it.
   - *Which inputs it pairs with.* The three model directories
     `.scratch/hmm-lush/Training/set02a/set02a_200/{m001_0001_001,m001_0005_005,m008_0001_008}.hmm`,
     quantised to `d` as `update-approx-*` quantises them, and the `set02a_200` corpus
     concatenated as one record. Inputs and outputs are tracked together.
   - *Not an equality check, for two reasons that have nothing to do with this recurrence.*
     The log prints six significant figures, which is 0.005 bits at these magnitudes. The
     parameters were saved to four decimals, so `x·d` near a half can round the other way from
     the original's full-precision value.
   - *The divergence, pinned.* `m001_0001_001` at `d = 29` lands **0.009 bits** below its log;
     the other two are within the print. The bound is `0.02` bits. **The same test asserts the
     unrounded description length misses by more than 1 bit** (1.5 to 29 measured), so the
     comparison can fail and the quantisation is load-bearing.
   - *What it reaches.* The forward recurrence and `DL` only, through
     `_data_description_length` in `_baum_welch.py`. It checks no `β`, no ξ and no count. It
     is carried by **TC-01**.
2. **`hmmlearn`'s `CategoricalHMM`**, 0.3.3, `implementation="scaling"`. It is an external
   implementation rather than the legacy one, and it is state-emission. So the comparison runs
   on the **destination-only** case, where `out_p[i, j, k]` does not depend on `i`, and needs
   a mapping whose two halves are both load-bearing. Its `startprob_` is
   `init_p @ trans_p`, since its first state is this model's `s_1`, and its posteriors are
   `γ[1:]`. Log-likelihood and posteriors agree to `1e-12` (measured `3e-15`). It cannot reach
   arc-dependent emission, which is exactly the family that is not its own. Carried by
   **TC-02 to TC-04**.
3. **Exact enumeration in rational arithmetic.** Every one of `S^(N+1)` paths is weighted in
   `Fraction`. That shares neither the recurrence nor any rounding with the kernel, and it
   reads precisely the floats the kernel reads. It was written in this repository, so it is
   not an oracle in this template's sense. It is still the check that catches a kernel correct
   for the wrong model, such as an arc table transposed everywhere, which passes every
   identity about values. It covers `S ≤ 4` and `N ≤ 9`. Carried by **TC-05 to TC-08**.

**A green differential suite is evidence about these inputs, not about the order.** None of
the three is bit-level. The oracle is a printed value, `hmmlearn` reduces in its own order,
and enumeration is compared at `rtol = 1e-12`. So all three would accept a reordered sum. The
evaluation order is held by TC-15 and, across backends, TC-22.

## Test Cases

Provenance is recorded per case: `oracle` (an output the legacy implementation produced),
`extracted` (from a test the implementation already passes), or `constructed` (written from
the recurrence). **A case extracted from a passing test is ratified behaviour, not specified
behaviour.** Tests are in `packages/pfsmgraph-hmm/tests/`.

### TC-01: the forward pass reproduces the original's logged data-dl — `oracle`, relational
    Input:  each of the three tracked models, quantised to its stored d; the corpus as one record
    Expect: DL within 0.02 bits of the log's data-dl; the unrounded DL more than 1 bit away
    Test:   test_baum_welch.py::test_the_data_description_length_is_the_originals_logged_value

### TC-02: DL is hmmlearn's negative log-likelihood — `extracted`, relational
    Input:  destination-only models; startprob_ = init_p @ trans_p
    Expect: DL = -log2 likelihood, rel 1e-12
    Test:   test_hmmlearn_oracle.py::test_description_length_matches_hmmlearn_log_likelihood

### TC-03: γ[1:] is hmmlearn's posteriors — `extracted`, relational
    Test:   test_hmmlearn_oracle.py::test_state_posteriors_match_hmmlearn

### TC-04: the hmmlearn comparison can fail — `extracted`, relational
    Expect: handing hmmlearn init_p unchanged, or reading γ one position early, misses by
            more than 1e6 × the tolerance
    Test:   test_hmmlearn_oracle.py::test_the_comparison_can_fail

### TC-05: DL is -log2 of the exact probability — `extracted`, relational
    Input:  six random models, S ≤ 4, N ≤ 9, up to half their arcs dead
    Expect: rel 1e-12 against rational enumeration
    Test:   test_forward_backward.py::test_the_description_length_is_minus_log2_of_the_exact_probability

### TC-06: γ is the exact occupation probability — `extracted`, relational
    Expect: rtol 1e-12, atol 0: an exact zero in the enumeration is an exact zero here
    Test:   test_forward_backward.py::test_the_state_posteriors_are_the_exact_occupation_probabilities

### TC-07: C_in, C_t and C_e are exact — `extracted`, relational
    Test:   test_forward_backward.py::test_the_{initial,transition,emission}_counts_are_exact

### TC-08: the seed is used as given — `extracted`, relational
    Input:  init_p summing to 1 + 1e-6, N = 6
    Expect: agreement with enumeration at rel 1e-12, which a renormalised seed misses by 1.4e-6 bits
    Test:   the seed-sums-to-1+1e-6 case of TC-05 to TC-07

### TC-09: the posterior identities hold for any model — `extracted`, property
    Input:  five random models S 1-24, N 1-300, up to half their arcs dead; the 8-state fixture
            over its 1268-symbol corpus
    Expect: every γ row sums to 1; Σ init_p·β[0] = 1; Σ_j ξ = γ[:N] and Σ_i ξ = γ[1:]
            (the second is what a transposed β breaks); Σ_k C_e = C_t; ΣC_t = N; C_e zero at
            unused symbols and on dead arcs; C_in = γ[0]; DL finite and ≥ 0. All within 1e-12
            per summed term.
    Test:   test_forward_backward.py, "the identities"

### TC-10: the outputs have the N+1 geometry — `extracted`, exact
    Expect: α, β, γ (N+1, S); Q (N+1,); C_in (S,); C_t (S, S); C_e (S, S, A)
    Test:   test_forward_backward.py::test_the_outputs_have_the_arc_emission_geometry

### TC-11: an empty record — `extracted`, exact
    Input:  N = 0, a seed summing to 1 within rounding
    Expect: α = [init_p]; Q = [1]; DL = 0; γ = α; every count zero
    Test:   test_forward_backward.py::test_an_empty_record_has_no_crossings_and_costs_nothing

### TC-12: an impossible sequence — `constructed` when written, now `extracted`, exact
    Input:  init (1, 0), trans = I, state 0 emits only 0, state 1 only 1; codes (0, 0, 1, 0)
    Expect: under errstate(all="raise"): Q = (1, 1, 1, 0, 0); α[3:] = 0; DL = +∞; β, γ and
            every count all zeros; no NaN anywhere
    Test:   test_forward_backward.py::test_an_impossible_sequence_is_infinite_bits_and_zero_everything_else

### TC-13: an exactly uniform model is computed exactly — `extracted`, exact
    Input:  S = A = 4, every probability 1/4; N ∈ {1, 7, 40}
    Expect: equality, not closeness: Q[1:] = 1/4, DL = 2N, γ = 1/4, C_in = 1/4, C_t = N/16,
            C_e[·,·,k] = count(k)/16. Every intermediate is dyadic, so nothing rounds.
    Test:   test_forward_backward.py::test_an_exactly_uniform_model_is_computed_exactly

### TC-14: a state reachable only through dead arcs is never occupied — `extracted`, exact
    Input:  ADR 0020's 3-state autograd model: state 2 cannot start and no live arc enters it
    Expect: every γ, count and C_in touching state 2 is an exact zero, no NaN; DL = N bits
    Test:   test_forward_backward.py::test_a_state_reachable_only_through_dead_arcs_is_never_occupied

### TC-15: the reference is the literal ascending loops, bit for bit — `extracted`, exact
    Input:  S = 16, A = 6, N = 50, 30% dead arcs
    Expect: α, β, Q, DL, C_in, C_t, C_e byte-identical to scalar Python loops summing in the
            order of the recurrence, with ξ grouped (α × w) × β
    Test:   test_forward_backward.py::test_the_reference_reproduces_the_literal_ascending_loops_bit_for_bit
    Note:   the only case that sees the evaluation order; every tolerance-based case accepts
            a reordered sum

### TC-16: every batched row is the per-record E-step, bit for bit — `extracted`, exact
    Input:  ragged batches: mixed lengths, almost all padding, S = 1, empty records, an
            impossible record, and PAD emittable in raw arrays
    Expect: each row's counts byte-identical and DL equal to E_STEP on that record alone
    Test:   test_baum_welch.py::test_every_row_of_a_batch_is_the_per_record_step_bit_for_bit

### TC-17: training ignores batch_size — `extracted`, exact
    Input:  five and seven ragged records; batch_size 1, 2, 3, None
    Expect: identical cycle counts and description-length histories
    Test:   test_baum_welch.py::test_training_is_bit_identical_at_every_batch_size

### TC-18: a non-ordered backend is held to N·eps·max(1, x) — `extracted`, relational
    Input:  generated models, the Lush fixtures, EM trajectories, and constructed subnormal arcs
    Expect: every count and DL within 4·N·eps·max(1, |reference|); no negative count
    Test:   test_baum_welch_backends.py, the kernel-level section
    Note:   a torch backend cannot fix its summation order, so this is the one tolerance in
            the specification. Compiled phases are held to TC-22, not to this.

### TC-19: nothing to count gives zero counts on every kernel — `extracted`, exact
    Input:  an empty record; a record carrying UNK
    Expect: all counts zero; DL 0 and +∞ respectively
    Test:   test_baum_welch_backends.py::test_every_kernel_returns_zero_counts_where_there_is_nothing_to_count

### TC-20: an empty record costs 0 bits whatever its seed sums to — `constructed`, exact
    Input:  N = 0; init_p scaled to sum to 1 + 1e-6, and to 1 - 1e-6
    Expect: DL = 0 exactly, and every count zero, on every backend
    Why:    deviation 5. Enumeration gives -log2 Σ init_p, and so did the original's trailing
            term; this repository specifies 0. TC-11 uses a seed summing to 1 within rounding,
            so it cannot tell the two apart.
    NOT IMPLEMENTED.

### TC-21: no backend fuses multiply-add — `constructed`, exact
    Input:  values chosen so that fma(α, w, c) ≠ c + α × w in float64 at an accumulation step,
            found at test time on the host
    Expect: every compiled phase's α, Q and counts byte-identical to the reference
    Why:    ADR 0020 §4. No current test contains an input on which contraction is visible, so a
            contracting build passes the suite. Settles numba-cuda at phase 4.
    NOT IMPLEMENTED: needs a compiled phase. Phase 2 writes it; phase 4 requires it on a device.

### TC-22: every backend's kernels are the reference's, bit for bit — `constructed`, exact
    Input:  generated models and ragged batches, including empty, length-1, impossible and
            PAD-emittable rows; for phase 3, at 1 thread and at the maximum
    Expect: _forward_backward's α, β and Q, and _e_step_batch's counts and DL, byte-identical
            to backend "python" on the same host
    Why:    the compiled phases are held to equality, not to TC-18's tolerance. The decode's
            TC-23 and TC-26 are the precedent.
    NOT IMPLEMENTED: needs a compiled phase.

### TC-23: the result does not depend on memory layout — `constructed`, exact
    Input:  trans_p and out_p as C-ordered and as Fortran-ordered copies of the same values
    Expect: every output byte-identical, per-record and batched
    Why:    terms.sum(axis=0) agrees with the ascending loop on a C-ordered array and disagreed
            in 50 of 50 trials on a Fortran-ordered copy at S ≥ 16 (the kernel's docstring), so
            a reduction that survives TC-15 can still depend on layout. TC-15 uses C order only.
            Checked 2026-09-15: the reference holds, 0 of 100 comparisons differing, S 16-47.
    NOT IMPLEMENTED.

## Notes

- **This document was recovered from an implementation, not written before one.** Where the
  two disagree, this document is wrong until someone establishes otherwise, and the
  `extracted` cases are ratified behaviour. Unlike the decode, the only legacy-produced check
  (TC-01) reaches the forward pass through quantised parameters and a printed value, so the
  backward pass and the counts rest on enumeration and `hmmlearn`, **neither of which is the
  original**.
- **`Derived from` decides staleness.** When `_forward_backward.py`'s SHA-256 stops matching,
  this document is stale and must be re-derived rather than patched. The compiled phases
  derive from the same file, so an edit there makes this document and every compiled phase
  stale at once.
- **Finding: the kernel's docstring overclaims at `N = 0`.** `_forward_backward`'s docstring
  says `Σ bits(scale)` is `-log2 P(codes)` "including when `init_state_p` sums to 1 only
  within `HMMParams`'s tolerance". That holds for `N ≥ 1`, since the unnormalised seed is
  absorbed into `Q[1]`. At `N = 0` the kernel returns `-0.0` where the claim requires
  `-log2 Σ init_p`, measured `-1.44e-6` bits at `1 + 1e-6`. No test sees it: the enumeration
  cases all have `N ≥ 4`, and TC-11's seed sums to 1 within rounding. **Not corrected here.**
  The specification follows the kernel (deviation 5, TC-20), and the docstring's wording is a
  separate change.
- **Finding: ADR 0020 §1's association argument covers only two-factor terms.** It reasons that
  "IEEE multiplication is commutative, so how a longer product associates never enters the
  contract". That is true of every forward and backward term, and false of ξ and γ, which have
  three factors: groupings differ in about 35% of products (deviation 8). TC-15 pins ξ's
  grouping. **γ's grouping is pinned by nothing**, which is harmless only because γ is in no
  backend's signature.
- **Finding for goal 2's oracle decision.** The manifest's comment on
  `[algorithms.forward_backward]` says the tracked models' `_total_dl` values are candidates but
  "scored with quantized parameters, so they are not a direct check of this kernel". TC-01 shows
  the quantised path *is* a check of this kernel's forward recurrence, through `data-dl` rather
  than `_total_dl`, to 0.02 bits. The three `_training_log` files are therefore candidates for an
  `oracles` entry. `hmmlearn` is a library, not a file, and does not fit the manifest's `oracles`
  shape.
- **ADR 0020's `Open` item on an associative scan is answered in *Parallel decomposition*.**
  Moving it to that ADR's `Resolved` section is a separate change to the ADR.
- **No tie case was constructed, and none is missing.** A sum has no argmin. The decode's tie
  case exists because learned parameters never tie. Here the analogous blind spot, a reordered
  sum that no tolerance can see, is already covered by TC-15, which this suite had before the
  recovery.
- **Specified and unimplemented: TC-20, TC-21, TC-22 and TC-23.** TC-21 and TC-22 need a compiled
  phase and belong to goals 2-4. TC-20 and TC-23 test the reference and could land at any time.
