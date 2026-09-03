# The Lush HMM library, in its own terms

**Subgoal 1 of revision `02-hmm-v0.1.0`**, executed on branch `docs/hmmlib-account`
(`docs/plan/docs-hmmlib-account/TODO.md`). This document describes what `Code/HMMlib/`
*is*, before any translation and without reference to the Python design it will become.
It is the `HMMlib` counterpart of [`ACCOUNT.md`](ACCOUNT.md), which covers `Code/SeqData/`
— the container half — and whose conventions it follows.

Keeping the account separate from the migration plan is deliberate, and the reason is the
one `ACCOUNT.md` gives: an account written with the target design in view tends to
describe the original as a deviation from a thing that does not exist yet. Where this
document says the code is wrong, it means wrong *in its own terms* — inconsistent with
itself, or with the arithmetic it has already committed to a few lines above.

**Sources.** Four files, 2,044 lines:

| File | Lines | Last compiled |
|---|---|---|
| `Code/HMMlib/hmm.lsh` | 319 | 2009-07-20 |
| `Code/HMMlib/hmm-param.lsh` | 386 | 2009-07-20 |
| `Code/HMMlib/hmm-trainer.lsh` | 1102 | 2011-02-01 |
| `Code/HMMlib/hmm-trainer-view.lsh` | 237 | not compiled |

plus the helpers `safe-/`, `safe-add--log2`, `safe->--log`, `log2`, `int-delta`,
`calculate-entropy`, `int-code-length`, `comb-code-length`, `round-using`, `round-to`,
`rand-p-vector`, `LU-solve`, `zero-padded` and the `setqC` macro from
`Code/Utility/util.lsh` (574 lines), and the entry scripts `Training/hmm-train-new` and
`Training/hmm-train-load`.

**The compile dates are evidence, not metadata.** The import reset every `.lsh` mtime to
the import date, but `Code/HMMlib/C/` holds the `dhc-make` C output and was not touched:
`hmm.c` and `hmm_param.c` are dated 2009-07-20, `hmm_trainer.c` 2011-02-01. Those are the
last times each source was compiled, so they bound the last edit from below. The
`hmm-trainer.lsh~` backup differs from `hmm-trainer.lsh`, so the trainer was edited at
least once after its last save.

**Two corpora are tracked as specimens**, the same two `ACCOUNT.md` measures against:
`Training/set01z0/set01z0_100.sds/` and `Training/set11a_dInt/set11a_dInt.sds/`. Every
quantitative claim below was measured against them, and the measurements are collected in
the appendix.

**Provenance is code, not history.** Nothing here claims what was *run*. Where the code
admits a behaviour that may never have been exercised, it is marked **provenance
unknown** rather than asserted as a bug that bit. That marking carries more weight in this
account than in `ACCOUNT.md`, because the training entry points are interactive (§11) and
nothing in the tree records which buttons were pressed.

---

## 1. The central structural fact: emissions are on transitions

`hmm.lsh:40` declares

```lisp
((-idx3- (-float-)) output-p)
```

allocated at `hmm.lsh:86` as `(float-matrix size size alphabet-size)`. Every read of it
in all four files is `(output-p state-i state-j symbol-k)`. **This is a Mealy HMM: a
symbol is emitted while crossing a transition `i → j`, not while occupying a state.**

Nothing in the tree says so in words. It has to be read off the shape of the tensor and
the indexing, and it is the single fact most likely to be lost in translation, because
every textbook presentation and every library API is Moore — `B[state, symbol]`.

The consequence for the recurrences is that the emission factor is never separable from
the transition factor. Both the forward pass and Viterbi carry the product

```lisp
(* (transition-p state-k state-j)
   (output-p state-k state-j (d-seq (1- position-i))))
```

as one term (`hmm-trainer.lsh:165-167`, `229-231`). A Moore port would factor the
emission out of the inner loop over `state-k`; here it cannot be, because it depends on
`state-k` and `state-j` both.

The consequence for parameter count is quadratic rather than linear in the number of
states. For `set11a_dInt`'s alphabet of 25, a 50-state model carries **62,500** emission
parameters where a Moore model of the same size would carry 1,250.

## 2. Why Mealy: the N/N+1 geometry

`ACCOUNT.md` §6 records that `seq-state` holds `symbol-data` of length `size` alongside
`path-states` and `path-entropy` of length `size + 1`, and reads the `+1` as exact: a path
emitting *N* symbols visits *N+1* states, so the state arrays are indexed by the gaps
between symbols.

The trainer confirms that reading from the other side. `delta` and `psi` are
`(1+ data-seq-size) × size` (`hmm-trainer.lsh:197-198`), the trellis is indexed
`position-i` from `0` to `data-seq-size` inclusive, and the symbol consumed on the step
into position `i` is `(d-seq (1- position-i))`.

So the Mealy structure and the `+1` are the same design decision seen twice. States sit
between symbols; symbols sit on the transitions between states. The container half of the
library was already shaped by it, which is why `seq-state` looked slightly odd in
`ACCOUNT.md` and looks inevitable here.

## 3. The arithmetic is description length in bits

This is the second fact that does not survive a casual reading, and it reaches all the way
into the kernel.

```lisp
(de safe-add--log2 (sum x)
  (if (or (= sum -1)
          (= x 0))
      -1
    (- sum (log2 x))))

(de safe->--log (x y)
  (and (<> y -1)
       (or (= x -1)
           (> x y))))
```

`safe-add--log2` accumulates `sum - log₂(x)`. So a quantity built with it is a **negative
log-base-2 probability — a description length in bits** — and it grows as the probability
falls. `-1` is the log-zero sentinel: a genuine description length satisfies `-log₂p ≥ 0`,
so `-1` is unreachable and unambiguous. It is absorbing — once a term is `-1`, every
subsequent accumulation returns `-1`.

`safe->--log` is named as though it compared in log space and reads backwards until the
sentinel handling is worked through. It is true when `y` is not the sentinel, and either
`x` is the sentinel or `x > y`. Since smaller bits mean higher probability, it means
**"`y` is strictly better than `x`", with the sentinel worst**. Both call sites use it in
exactly that shape:

```lisp
(when (safe->--log (delta position-i state-j) cand-delta)
  (delta position-i state-j cand-delta)
  (psi position-i state-j state-k))
```

**So Viterbi here is a min-sum over bits, not a max-product over probabilities.** The two
are the same algorithm, but a port that reaches for `max` will invert every comparison.

A naming trap follows from this and is worth stating on its own: the slot `data-p`
(`hmm-trainer.lsh:43`) and the local `result-p` (`134`) are accumulated with
`safe-add--log2` and therefore hold **description lengths**, not probabilities, despite
the `-p` suffix that everywhere else in the library means "probability". `update-total-dl`
depends on it — `(if (< data-dl 0) (setq total-dl 1e100) ...)` at `430-433` is testing for
the `-1` sentinel, which only makes sense for a quantity in bits.

## 4. `hmm` — the model (`hmm.lsh`)

A plain state container with load/save and two computed summaries. Slots: `name`,
`alphabet-size`, `alphabet` (a `gptr` array of interned strings), `size`, `counter`,
`state-p`, `state-entropies`, `entropy`, `init-state-p`, `transition-p`, `output-p`, `d`,
`training-log`.

The constructor takes `(hmm-name data-seq-name in-size &optional in-counter)`. With
`counter = 0` it builds a fresh model: reads `_alphabet_size` and `_alphabet` **out of the
`.sds` corpus directory**, allocates the parameter matrices, sets `d` to `10000.0`, and
calls `init-random` with a noise width of `0.001`. With `counter ≠ 0` it delegates to
`load`, reconstructing the basename as `<name>_<counter:04d>_<size:03d>.hmm/`.

Two things about this are worth carrying forward. First, **the model borrows the corpus's
alphabet by reading its files directly** — there is no vocabulary object passed between
them, only a path and a file format. That is downstream of the seam `ACCOUNT.md` §1
identifies, where encoding is a build-time artefact of `format-sds` and a loaded corpus
has no encoding path at all. Second, the model's identity on disk encodes its *size*, so a
topology change produces a new file rather than overwriting one; `counter` increments on
every `keep-model` (`hmm-trainer.lsh:685`).

`init-random` (`217-225`) fills `init-state-p`, each row of `transition-p`, and each
`(i,j)` fibre of `output-p` with `rand-p-vector`, then recomputes the entropies. It is one
of the three methods in this file that `dhc-make` compiles.

### `update-entropy`, and the stationary distribution

`hmm.lsh:228-262` computes three things in sequence, and the first is a linear solve:

```lisp
(for (state-j 0 (1- size))
     (A 0 state-j 1.0)
     (for (state-i 1 (1- size))
          (A state-i state-j
             (- (transition-p state-j state-i)
                (int-delta state-i state-j)))))
(B 0 1.0)
(for (state-i 1 (1- size))
     (B state-i 0.0))
(LU-solve A B state-p)
```

Read carefully: row `0` of `A` is set to all ones and `B[0]` to `1`, so the first equation
is `Σⱼ πⱼ = 1`. Rows `1 … size-1` hold `A[i][j] = P[j][i] - δ[i][j]`, i.e. `(Pᵀ - I)`, with
`B[i] = 0`. So the system is

> `(Pᵀ - I)π = 0`, with the **first row replaced** by the normalization `Σπ = 1`.

`state-p` is therefore the **stationary distribution** of the transition matrix. The row
replacement is not decoration: `(Pᵀ - I)` is singular by construction — that is what makes
`π` an eigenvector — so the solve exists only because one redundant equation was traded
for the normalization. A port that writes down the homogeneous system as stated and hands
it to a dense solver gets a singular matrix, and a port that writes `(I - Pᵀ)` instead
gets the same null space but must still replace a row.

The remaining two blocks derive from it. State entropies marginalize the Mealy emission
over successor states,

```
p_i(k) = Σⱼ transition-p[i][j] · output-p[i][j][k]
```

and take `calculate-entropy` of that; the model entropy is `Σᵢ state-p[i] · H(i)` — the
stationary-weighted mean. Note that `p_i` is a genuine distribution over symbols only
because `Σⱼ transition-p[i][j] = 1` and each `output-p[i][j][·]` sums to 1.

### Defects in `hmm.lsh`

- **`hmm.lsh:186` reads a variable that is not a slot.** `load` does
  `(reading (open-read (concat data-seq-name "_alphabet")) ...)`, but `data-seq-name`
  appears nowhere in the `defclass` — it is a *parameter of the constructor*, rebound at
  line 63 to `<path>.sds/`. Under Lush's dynamic scoping the binding is visible when
  `load` is reached through the constructor at line 90, which is the only call site in the
  tree, so the path resolves to the corpus rather than to the model directory. Every other
  read in `load` uses `dir-name`. Calling `load` directly on an existing model would fail
  or read an unrelated file. **Provenance unknown** — the surviving call path works.
- **`hmm.lsh:136`, `(setqC basename "_trunc")`.** `setqC` is `(setq lhs (concat lhs rhs))`
  (`util.lsh:40-41`), so this appends rather than assigns and the line is correct — noted
  only because it reads as a typo for `setq` until the macro is found.

## 5. `hmm-param` — the working copy (`hmm-param.lsh`)

A near-duplicate of the model's parameter slots plus a back-reference to it, with
`copy-from-model` / `copy-to-model` as the only synchronization. The trainer mutates
`params` and commits to `model` on `keep-model`; `reset-model` copies the other way. This
is the undo mechanism, and it is why the GUI can offer "Keep model" and "Reset model"
buttons against a speculative split.

`update-entropy` (`66-100`) is **verbatim identical** to `hmm.lsh:228-262` except that
`alphabet-size` becomes `:model:alphabet-size`. Thirty-five duplicated lines, including
the stationary solve.

`split-state` (`142-215`) and `merge-states` (`218-369`) do the actual parameter surgery
for topology search — 228 lines. **They live here, not in the trainer.** The trainer's
`try-split` / `try-merge` are wrappers that call into these.

`split-state` adds one state at index `old-size`, halves the initial and inbound
transition mass between the original and its new partner, copies the original's inbound
emissions to the partner, and randomizes every outbound emission fibre with
`rand-p-vector … 0.1`. `merge-states` removes the higher-indexed of the two, remaps
indices through an `old-state-numbers` table, sums inbound transition mass, and takes
**stationary-probability-weighted** averages of the outbound transitions and emissions —
`p-1` and `p-2` at `299-313` are `state-p` values renormalized to sum to one. Both use
`safe-/` throughout, so a merge of two unreachable states yields zeros rather than a
division by zero.

### Defect in both surgery methods

- **`hmm-param.lsh:172` and `:262` seed the new initial distribution from the stationary
  one.** Both read

  ```lisp
  (new-init-state-p i (state-p ...))
  ```

  where the surrounding code is copying `init-state-p`. In `split-state` the very next
  block (`179-180`) *does* use `(init-state-p state-n)`, so a single method disagrees with
  itself four lines apart. The effect is that after any split or merge, every state not
  directly involved has its initial probability overwritten by its stationary probability,
  and the vector is not renormalized afterwards. **Provenance unknown.** It is consistent
  across both methods, which is weak evidence for intent, but the internal inconsistency
  in `split-state` is stronger evidence for a copy-paste slip.

## 6. `update-data-p` — the forward pass

`hmm-trainer.lsh:126-184`. A scaled forward recurrence, Mealy-indexed:

```lisp
(alpha* position-i state-j
        (+ (alpha* position-i state-j)
           (* (alpha* (1- position-i) state-k)
              (transition-p state-k state-j)
              (output-p state-k state-j (d-seq (1- position-i))))))
```

α is initialized to `init-state-p` at position 0, and each column is divided by its own sum
`Q-t[i]` after accumulation. The scale factors are the output: `result-p` accumulates
`safe-add--log2 result-p (Q-t position-i)` over all positions, plus a final term for the
sum of the last column. This is the standard scaled-forward trick — the log-likelihood
falls out of the scale factors, and α itself never underflows.

`data-p` is set to `(* (*factor*) result-p)`, where `*factor*` is a module-level constant
fixed at `1.0` with four commented-out alternatives (`2.0`, `4.0`, `8.0`, `16.0`) directly
above it at `hmm-trainer.lsh:10-15`. Both `*factor*` call sites are marked `;;; ***`.
Whatever experiment that was, it is switched off, and the constant is compiled
(`dhc-make` at `1077`).

**Nothing is stored per position except the scale factors.** α is a `let*` local; the
method's entire persistent output is the scalar `data-p`.

## 7. `update-viterbi-path` — the decode

`hmm-trainer.lsh:188-253`. Two stages plus an annotation pass.

Stage 1 fills `delta` and `psi` with a min-sum recurrence in bits, using `safe-add--log2`
to combine and `safe->--log` to compare, as §3 describes. Stage 2 picks the best final
state by the same comparison and walks `psi` backwards into `:data-seq:path-states`. The
annotation pass writes `:data-seq:path-entropy` from `:params:state-entropies`.

**It reads no forward variable.** Its `let*` binds `init-state-p`, `transition-p`,
`output-p`, `state-entropies`, `d-seq`, `path-states`, `path-entropy`, and allocates
`delta` and `psi` itself. `alpha*` is a local of `update-data-p` and is not reachable from
here; there is no α slot on the class. The two methods are called together by
`update-data` (`116-120`) with a comment saying they "always occur together" and are split
only for readability, but the coupling is scheduling, not data.

The one genuine dependency outside the decode is the annotation: `state-entropies` is
computed by `update-entropy`, which needs the stationary solve of §4. The δ/ψ recurrence
and the backtrace need neither, so the annotation is separable from the decode.

### Two defects in the decode

- **δ is seeded with raw probabilities into a bit-domain accumulator.** Line 216-218:

  ```lisp
  (for (state-j 0 model-size-1)
       (delta 0 state-j
              (init-state-p state-j)))
  ```

  Every other δ value is a description length in bits, but `init-state-p[j]` is a
  probability in `[0,1]`. At position 1 the recurrence computes
  `init_p[k] - log₂(transition·output)` where consistency requires
  `-log₂(init_p[k]) - log₂(transition·output)`.

  Because the comparison prefers *smaller* values, the sign of the preference inverts: a
  state with `init_p = 0.9` contributes `0.9` where it should contribute `0.152`, and one
  with `init_p = 0.01` contributes `0.01` where it should contribute `6.64`. **The decode
  is biased toward improbable start states.** The distortion is bounded by one bit, since
  `init_p ≤ 1`, so it only decides between paths that are within a bit of each other after
  the first transition — but within that band it decides them backwards.

  The degenerate case is worse than the typical one. If `init_p[j]` is exactly `0`, δ
  becomes `0.0`, the best possible value, rather than the `-1` sentinel that means
  impossible. `init-random` never produces an exact zero, but `split-state` halves initial
  probabilities and `merge-states` sums them, and §5 records that both write `state-p`
  values into that slot.

  Line 147-149 of `update-data-p` is the identical line and is correct there, because α is
  a raw probability throughout. This reads as a line copied between two methods that do
  not share a numeric domain. **Provenance unknown.**

- **`psi` is a float matrix holding state indices.** Declared `(float-matrix …)` at line
  198, assigned `state-k` at 234, and used as an index at 247:
  `(psi (1+ position-i) (path-states (1+ position-i)))`. It round-trips integers through
  floats. Harmless for models far below 2²⁴ states, and worth not reproducing. The author
  noted the adjacent subtlety at line 219 — `(psi 0 state-j) is never used, and is in fact
  undefined.`

## 8. The MDL apparatus

Three groups of methods, and the plan's assumption that lines 257-346 hold the M-step is
wrong — they hold this.

**Quantization** (`265-343`). `update-approx-init-state-p`, `-transition-p`, `-output-p`
each round every probability with `round-using(x, d) = round(x·d)/d` and renormalize by
the rounded sum, writing into the trainer's own `init-state-p-r`, `transition-p-r`,
`output-p-r` slots. So `d` is a **quantization resolution**, and the `-r` matrices are the
model as it would actually be transmitted. With the default `d = 10000.0` the grid step is
`1e-4`.

**Data description length** (`346-399`). `update-data-dl` is a near-verbatim clone of
`update-data-p` — the file says so at line 347 — differing only in reading the `-r`
matrices instead of `:params:`. Fifty duplicated lines, including the scaling logic.

**Model description length** (`402-427`):

```lisp
(result-dl (+ (int-code-length n-states-r)
              (int-code-length d)
              (* (1+ n-states-r)
                 (comb-code-length d (1+ n-states)))))
```

plus `n-non-0-transitions × comb-code-length(d, 1+n-symbols)`, where a transition counts
as non-zero **after quantization** — the point of the whole scheme, since rounding at
resolution `1/d` is what drives small transitions to exactly zero and makes a sparse
topology cheaper to describe than a dense one.

The two code-length primitives are standard and worth naming, because they are the part a
port should not re-derive:

- `int-code-length(n)` is Rissanen's universal prior for the integers — `log₂ 2.865` plus
  the iterated logarithm `log₂ n + log₂ log₂ n + …` while the term exceeds 1.
- `comb-code-length(sum, m)` evaluates to `log₂(sum+m) + log₂ C(sum+m−1, m−1)`, the cost
  of coding a composition of `sum` into `m` parts. Called with `sum = d`, it is exactly the
  cost of a probability vector quantized to `1/d` — which is why `d` appears both as the
  rounding grid and as an argument here.

`update-total-dl` (`430-433`) sums data and model DL, mapping the `-1` sentinel to `1e100`
so an impossible model sorts last rather than best.

## 9. `run-add` — Baum-Welch

`hmm-trainer.lsh:483-652`, a single 170-line method, and the real M-step. Its `let*`
allocates the full apparatus: `alpha*`, `beta*`, `Q-t`, `P-t` of shape
`(N+1) × size × size`, and the accumulators `C-in` (`size`), `C-t` (`size × size`) and
`C-t-y` (`size × size × alphabet-size`). It runs `n-cycles` complete EM iterations in one
call, re-estimating in place at `640-649`:

```lisp
(transition-p state-j state-k
              (safe-/ (C-t state-j state-k) out-count-j))
(output-p state-j state-k symbol-l
          (safe-/ (C-t-y state-j state-k symbol-l)
                  (C-t state-j state-k)))
```

The emission update is normalized by the transition count for that *pair*, which is the
Mealy analogue of the usual per-state normalization.

`P-t` is the ξ tensor and it is materialized for every position: `(N+1)·S²` floats. On
`set11a_dInt` at 1,449 symbols a 50-state model needs 3.6 million entries. That is the
memory profile revision 03 has to plan around, and it is the one place in the library
where the flat-stream design (§10) is expensive rather than merely odd.

`run-converge` (`661-674`) is the convergence loop: `run-add 10`, recompute `data-p`,
and count consecutive iterations whose change is below `(* 1e-1 (*factor*))`, stopping
after three. The tolerance carries a commented-out alternative and the author's note —
`Also tried 1e-4, 1e-5, 1e-10.  Even 1e-4 takes forever sometimes.`

## 10. The corpus is one flat stream

`hmm-trainer.lsh:66-68`:

```lisp
(setq data-seq (==> data-source fprop-all))
(==> data-seq set-alphabet :model:alphabet)
(setq data-seq-size (idx-nelements :data-seq:symbol-data))
```

`ACCOUNT.md` §5 records what `fprop-all` does: it concatenates every sequence into one
flat stream, with the `begin`/`end` codes left inline as the only marker of where one
sequence ends and the next begins. **So the trainer has no batch dimension at all.** Every
recurrence in the library runs over a single sequence of length `data-seq-size`, and the
100-sequence `set01z0` corpus and the 1-sequence `set11a_dInt` corpus present to it as the
same kind of object — 582 symbols and 1,449 symbols respectively.

This is the structural fact with the widest consequences for a port. There is no masking,
no padding, no per-sequence likelihood, and no independence between sequences: a
transition from the last state of one sequence to the first state of the next is an
ordinary modelled transition, and the model is free to learn it. Whether that was intended
or merely tolerated is not recoverable from the tree — the `begin`/`end` codes give the
model the *means* to learn a boundary, which is at least consistent with intent.
**Provenance unknown.**

## 11. There is no headless entry point

Both training scripts under `Training/` end the same way:

```lisp
(defvar trainer (new hmm-trainer model dset))
(defvar trainer-w (new HMMtrainerWindow trainer))
(wait trainer-w)
```

`hmm-trainer-view.lsh` is that window and nothing else: 237 lines of `WindowObject`
subclass whose buttons are `Try Split`, `Suggest Split`, `Try Merge`, `Suggest Merge`,
`Suggest Move`, `Add`, `Converge`, `Continue for `, `Keep model`, `Reset model`,
`Update DL`, `Suggest d` and `Update View`. It is presentation only — every button calls a
trainer method and then `update-view` — so it migrates nowhere, but it is the best
available statement of the intended workflow.

**Topology search was driven by hand.** The library offers `suggest-split`,
`suggest-merge` and `suggest-move`, which score candidates, but nothing in the tree loops
over them automatically. A training run was a person watching the description length and
pressing buttons. Anything in revision 04 that reads as "the search strategy" will be a
design decision being made for the first time, not a translation.

`hmm-train-new` also fixes the starting point: `model-starting-size` defaults to **1**. A
run began with a single-state model and grew it by splitting.

## 12. What is compiled

`hmm-trainer.lsh:1075-1100` lists what `dhc-make` compiles, and the omissions are the
informative part:

> `update-params`, `update-data-p`, `update-viterbi-path`, `update-data`, the three
> `update-approx-*`, `update-approx-p`, `update-data-dl`, `update-model-dl`,
> `update-total-dl`, `update-dl`, `update-trainer`, `update-all`, `run-add`,
> `run-converge`, `run-converge-with-update`.

Not compiled: **every topology-search method** — `try-split`, `suggest-split`,
`try-merge`, `suggest-merge`, `suggest-move` — and every persistence method —
`keep-model`, `reset-model`, `save-model`, `save-viterbi-path`, `save-total-dl`,
`suggest-d`, `keep-d`, `reset-d`, `update-training-log`.

`hmm.lsh` compiles only `hmm`, `update-entropy` and `init-random`; `hmm-param.lsh`
compiles all six of its methods including `split-state` and `merge-states`.

The line is drawn exactly where the numeric inner loops are. Search *drives* compiled
work — a split trial calls `run-converge`, which is compiled — so leaving the driver
interpreted costs little. This is the same judgement `ACCOUNT.md` §7 records on the
container side, and it is a useful prior for where ADR 0002's phase 2 will actually pay.

## 13. Duplication, in one place

Three pairs of near-identical code, all involving the recurrences:

| | | Lines |
|---|---|---|
| `hmm.lsh:228-262` | `hmm-param.lsh:66-100` | 35, verbatim but for one symbol |
| `update-data-p` `126-184` | `update-data-dl` `346-399` | ~50, differ only in `:params:` vs `-r` |
| `update-data-p` forward | `run-add` stage 1 `511-560` | ~40, the same recurrence a third time |

The forward recurrence therefore appears **three times** in `hmm-trainer.lsh`, and the
stationary solve **twice** across two files. Any change to the Mealy indexing has to be
made in three places. This is worth recording not as criticism but as a migration hazard:
a port that unifies them is not merely tidier, it is the only version in which the three
copies cannot drift, and the three copies are the most likely place for a translation
error to hide — they are similar enough to skim and different enough to matter.

## 14. What the original does not have

- **No batching, masking or padding** (§10). One sequence, always.
- **No log-domain forward pass.** Scaling is used instead, and only Viterbi works in bits.
  The two halves of the library use different numerical strategies for the same underlying
  problem.
- **No automatic topology search** (§11). Only scored suggestions and a human.
- **No test suite.** `Code/test.lsh` and `Code/test-compile.lsh` are load-and-compile
  smoke scripts, not assertions.
- **No separation between decode and training.** Viterbi is a method on `hmm-trainer`, so
  decoding a sequence requires constructing a trainer, which requires a corpus — see §15.
- **No per-sequence output.** `save-viterbi-path` writes the path over the flat stream.

## 15. The constructor is not separable

`hmm-trainer.lsh:58-90`. The constructor takes `(in-model in-data-source &optional (new? t))`
and branches at line 81:

```lisp
(if (not new?)
    (==> this update-trainer t)
  (==> this update-data-p)
  (==> this update-approx-p t)
  ;; Run Baum-Welch on the initial model before saving it
  (==> this run-converge)
  (==> this suggest-d)
  (==> this keep-d)
  (==> this update-all ())
  (==> this keep-model))
```

**Neither branch is free of the training apparatus.** The default branch runs
`run-converge` — full Baum-Welch — plus the `d` machinery and a model save. The `new? = ()`
branch calls `update-trainer`, which is `update-data` (forward pass *and* Viterbi) followed
by `update-dl` (quantization, data DL, model DL). `Training/hmm-train-new` takes the
default; `Training/hmm-train-load` passes `()`.

Constructing an `hmm-trainer` also requires a `dsource-seq`, since line 66 calls
`fprop-all` on it unconditionally and line 68 measures the result.

So there is no way to obtain the object that owns `update-viterbi-path` without either
training a model or, at minimum, computing a description length over a corpus. A decode-only
use of this library is not expressible in its own terms.

## Appendix: measurements

Both specimens, measured directly rather than inferred.

| | `set01z0_100.sds` | `set11a_dInt.sds` |
|---|---|---|
| `_size` (sequences) | 100 | 1 |
| `_alphabet_size` | 4 | 25 |
| `_seq_size_max` | 20 | 1449 |
| Σ `_seq_sizes` = `data-seq-size` | **582** | **1449** |
| Viterbi trellis cells, `(N+1)·S`, at `S=10` | 5,830 | 14,500 |
| ξ tensor cells, `(N+1)·S²`, at `S=10` | 58,300 | 145,000 |
| ξ tensor cells, `(N+1)·S²`, at `S=50` | 1,457,500 | 3,625,000 |

Emission tensor size is independent of the corpus, being `S²·A`:

| | `A=4` | `A=25` |
|---|---|---|
| `S=1` (the starting size) | 4 | 25 |
| `S=10` | 400 | 2,500 |
| `S=50` | 10,000 | 62,500 |
| Moore equivalent `S·A` at `S=50` | 200 | 1,250 |

Quantization grid at the default `d = 10000.0` is `1e-4`, so a probability below
`5e-5` rounds to zero and drops out of `n-non-0-transitions`.

Definition counts in `hmm-trainer.lsh`: 3 module constants, 1 class, 37 methods. Of the
37, **17 are compiled** and 20 are not (§12).

Corrected region map, against the structural survey the revision plans were drafted from:

| Region | Survey said | Actually |
|---|---|---|
| `21-126` | shared scaffolding | class + a constructor that runs Baum-Welch (§15) |
| `126-188` | forward | forward, `126-184` |
| `188-257` | Viterbi | Viterbi, `188-253` |
| `257-346` | **the M-step** | `update-dl` + parameter quantization (§8) |
| `346-433` | — | data DL (a clone of the forward pass) + model DL |
| `483-652` | — | **Baum-Welch, the actual M-step** (§9) |
| `738-1073` | topology search | topology search *driver* only; 228 lines of parameter surgery are in `hmm-param.lsh:142-369` (§5) |
