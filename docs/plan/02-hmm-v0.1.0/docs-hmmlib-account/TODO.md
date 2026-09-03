# docs/hmmlib-account

**Status**: merged — PR #13 — 2026-09-03
**Created**: 2026-09-03
**Subgoal**: Read `Code/HMMlib/` in its own terms and write `.scratch/hmm-lush/HMMLIB-ACCOUNT.md`, following `ACCOUNT.md`'s conventions — measurements against the two tracked specimen corpora, and **provenance unknown** for behaviours the code admits but may never have exercised. Check the three falsifiers and revise the plan if any holds (revision `02-hmm-v0.1.0`)

Markers: `[ ]` not started · `[~]` in progress · `[x]` complete · `[!]` blocked · `[-]` deferred

**The falsifier check is not a formality.** Revision 02's boundaries, and 03's and 04's
behind them, were drawn from a structural survey rather than from the source. Goal 2 is the
first time anything in this repository can contradict them, so a held falsifier is the
method working. Record the negative verdicts with the same care as a positive one: "checked,
does not hold" is a finding, while silence is indistinguishable from not having looked.

## Goals

- [x] Read the four `HMMlib` files and write `HMMLIB-ACCOUNT.md`
  - [x] Read `hmm.lsh` (319) and `hmm-param.lsh` (386) — the model and its parameters — before the trainer, so the trainer is read against a model that is already understood
  - [x] Read `hmm-trainer.lsh` (1102), noting which of its regions belong to which revision: `21-126` scaffolding, `126-188` forward, `188-257` Viterbi, `257-346` M-step, `738-1073` topology search
  - [x] Read `hmm-trainer-view.lsh` (237) far enough to say whether it is presentation only, and therefore whether it migrates at all
  - [x] Follow `HMMlib`'s calls into `Code/Utility/` only as far as they go; the migration itself is subgoal 3, not this branch
  - [x] Write the account to `ACCOUNT.md`'s conventions — **Sources** block with line counts and dates, structure before behaviour, an appendix collecting every measurement, and **provenance unknown** where the code admits a behaviour that may never have run
  > **Done:** `.scratch/hmm-lush/HMMLIB-ACCOUNT.md`, 600 lines, 15 sections and an
  > appendix. Tracked-file check done first, not after: `git check-ignore -v` named
  > `.gitignore:42`'s `!/*.md`, and `git status` shows the file as `??` rather than
  > swallowing it. Compile dates for the **Sources** block came from `Code/HMMlib/C/`,
  > whose mtimes the import did not reset -- `hmm.c` and `hmm_param.c` 2009-07-20,
  > `hmm_trainer.c` 2011-02-01 -- so the block dates the last *compile* of each source
  > rather than claiming an edit date it cannot evidence.
  > **Found:** the model is **Mealy**, not Moore. `output-p` is
  > `(size, size, alphabet-size)` and every read is `(output-p state-i state-j symbol-k)`:
  > symbols are emitted on transitions, not in states. Nothing in the PRD, the ADRs or any
  > of the three revision plans anticipates this. It is the same design decision as
  > `seq-state`'s `+1` that `ACCOUNT.md` §6 already documented -- a path emitting *N*
  > symbols visits *N+1* states -- seen from the other side. Two consequences reach the
  > kernel: the emission factor cannot be hoisted out of the inner loop, since it depends
  > on both endpoints; and the emission tensor is `S²·A` rather than `S·A`, which at
  > `set11a_dInt`'s alphabet of 25 is 62,500 parameters for a 50-state model against 1,250
  > for the Moore equivalent.
  > **Found:** the arithmetic is **description length in bits**, not probability.
  > `safe-add--log2` accumulates `sum - log2(x)` and `-1` is an absorbing log-zero
  > sentinel, unreachable because a real DL is non-negative. `safe->--log x y` reads
  > backwards until the sentinel handling is worked through: it means "*y* beats *x*".
  > So Viterbi is **min-sum over bits** and a port reaching for `max` inverts every
  > comparison. The MDL framing is not a layer above the kernel; it is the kernel's
  > number system. Corollary trap: the slot `data-p` and the local `result-p` hold DLs
  > despite the `-p` suffix that means "probability" everywhere else in the library.
  > **Found:** three defects, all **provenance unknown**. (1) `update-viterbi-path:216-218`
  > seeds δ with raw `init-state-p` into that bit-domain accumulator; since smaller is
  > better the preference *inverts*, biasing the decode toward improbable start states,
  > and an exactly-zero initial probability becomes the best possible value rather than
  > the sentinel. The identical line in `update-data-p` is correct, because α is a raw
  > probability throughout -- it reads as a line copied between methods that do not share
  > a numeric domain. (2) `hmm.lsh:186` reads `data-seq-name`, which is not a slot; it
  > resolves only through Lush's dynamic scoping from the one surviving call path.
  > (3) `hmm-param.lsh:172` and `:262` fill the new *initial* distribution from the
  > *stationary* one, while `split-state` uses `init-state-p` four lines later --
  > one method disagreeing with itself.
  > **Found:** the trainer has **no batch dimension**. `fprop-all` flattens the corpus to
  > one stream and `data-seq-size` is its element count, so the 100-sequence and
  > 1-sequence specimens are the same kind of object to it -- no masking, no padding, no
  > per-sequence likelihood, and cross-sequence transitions are ordinary modelled
  > transitions. And there is **no headless entry point**: both `Training/` scripts end in
  > `(new HMMtrainerWindow trainer)`, so topology search was driven by hand from a GUI and
  > anything in revision 04 that reads as "the search strategy" is a decision being made
  > for the first time, not a translation.
  > **Commit:** the five sub-items were one continuous reading, so they were taken as a
  > unit rather than pausing for a commit checkpoint between each.
  > **Correction:** this sub-item's own region note above — "`257-346` M-step" — is wrong,
  > found while checking falsifier 3 in goal 2. `HMMLIB-ACCOUNT.md` §8-9 establish `257-346`
  > as parameter quantization (`update-approx-*`) plus `update-dl`, not expected-count
  > accumulation; the real M-step is `run-add` at `483-652`. The account's own "Corrected
  > region map" appendix table already carries this; left uncorrected here until now
  > because nothing had re-read this line against it. Not rewriting the line above —
  > recording the correction instead, the way the account narrates its own corrections.

- [x] Check the three falsifiers the master plan names, and record each verdict
  - [x] **Does Viterbi read the forward variables?** The 02/03 split assumes `update-viterbi-path` (`hmm-trainer.lsh:188-257`) computes δ independently of the α that `update-data-p` (`126-188`) builds. If it reads α, the boundary moves and Viterbi drags the forward pass into 02 with it
    > **Verdict:** Does not hold. `update-viterbi-path` reads no forward variable — `alpha*` is a local of `update-data-p` alone and there is no α slot on the class; the two methods are scheduled together by `update-data` only for readability, per the source's own comment, not a data dependency (`HMMLIB-ACCOUNT.md` §7). The 02/03 boundary as drawn stands.
  - [x] **Is the stationary-distribution solve what it looks like?** `hmm-param.lsh:82` and `hmm.lsh:244` build a matrix from `int-delta` and call `LU-solve`; that reads as `(I - Pᵀ)π = 0`, inferred from two lines of context
    > **Verdict:** Does not hold. The solve is `(Pᵀ - I)π = 0` with the first row replaced by `Σπ = 1` (`HMMLIB-ACCOUNT.md` §4) — the guessed sign is flipped from the derived one, but `(Pᵀ - I)` and `(I - Pᵀ)` share the same null space, so the port is unaffected.
  - [x] **Is `hmm-trainer.lsh:21-126` separable?** The scaffolding is assumed shareable across 02 and 03. If the constructor demands the training apparatus, 02 gets no trainer at all and Viterbi becomes a free function over a model — which may be the better design regardless
    > **Verdict:** Holds. Neither constructor branch is free of the training apparatus: the default branch runs full Baum-Welch (`run-converge`) plus the `d` machinery and a model save before returning; the other runs the forward pass, Viterbi and a description-length computation. Both require a corpus unconditionally (`fprop-all`, line 66). "A decode-only use of this library is not expressible in its own terms" (`HMMLIB-ACCOUNT.md` §15). `21-126` is not shareable scaffolding — it is the class plus a constructor that trains. This confirms rather than merely assumes revision 02 subgoal 2's premise ("what Viterbi is a method *on* given there is no trainer object in this release").
  - [x] Revise `docs/plan/TODO.md` for any falsifier that holds, and amend the affected `planned/` draft if the consequence reaches 03 or 04
    > **Done:** Appended the three verdicts above, in place, to revision 02's falsifier bullets in `docs/plan/TODO.md`. Falsifier 3's consequence stays inside revision 02 — 03's draft already treats "the trainer" as something *it* introduces (e.g. "Batch the trainer over sequences"), so nothing there needed amending on this account. A second, unrelated correction surfaced while reading §8-9 for this check: the region map both this branch's own goal-1 annotation and `03-hmm-v0.2.0.md`'s falsifier text inherited from the structural survey — "`257-346` M-step" — is wrong; `HMMLIB-ACCOUNT.md` §8-9 (and its own "Corrected region map" appendix table) establish `257-346` as quantization + `update-dl`, and the real M-step is `run-add` at `483-652`. Corrected `03-hmm-v0.2.0.md`'s citations; see the correction note under goal 1 below for this branch's own copy of the mislabeling.

- [x] Record what the account changes for the subgoals downstream of it
  - [x] Note anything subgoal 2 (the public surface) now has evidence for that it previously had only a survey of
    > **Verdict:** Two constraints, added to subgoal 2 in place. Lush's model reads its
    > alphabet directly out of the corpus's `.sds` directory rather than taking a
    > vocabulary object (`HMMLIB-ACCOUNT.md` §4) — the file-coupled seam `ACCOUNT.md` §1
    > already found on the container side, not to be reproduced; take a `SymbolTable`
    > explicitly instead. And Lush's Viterbi always decodes one sequence, since batching
    > belongs to a trainer that does not exist in 0.1.0 — so `pad_collate` is plausibly out
    > of scope for this subgoal entirely, deferred to revision 03's batched training.
  - [x] Note anything subgoal 3 (the `Utility` migration) should expect, without doing it here
    > **Verdict:** One naming gap and one implementation gotcha, added to subgoal 3 in
    > place. `safe-add--log2` — the log₂-domain accumulator Viterbi's inner loop actually
    > calls, and home of the `-1` sentinel (§3) — was previously named only by allusion
    > ("the log-zero sentinel `safe->--log` implies"), naming the comparator but not the
    > accumulator; now named directly. And the stationary solve needs its row-replacement
    > trick reproduced, not just its numerical result: `(Pᵀ - I)π = 0` is singular by
    > construction, so handing the homogeneous system as stated to `numpy.linalg.solve`
    > fails outright (§4).
  > **Done:** Beyond this goal's two named sub-items, one more finding from goal 1's own
  > reading surfaced while re-checking the account for this goal and was added, with the
  > user's explicit go-ahead, to the "Implement Viterbi at ADR 0002 phase 1" subgoal rather
  > than left to be rediscovered there: `update-viterbi-path` seeds δ with a raw probability
  > into the bit-domain accumulator (§7), inverting the start-state preference and turning
  > an exact zero into the *best* possible δ rather than impossible. That subgoal now
  > requires deciding, and recording, whether the bug is fixed or faithfully reproduced.
