> **Provenance and status.** This is a **conversation handoff**, not a decision record.
> It was written in an earlier design conversation — the file carries an mtime of
> 2026-06-13 — and moved here from an untracked scratch directory on 2026-09-03, verbatim
> below this block. It predates the `hmm` migration planning, the reading of the Lush
> source, and every ADR after 0012.
>
> **It is authoritative for nothing.** Where it disagrees with an
> [ADR](adr/README.md), the ADR wins ([`adr/README.md`](adr/README.md)); where it
> disagrees with [`docs/plan/TODO.md`](../plan/TODO.md), the plan wins. It is kept
> because it is the **only written source** for two things that exist nowhere else in
> this repository: the arc-emission commitment's *rationale and lineage* (as opposed to
> the imported code's mere behaviour), and the alignment-derived seed described in §1,
> which is a mechanism the PRD asserts the existence of without naming.
>
> **What has since been checked against the source**, by
> [`.scratch/hmm-lush/HMMLIB-ACCOUNT.md`](../../.scratch/hmm-lush/HMMLIB-ACCOUNT.md):
>
> - §1's arc-emission claim — **confirmed** independently. `output-p` is
>   `(size, size, alphabet-size)` and every read is `(output-p state-i state-j symbol-k)`.
>   Recorded as [ADR 0015](adr/0015-arc-emission-mealy-formulation.md).
> - §1's "started from a single-state model" — **confirmed**. The starting size defaults
>   to `1` and there is no headless driver (account §11).
> - §1's premise that the seed's gaps arrive as epsilon emissions — **the machinery does
>   not exist in the source.** A search of `Code/HMMlib/` and `Code/Utility/` for
>   `epsilon|silent|null-|empty-symbol` returns zero hits (2026-09-03). Epsilon removal
>   is new work, not a port.
> - §3's autograd identity — **not independently checked.** It is the same claim as the
>   Eisner 2016 entry in [`references.md`](references.md), stated from recollection in
>   both places, so the two agreeing is one memory twice rather than corroboration.
>
> **What looks like a conflict and is not.** §2.1 recommends building in PyTorch. The
> master plan makes numpy the reference implementation and `torch` an optional backend
> from revision 03. These are answers to different questions: this document argues torch
> is the right *substrate* for the E-step and never addresses whether it should be a
> *required dependency*. The planning branch took up the autograd argument, accepted it,
> and then decided the packaging question on grounds this document does not discuss — a
> ~2GB mandatory dependency, and `core.md`'s invariant that "'GPU' means two unrelated
> things". Do not read §2.1 as reopening that.
>
> **What is drawn from it and recorded elsewhere**, so this file need not be consulted to
> find them: [ADR 0015](adr/0015-arc-emission-mealy-formulation.md) (arc-emission);
> [`docs/plan/DEFERRED.md`](../plan/DEFERRED.md), under the trigger "align able to
> produce a multiple alignment" (the alignment-derived seed); and
> [`references.md`](references.md)
> (the bibliography in §5, entered with what each item is relied on for).
>
> Everything below this block is the original document, unedited.

---

# Project Handoff: Arc-Emission HMM with Topology Search via State Merging

*A working reference distilled from a design conversation. Domain: programmatic modeling of music sequences.*

---

## 1. What you're building

A **hidden Markov model in the arc-emission (Mealy / transducer) formulation** — emissions live on transitions, not states — trained and searched in PyTorch.

- **Conceptual frame.** Newell & Simon problem space: a *state* is a current situation (e.g. the state of the melody) that calls for an action; an *arc* is an operator that emits a musical token and produces the successor state. This maps term-for-term onto a weighted transducer: states = situations, arcs = weighted operators, a derivation = a path, path weight = sequence probability.
- **Training architecture.** Baum-Welch (forward-backward + EM) as the **inner loop**; a **state merging/splitting search over topologies** as the **outer loop**.
- **Prior art (yours).** A previous implementation in **Lush** (LeCun & Bottou's Lisp/compiled-C ML environment). The plan is to recast it into PyTorch, with Claude Code assisting the port.
- **The new idea.** Replace the old brute-force search — which started from a single-state model and was quadratic in model size because merges tried all state pairs — with a **sequence-alignment-derived seed**. A rough (multiple) alignment yields a starting FSM whose gaps appear as **empty-symbol (epsilon) emissions**. Removing those silent transitions up front gives a smaller, better-initialized model for the merge/split search, directly attacking the quadratic cost term.

---

## 2. Key decisions

1. **Build custom in PyTorch rather than adopt a library.**
   - The valuable, novel part — the merge/split topology search — has **no library substitute** (not `hmmlearn`, not `pomegranate`, not OpenGrm).
   - The inner loop (arc-emission forward-backward + EM) is well-understood, numerically bounded tensor computation: PyTorch is the right substrate.
   - State-emission libraries (`hmmlearn`, `pomegranate`) can't represent arc-emission without fighting the model definition.

2. **Use WFST operations sparingly — preprocessing only, never in the inner loop.** The only FSM-algebra step needed is empty-symbol removal on the alignment-derived seed.

3. **Roll your own acyclic weighted epsilon-removal** instead of OpenFst/OpenGrm `RmEpsilon`.
   - Your epsilons come from alignment gaps, which are **strictly time-ordered and therefore acyclic**. (Cycles can be introduced *later* by state merging — but by then the epsilons are already gone.)
   - OpenFst's `RmEpsilon` is parameterized by *semiring*, not by closure logic; custom redistribution isn't exposed without C++ template work, and even less through `pynini`. Not worth the toolchain weight for the acyclic case.

---

## 3. Technical guidance for the implementation

### The autograd shortcut for the E-step
In an autograd framework you usually need not hand-code the backward recursion. The forward/inside pass produces the log-likelihood (log partition function); **the gradient of that scalar with respect to the arc log-weights is exactly the vector of expected arc counts** — the sufficient statistics Baum-Welch's E-step needs. For arc-emission this is especially clean: the arc posterior is precisely ∂ log Z / ∂(arc log-potential). Implement the forward pass in log-space and let `.backward()` give you the expected counts. (This is the Eisner "inside = forward, gradient = expected counts" identity; it halves the code and removes a class of backward-pass bugs.)

### Representation
- Arc-emission pushes naturally toward an **edge-list / sparse** representation rather than a dense N×N transition tensor — and that pays off as topologies grow during splitting.
- Accumulation primitives: `logsumexp` (stability) and `scatter_add` over arc indices; `einsum` for dense sub-blocks.
- **Define-by-run is an asset:** rebuild the transition structure each outer iteration as merge/split changes the state space — no static-graph friction.

### Epsilon removal — the real spec
The operation is **not** "delete the epsilon arcs." It is "remove the silent transitions while *preserving the model's marginal distribution over observable sequences*." A silent arc still carries transition probability and consumes a step; removing it means folding its weight into the epsilon-closure of the surrounding arcs (composing the silent arc's probability into each downstream arc).
- **Acyclic case (yours):** degenerates to a topological-order weight propagation — sort the silent subgraph, push each silent arc's weight onto its successors, splice. ~50 lines, exact.
- **Decide the exactness bar first.** If the merge loop re-estimates all weights via Baum-Welch anyway, you may only need the *topology* and rough weights to be right, which makes the propagation forgiving. If you want a faithful probabilistic object before the first EM round, hold yourself to exact closure.

### Validation
Use `hmmlearn`'s `CategoricalHMM` as a **test oracle**. An arc-emission HMM whose emission distribution depends only on the destination state reduces to a standard state-emission HMM; on that special case you can check your PyTorch forward-backward against `hmmlearn` to machine precision before trusting it under the search.

### Porting caution
The merge/split bookkeeping and control flow port from Lush fairly directly. **Resist a line-by-line transcription of the inner numerical loop** — Lush/C code of that era is loop-heavy in a way that's correct but slow as literal PyTorch. The win is rethinking the inner loop as batched tensor ops over sequences and arcs.

---

## 4. Conceptual clarifications worth keeping straight

- **Arc-emission vs state-emission = Mealy vs Moore.** Lifting a state-emission model to arc-emission is the *easy* direction (replicate each state's emission onto its incoming arcs). The reverse (arc→state) can force state splitting when arcs into the same state carry different emissions. You're going the friendly way.
- **Silent / delete states = epsilon-emitting arcs** in arc-emission terms. The object profile-HMM literature *keeps and handles* is the object you *remove* in preprocessing.
- **The arc↔state equivalence is distributional, not parameter-count-preserving.** Same string distributions (under conditions on final/halting probabilities and determinism), but different parameter counts — which is exactly what a model-selection criterion sees.
- **MDL, two-part vs refined (one-part):**
  - *Two-part code* L(M)+L(D|M): parameter counting lives in L(M); the (k/2)·log n term is BIC.
  - *Refined / one-part (NML, stochastic complexity):* codelength = −log P(D | θ̂) + COMP(M), where the parametric-complexity term counts *distinguishable distributions* (Fisher-information / Jeffreys volume), not raw parameters. Two equal-parameter models can differ here.
  - **Caveat:** exact NML for HMM/PFSM classes is intractable (exponential sum over all data sequences). Practical routes: BIC/MAP surrogates, or **factorized/sequential NML** built on the tractable **multinomial NML** (each transition/emission distribution is a multinomial).

---

## 5. Annotated bibliography

### A. Profile HMMs & alignment → model construction
- **Durbin, Eddy, Krogh & Mitchison (1998).** *Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids.* Cambridge University Press. ISBN 0-521-62971-3 (pbk).
  - Ch. 3: Markov chains and HMM fundamentals (forward/backward, Viterbi, Baum-Welch).
  - **Ch. 5: profile HMMs** — match/insert/delete architecture; delete states are the *silent* (epsilon) states; silent-state-aware forward-backward with within-column topological ordering. The canonical written treatment of "alignment with gaps → HMM with silent states." (State-emission; translate on the fly.)
  - Ch. 6: multiple alignment methods.
- **Krogh, Brown, Mian, Sjölander & Haussler (1994).** "Hidden Markov Models in Computational Biology: Applications to Protein Modeling." *J. Mol. Biol.* 235(5):1501–1531. The primary source that introduced the profile-HMM architecture the book later systematized.

### B. Arc-emission lineage (IBM / information-theoretic)
- **Jelinek (1997).** *Statistical Methods for Speech Recognition.* MIT Press (Language, Speech, and Communication). The book-length treatment using arc-emission throughout, including Baum-Welch in arc-emission form — the version to check your Lush math against.
- **Bahl, Jelinek & Mercer (1983).** "A Maximum Likelihood Approach to Continuous Speech Recognition." *IEEE Trans. PAMI* PAMI-5(2):179–190. A prominent landmark for the transition-output ("Markov source") formulation — methodology paper, not a manifesto. (Arc-emission here is downstream of the source-channel / information-theory worldview, not a contrarian stance; the state-emission "norm" largely crystallized later, with Rabiner's 1989 tutorial.)
- **Manning & Schütze.** *Foundations of Statistical Natural Language Processing.* MIT Press. Its HMM chapter presents **both** arc-emission and state-emission and discusses their equivalence — likely where the terms were first encountered.

### C. Arc ↔ state equivalence and translation
- **Dupont, Denis & Esposito (2005).** "Links between probabilistic automata and hidden Markov models: probability distributions, learning models and induction algorithms." *Pattern Recognition* 38(9):1349–1371. **The most on-point reference.** Clarifies PA↔HMM links; necessary/sufficient conditions for an automaton to define a probabilistic language; PDFA proven a proper subclass of PNFA; equivalence holds under stated conditions on final probabilities.
- **Vidal, Thollard, de la Higuera, Casacuberta & Carrasco (2005).** "Probabilistic Finite-State Machines — Part I & Part II." *IEEE Trans. PAMI* 27(7):1013–1025 (Part I) and 1026–1039 (Part II). Part II studies relations among PFA, HMMs, and n-grams with theorems and algorithms; Part I surveys the objects, topology, consistency, and equivalence.
- **Mohri, Pereira & Riley (2002).** "Weighted Finite-State Transducers in Speech Recognition." *Computer Speech & Language* 16(1):69–88. The WFST machinery; **weight pushing** is the operation that redistributes weight mass along paths — the concrete mechanism for moving between state-located and arc-located mass.
- **Droste, Kuich & Vogler, eds. (2009).** *Handbook of Weighted Automata.* Springer. Standard reference for the semiring foundations.
- **Sakarovitch (2009).** *Elements of Automata Theory.* Cambridge University Press. Comprehensive modern treatment of weighted automata and transducers.
- **Berstel & Reutenauer (2011).** *Noncommutative Rational Series with Applications.* Cambridge University Press (expanded successor to *Rational Series and Their Languages*, 1988). Schützenberger theory: a weighted automaton is (initial vector, per-symbol transition-weight matrices, final vector) — the level at which "emission on states vs. arcs" dissolves into matrix structure. Likely the most satisfying altitude given an MDL/coding mindset.
- **Hopcroft, Motwani & Ullman.** *Introduction to Automata Theory, Languages, and Computation.* The classical Mealy/Moore equivalence — unweighted ancestor of the whole question.

### D. MDL applied to probabilistic FSMs
- **Grünwald (2007).** *The Minimum Description Length Principle.* MIT Press. The comprehensive textbook; develops two-part → refined → NML with the Fisher-information / differential-geometric interpretation.
- **Grünwald (2004).** "A Tutorial Introduction to the Minimum Description Length Principle." arXiv:math/0406077. (Also the lead chapter of Grünwald, Myung & Pitt, eds., *Advances in Minimum Description Length: Theory and Applications*, MIT Press, 2005.) The most efficient on-ramp — same arc in ~80 pages.
- **Rissanen (1996).** "Fisher Information and Stochastic Complexity." *IEEE Trans. Information Theory* 42(1):40–47. Where refined MDL / NML crystallizes. (Origin: Rissanen, "Modeling by Shortest Data Description," *Automatica* 14:465–471, 1978. Late synthesis: *Information and Complexity in Statistical Modeling*, Springer, 2007.)
- **Stolcke & Omohundro (1992/93).** "Hidden Markov Model Induction by Bayesian Model Merging." *NIPS 5*, pp. 11–18. Fuller version: "Best-first Model Merging for Hidden Markov Model Induction," ICSI TR-94-003, 1994; developed in Stolcke's 1994 Berkeley PhD thesis. **Closest match to your method** — ML model encoding the data → merge states → stop on a complexity-penalized criterion. Uses a Bayesian MAP score (≈ two-part code) and explicitly punts on full parameter integration because it's infeasible for HMMs — a concrete demonstration of the refined-MDL intractability.
- **Thollard, Dupont & de la Higuera (2000).** "Probabilistic DFA Inference using Kullback-Leibler Divergence and Minimality" (the MDI algorithm). *Proc. 17th ICML.* A description-length-flavored state-merging criterion for probabilistic DFA.
- **Kontkanen & Myllymäki (2007).** "A Linear-Time Algorithm for Computing the Multinomial Stochastic Complexity." *Information Processing Letters.* Tractable multinomial NML — the building block for factorized/sequential NML over Markov sources, i.e. the practical way to get a refined one-part criterion into a merge loop.
- **de la Higuera (2010).** *Grammatical Inference: Learning Automata and Grammars.* Cambridge University Press. Puts probabilistic-automata learning, state merging, and MDL/Bayesian criteria side by side. **Recommended desk book** — serves both the equivalence question and the search method.

### E. Tooling (only if WFST algebra is needed later)
- **`pynini`** (conda-forge: `conda install -c conda-forge pynini`, or `pip install pynini`). OpenGrm Python wrapper for building/manipulating WFSTs. The conda route also installs OpenFst CLI tools; the pip wheel does not.
- **OpenGrm BaumWelch.** Arc-emission Baum-Welch over weighted transducers; **log semiring → true Baum-Welch**, **tropical semiring → Viterbi training**. Built from source (OpenFst with `--enable-grm`, C++17); training driven largely via its `baumwelch` CLI.
- **`hmmlearn` (`CategoricalHMM`).** State-emission discrete HMM; use as a validation oracle (see §3).

---

## 6. Open questions / next steps

1. **Exactness bar for epsilon-removal** — rough seed vs. faithful probabilistic object, given the merge loop re-estimates weights downstream (§3).
2. **Merge criterion** — BIC/MAP surrogate (à la Stolcke & Omohundro) vs. factorized/sequential NML built on multinomial NML. The latter is the principled one-part route; the former is what the literature actually shipped.
3. **Port plan** — merge/split control flow transcribes directly; budget real time to re-express the inner numerical loop as batched tensor ops rather than porting it literally.
4. **Sequencing for the MDL reading** — Grünwald tutorial (conceptual shift) → Stolcke thesis (criterion grafted onto your exact merge loop, and *why* he stopped short of the refined version) → fNML / multinomial-NML thread (if you want Rissanen's one-part code in the loop).

---

*Prepared as a conversation handoff. All references were checked during the discussion; verify page-level details against the originals before citing in writing.*
