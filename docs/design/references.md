# References

External work this project's design depends on, or that a decision recorded in an
[ADR](adr/README.md) leans on. An entry belongs here when a reader would otherwise have to
take a claim in our own documentation on trust.

Each entry says **what it is relied on for**, since that is the part a citation alone does
not carry — and it is what makes a stale or misremembered reference detectable.

---

## Hidden Markov models

- **Eisner, Jason (2016). "Inside-Outside and Forward-Backward Algorithms Are Just
  Backprop (Tutorial Paper)."** *Proceedings of the Workshop on Structured Prediction for
  NLP*, pages 1–17, Austin, TX, November 2016. ACL Anthology
  [`W16-5901`](https://aclanthology.org/W16-5901/) ·
  [PDF](https://aclanthology.org/W16-5901.pdf)

  **Relied on for:** the claim that reverse-mode automatic differentiation through the
  forward pass *computes the backward algorithm*, so that the Baum-Welch E-step can be
  read off `.grad` rather than written by hand. Concretely: for log-parameters,
  `∂ log P(O|θ) / ∂ log a_ij` is the expected transition count `Σ_t ξ_t(i,j)`, and the
  M-step is then a row normalisation. The same identity gives the Viterbi backtrace as the
  subgradient of a max-plus forward pass, with no backpointers.

  This is the basis of revision `03-hmm-v0.2.0`'s decision to hold an optional `torch`
  autograd backend against the numpy reference's explicit forward-backward as an
  [ADR 0003](adr/0003-one-parameterized-test-suite-per-algorithm.md) equivalence test.
  The two implementations share no code, which is what makes their agreement worth
  something.

  *Bibliographic details verified 2026-09-03 against the ACL Anthology. The **use** made
  of the paper above has not been checked against its text — it is stated from
  recollection, and revision 03 should confirm it before the equivalence test is written
  on its authority.*

- **Jelinek, Frederick (1997). *Statistical Methods for Speech Recognition.*** MIT Press,
  Language, Speech, and Communication series.

  **Relied on for:** a book-length treatment of HMMs in the **arc-emission** formulation
  throughout, including Baum-Welch in arc-emission form. This is the reference against
  which the migrated recurrences should be checked, because it is the only one on this
  list that states the algorithms in the shape
  [ADR 0015](adr/0015-arc-emission-mealy-formulation.md) commits to — every other
  presentation is state-emission and has to be translated before it can be compared.

  Also relied on for the lineage claim in ADR 0015's Context: arc-emission belongs to the
  IBM source-channel tradition rather than being a departure from a settled norm. The
  earlier landmark for the transition-output ("Markov source") formulation is **Bahl,
  Jelinek & Mercer (1983), "A Maximum Likelihood Approach to Continuous Speech
  Recognition," *IEEE Trans. PAMI* PAMI-5(2):179–190**; the state-emission presentation
  became the default later, with Rabiner's 1989 tutorial.

## Probabilistic automata and HMM equivalence

- **Dupont, Pierre; Denis, François; Esposito, Yann (2005). "Links between probabilistic
  automata and hidden Markov models: probability distributions, learning models and
  induction algorithms."** *Pattern Recognition* 38(9):1349–1371.

  **Relied on for:** the two claims [ADR 0015](adr/0015-arc-emission-mealy-formulation.md)
  makes about the arc ↔ state correspondence and cannot check from the imported source.
  First, that the equivalence is **distributional and holds under stated conditions** on
  final/halting probabilities, rather than being an unconditional identity. Second, that
  it is **not parameter-count-preserving** — which is the load-bearing half, because a
  description-length criterion sees parameter counts directly, so an MDL score computed
  for a state-emission HMM is not a baseline for an arc-emission one of the "same" model.

  Revision `04-hmm-v0.3.0` must confirm this against the paper before publishing any MDL
  comparison that leans on it.

- **Vidal, E.; Thollard, F.; de la Higuera, C.; Casacuberta, F.; Carrasco, R. C. (2005).
  "Probabilistic Finite-State Machines — Part I and Part II."** *IEEE Trans. PAMI*
  27(7):1013–1025 and 1026–1039.

  **Relied on for:** corroboration of the above, at survey depth. Part II studies the
  relations among probabilistic finite automata, HMMs and n-grams with theorems and
  algorithms; Part I covers the objects themselves, their topology and consistency
  conditions. Listed as the second opinion rather than the primary source — where the two
  disagree, Dupont et al. is the more on-point.

*Bibliographic details for this section were taken from
[`arc-emission-hmm-handoff.md`](arc-emission-hmm-handoff.md) §5C, which records them as
checked during the conversation that produced it, and were **not** re-verified here. The
uses made of the works are stated from recollection. Both should be confirmed before
either is cited in writing intended for publication.*

## Model selection and description length

The criterion that scores revision `04-hmm-v0.3.0`'s topology search. Which of these the
project ends up using is an open question — [PRD §8](PRD.md), *"Which description length
scores the topology search"*.

- **Grünwald, Peter D. (2007). *The Minimum Description Length Principle.*** MIT Press.
  With the shorter **Grünwald (2004), "A Tutorial Introduction to the Minimum Description
  Length Principle," arXiv:[math/0406077](https://arxiv.org/abs/math/0406077)** — also the
  lead chapter of Grünwald, Myung & Pitt, eds., *Advances in Minimum Description Length*,
  MIT Press, 2005.

  **Relied on for:** the distinction the open question turns on — **two-part** codes
  L(M) + L(D|M), where model cost counts parameters and the (k/2)·log n term is BIC,
  versus **refined / one-part** codes, where the penalty is a parametric-complexity term
  counting *distinguishable distributions* by Fisher-information volume. The tutorial is
  the efficient on-ramp; the book is where the differential-geometric reading is
  developed.

- **Rissanen, Jorma (1996). "Fisher Information and Stochastic Complexity."** *IEEE Trans.
  Information Theory* 42(1):40–47.

  **Relied on for:** the refined-MDL / NML formulation itself, `−log P(D | θ̂) + COMP(M)`.
  Origin of the principle: **Rissanen (1978), "Modeling by Shortest Data Description,"
  *Automatica* 14:465–471**; late synthesis: *Information and Complexity in Statistical
  Modeling*, Springer, 2007. The imported Lush code's `int-code-length` is a universal
  code for integers in this lineage — revision 04's first subgoal is to establish *which*
  one, since the search behaviour depends on it.

- **Stolcke, Andreas & Omohundro, Stephen (1992/93). "Hidden Markov Model Induction by
  Bayesian Model Merging."** *NIPS 5*, pp. 11–18. Fuller version: "Best-first Model
  Merging for Hidden Markov Model Induction," ICSI TR-94-003, 1994; developed in Stolcke's
  1994 Berkeley PhD thesis.

  **Relied on for:** two things. First, that this project's method has a **published
  relative** — build a model that encodes the data, merge states, stop on a
  complexity-penalised criterion — so revision 04 is reproducing a known family of
  algorithm rather than inventing one. Second, and more usefully, for the claim in PRD §8
  that the refined criterion is *not simply available*: they used a Bayesian MAP score
  (≈ a two-part code) and explicitly declined full parameter integration as infeasible for
  HMMs. That is a concrete demonstration of the intractability, not our inference from it.

- **Kontkanen, Petri & Myllymäki, Petri (2007). "A Linear-Time Algorithm for Computing the
  Multinomial Stochastic Complexity."** *Information Processing Letters* 103(6):227–233.

  **Relied on for:** the existence of a **tractable** route to a refined code. Exact NML
  over an HMM class requires a sum over all possible data sequences and is out of reach;
  factorised/sequential NML sidesteps this by composing the multinomial case, which
  applies here because every transition and emission distribution in the model is a
  multinomial. This is the reference that makes the open question answerable in principle
  rather than merely well-posed.

- **de la Higuera, Colin (2010). *Grammatical Inference: Learning Automata and Grammars.***
  Cambridge University Press.

  **Relied on for:** nothing specific — listed as the desk reference that puts
  probabilistic-automata learning, state merging, and MDL/Bayesian criteria side by side,
  and so spans both this section and the equivalence one above. The state-merging
  criterion with a description-length flavour is **Thollard, Dupont & de la Higuera
  (2000), "Probabilistic DFA Inference using Kullback-Leibler Divergence and Minimality"
  (the MDI algorithm), *Proc. 17th ICML***.

*Bibliographic details for this section were taken from
[`arc-emission-hmm-handoff.md`](arc-emission-hmm-handoff.md) §5D, which records them as
checked during the conversation that produced it, and were **not** re-verified here. The
uses made of the works are stated from recollection.*

## Numerical methods

- **Press, W. H. et al. *Numerical Recipes in C*.** Not a dependency, but the acknowledged
  source of several routines in the imported Lush tree: `LU-decomposition`,
  `LU-back-substitution` and the Brent minimiser in
  `.scratch/hmm-lush/Code/Utility/util.lsh` are transcriptions, and say so in their own
  comments. Recorded here because the migration replaces them with library calls rather
  than translating them, and the provenance is the reason that is safe.
