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

## Numerical methods

- **Press, W. H. et al. *Numerical Recipes in C*.** Not a dependency, but the acknowledged
  source of several routines in the imported Lush tree: `LU-decomposition`,
  `LU-back-substitution` and the Brent minimiser in
  `.scratch/hmm-lush/Code/Utility/util.lsh` are transcriptions, and say so in their own
  comments. Recorded here because the migration replaces them with library calls rather
  than translating them, and the provenance is the reason that is safe.
