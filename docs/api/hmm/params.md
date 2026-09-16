# Parameters

`HMMParams`: the three arc-emission arrays and the vocabulary that gives them meaning, as
one frozen value. See [README.md](README.md) for the contracts, and
[viterbi.md](viterbi.md) for the decode that reads it.

Every example on this page starts from the same model:

```python
import numpy as np
from pfsmgraph.dataseq import USER_BASE, SymbolTable
from pfsmgraph.hmm import HMMParams

vocab = SymbolTable(["a", "b"])

output_p = np.zeros((2, 2, vocab.size))
output_p[0, 0, USER_BASE:] = [0.9, 0.1]
output_p[0, 1, USER_BASE:] = [0.2, 0.8]
output_p[1, 0, USER_BASE:] = [0.5, 0.5]
output_p[1, 1, USER_BASE:] = [0.1, 0.9]

init_state_p = [0.6, 0.4]
transition_p = [[0.7, 0.3], [0.4, 0.6]]

params = HMMParams(init_state_p, transition_p, output_p, vocab)
```

```python
>>> params
HMMParams(n_states=2, n_symbols=8)
```

## `HMMParams`

```python
HMMParams(init_state_p: np.ndarray, transition_p: np.ndarray, output_p: np.ndarray, vocabulary: Vocabulary)
```

| Field | Shape | Meaning |
|---|---|---|
| `init_state_p` | `(S,)` | the initial state distribution |
| `transition_p` | `(S, S)` | `transition_p[i, j]`, the probability of the arc `i → j` |
| `output_p` | `(S, S, A)` | `output_p[i, j, k]`, the probability of emitting code `k` on that arc |
| `vocabulary` | — | the `dataseq` `Vocabulary` the symbol axis was sized against |

Any array-like is accepted and converted to `float64`. `vocabulary` is typed as the
`Vocabulary` protocol rather than `SymbolTable`, and only its `size` is read.

```python
>>> (params.n_states, params.n_symbols)
(2, 8)
```

Both are derived from the array shapes, never stored separately.

### The symbol axis spans the whole vocabulary

`A` must equal `vocabulary.size`, reserved block included:

```python
>>> HMMParams(init_state_p, transition_p, output_p[:, :, USER_BASE:], vocab)
ValueError: output_p's symbol axis is 2 but the vocabulary has 8 codes. The axis spans the whole vocabulary, reserved block included, so a code indexes it directly with no offset; sizing it to the 2 user symbols instead would make an UNK-bearing record index backwards without error
```

The rejected alternative, sizing the axis to the user symbols and subtracting `USER_BASE`
at every index, fails **silently**. `encode(..., on_unknown="unk")` is a documented way to
put `UNK` (code 1) into a record, `1 - USER_BASE` is `-5`, and numpy accepts a negative
index without complaint: the decode would return a confident path built from some other
symbol's probabilities. Sized to the whole vocabulary, the same record reaches a zero and
is reported impossible instead of wrong.

The cost is `6 · S²` entries that are always zero, which is 120 KB at `S = 50`.

The first two axes must match the state count too, and the error says why there are two:

```python
>>> HMMParams(init_state_p, transition_p, output_p[:1], vocab)
ValueError: output_p's first two axes must be (2, 2) -- source and destination state, since emission is on the arc (ADR 0015) -- got (1, 2) from a full shape of (1, 2, 8)
```

### What is validated

Each array must have its stated number of dimensions and hold only finite, non-negative
values. `+inf` means "impossible" in the bit domain the decode works in; a probability
of zero is spelled `0.0`.

Every axis that is a probability distribution must sum to 1, within `1e-5`:

```python
>>> HMMParams([0.6, 0.5], transition_p, output_p, vocab)
ValueError: init_state_p sums to 1.1, not 1 (tolerance 1e-05)
```

The tolerance is wide enough for parameters normalised in `float32` and widened here, and
no wider.

**The reserved fibres must be exactly zero** — not within a tolerance:

```python
>>> bad = output_p.copy()
>>> bad[1, 0, 1] = 0.25
>>> HMMParams(init_state_p, transition_p, bad, vocab)
ValueError: output_p assigns 0.25 to the reserved symbol UNK (code 1) on the arc 1 -> 0, and 1 reserved entries are non-zero. The reserved block must be exactly zero: it is what makes emitting PAD/UNK/BOS/EOS/GAP/MSK impossible by arithmetic -- bits(0) is +inf -- rather than by convention
```

**A row of zeros in `transition_p` is rejected**, with no exemption for unreachable
states:

```python
>>> HMMParams(init_state_p, [[0.7, 0.3], [0.0, 0.0]], output_p, vocab)
ValueError: 1 row(s) of transition_p do not sum to 1: row 1 sums to 0.0. Rows are [1]
```

A state with nowhere to go is not a valid Markov chain, and it is better refused where the
parameters are built than discovered later in a computation that has no answer for it.

**A fibre on a live arc must sum to 1; a fibre on a dead arc is not checked.**

```python
>>> dead = output_p.copy()
>>> dead[0, 1, :] = 0.0
>>> HMMParams(init_state_p, transition_p, dead, vocab)
ValueError: output_p[0, 1] sums to 0.0, not 1, on an arc carrying transition_p[0, 1] = 0.3. 1 such arc(s); fibres on arcs of probability 0 are not checked
>>> HMMParams(init_state_p, [[1.0, 0.0], [0.4, 0.6]], dead, vocab)
HMMParams(n_states=2, n_symbols=8)
```

The same all-zero fibre is an error on an arc of probability 0.3 and accepted on an arc
of probability 0. No path can cross a dead arc, so its emission probabilities cannot
change any result. Models with many dead arcs, which learned topologies have, store zeros
there.

### It is frozen, and so are its arrays

The dataclass is frozen, and every array buffer is read-only as well. A frozen dataclass
alone would stop `params.output_p` being rebound and do nothing about writing through it:

```python
>>> params.output_p[0, 0, 6] = 1.0
ValueError: assignment destination is read-only
```

The arrays are **copied** before being frozen, so changing your own array afterwards does
not reach into the model:

```python
>>> source = np.array([0.6, 0.4])
>>> copied = HMMParams(source, transition_p, output_p, vocab)
>>> source[0] = 0.0
>>> copied.init_state_p
array([0.6, 0.4])
```

To change a model, build a new `HMMParams`. There is no method that mutates one.

### Derived quantities are computed, not stored

Three quantities derive from the arrays. Each is computed on first access, cached, and
returned read-only:

```python
>>> params.state_p
array([0.57142857, 0.42857143])
>>> params.state_entropies
array([0.89317346, 0.82674637])
>>> params.entropy
0.8647047072841829
>>> params.state_p is params.state_p
True
>>> params.state_p.flags.writeable
False
```

- `state_p` — `(S,)`, the stationary distribution of `transition_p`.
- `state_entropies` — `(S,)`, the Shannon entropy in bits of the symbols emitted on
  leaving each state: the marginal `Σⱼ transition_p[i, j] · output_p[i, j, k]`, over `k`.
- `entropy` — `state_entropies` weighted by `state_p`.

Because the value is frozen, a cached result can never describe parameters the object no
longer has.

**`state_p` raises for a reducible chain**, and since it is computed on access, so does
`entropy`. Construction succeeds; the attribute access is where the error surfaces:

```python
>>> split = HMMParams([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]], np.where(np.eye(2)[:, :, None] > 0, output_p, 0.0), vocab)
>>> split
HMMParams(n_states=2, n_symbols=8)
>>> split.state_p
ValueError: transition matrix is reducible: (P.T - I) has a null space of dimension > 1, so the stationary distribution is not unique and replacing one row with the normalization cannot determine it
```

Two states that never leave themselves have no single stationary distribution. The
decode does not read `state_p`, so `viterbi` works on such a model.

### Equality is identity

```python
>>> params == params
True
>>> params == HMMParams(init_state_p, transition_p, output_p, vocab)
False
```

Two models built from the same arrays compare unequal. Comparing arrays has no single right
answer — exactly, or within a tolerance, and if so whose — and nothing in 0.2.0 needs it,
so the question is left open rather than guessed at. Compare the arrays yourself, with the
tolerance your use calls for.
