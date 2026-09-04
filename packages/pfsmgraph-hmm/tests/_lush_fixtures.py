"""Reading the tracked Lush `.hmm` model directories as differential fixtures.

Each saved model holds the **inputs and the outputs** of a computation being
ported -- `transition_p` beside the `state_p` the original solved from it,
`output_p` beside the `state_entropies` it marginalized -- so a test can check a
translation against numbers the original itself produced rather than against
numbers we chose.

This module exists because a second consumer arrived. `test_numeric.py` carried
the reader with a docstring naming its own trigger: "When revision 03's
differential tests want it too, that is the moment a shared fixture earns
itself." The condition was right and the timing was off by a revision -- the
second consumer is `test_params.py`, in revision 02. Extracting now rather than
copying is the project's own finding applied to itself:
`HMMLIB-ACCOUNT.md` section 13 names near-duplicate code as the migration's most
likely hiding place for a translation error, "similar enough to skim and
different enough to matter."

Reading `.scratch/` in place is not the `.notebooks/`/`.data/` prohibition being
bent. Those are barred because their contents exist on one machine only, which
is exactly what tracking removes; these files are tracked. Nothing under
`packages/*/src/` reaches them, and every member's wheel packages only
`src/pfsmgraph`, so "the fixture is absent in an installed wheel" is not a
scenario this repo has.

**The prints are four decimals, and that is a trap rather than a detail.** A
saved `transition_p` row sums to 1 +/- 1e-4, which is enough to destroy exact
singularity and enough to fail `HMMParams`'s `SUM_TOL`. :func:`load_params`
repairs it by renormalizing -- restoring the hypothesis -- rather than by
widening a tolerance, which would only loosen a conclusion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pfsmgraph.dataseq import USER_BASE, SequenceRecord, SymbolTable

from pfsmgraph.hmm import HMMParams

#: The tracked corpus of saved models. Four levels up from this file is the repo
#: root: tests/ -> pfsmgraph-hmm/ -> packages/ -> root.
FIXTURES = (
    Path(__file__).resolve().parents[3]
    / ".scratch"
    / "hmm-lush"
    / "Training"
    / "set02a"
    / "set02a_200"
)

#: The three tracked models: two 5-state and one 8-state.
SAVED_MODELS = ("m001_0001_001.hmm", "m001_0005_005.hmm", "m008_0001_008.hmm")

#: The corpus the three models were trained on, tracked beside them. Holds the
#: `.seq` files whose concatenation the saved decodes are aligned to, and the
#: `_alphabet` naming the codes -- which the models themselves cannot.
CORPUS = FIXTURES.parent / "set02a_200.sds"


def read_ascii_matrix(path):
    """Read Lush's `save-ascii-matrix` format: `.MAT <ndim> <dims...>`, then values."""
    tokens = path.read_text().split()
    if tokens[0] != ".MAT":
        raise ValueError(f"{path} is not a Lush ASCII matrix")
    ndim = int(tokens[1])
    dims = [int(t) for t in tokens[2 : 2 + ndim]]
    values = np.array([float(t) for t in tokens[2 + ndim :]], dtype=np.float64)
    return values.reshape(dims)


def read_scalar(path):
    """Read a saved scalar slot -- `entropy`, `_total_dl` -- which carries no header."""
    return float(path.read_text().strip())


def load_params(directory) -> HMMParams:
    """Build an :class:`HMMParams` from one saved model directory.

    Two repairs stand between the saved arrays and a constructible parameter set,
    and both are properties of the save format rather than of the model.

    **The symbol axis is renumbered onto the reserved block.** Lush's alphabet
    starts its user symbols at code 2; ADR 0011 puts them at 6. The saved
    `output_p` is therefore `(S, S, alphabet_size)` where ours is
    `(S, S, USER_BASE + alphabet_size)`, with the fixture's fibres placed at
    `[..., USER_BASE:]` and the reserved block left zero. Every derived quantity
    is invariant under this: `state_p` never reads `output_p` at all, and
    zero-padding a symbol axis adds only `0 * log2(1) = 0` terms to an entropy.

    **The four-decimal prints are renormalized.** Rows sum to 1 +/- 1e-4, which
    `SUM_TOL` (1e-5) rejects. Only *live* arcs -- `transition_p[i, j] > 0` -- have
    their emission fibres renormalized; a dead arc's fibre is all zeros in these
    files and stays that way, which is exactly the case `HMMParams` exempts.

    The saved `_alphabet` is not read, because it cannot be: Lush serialized its
    symbols as pointer addresses (`#$11F50E0`), so a saved *model* does not carry
    its symbol names. They are not lost -- :func:`corpus_alphabet` reads them
    from the corpus that trained it -- but they are not here, and rejoining the
    two is exactly ADR 0001's cost clause: persistence split the mapping from the
    model, so the model alone stops denoting. Only the codes matter for this
    function, so the names are synthesized below.

    **Lush's `begin` and `end` become user symbols, not BOS/EOS**, and the
    obvious alternative fails loudly rather than quietly. Lush codes 0 and 1 are
    the corpus's own boundary markers, which ADR 0011 would put at BOS=2 and
    EOS=3; mapping them there instead of through `USER_BASE + c` would place
    real emission mass on reserved fibres, and `HMMParams` rejects a non-zero
    reserved block at construction -- so the fixtures would stop loading rather
    than decode subtly wrongly. That is the good failure. Whether the library
    should ever treat a corpus's boundary markers as BOS/EOS is a separate
    question, and it belongs to `dataseq`'s encoder rather than here.
    """
    directory = Path(directory)
    init_state_p = read_ascii_matrix(directory / "init_state_p")
    transition_p = read_ascii_matrix(directory / "transition_p")
    saved_output_p = read_ascii_matrix(directory / "output_p")

    init_state_p = init_state_p / init_state_p.sum()
    transition_p = transition_p / transition_p.sum(axis=1, keepdims=True)

    size, _, alphabet_size = saved_output_p.shape
    output_p = np.zeros((size, size, USER_BASE + alphabet_size), dtype=np.float64)
    output_p[:, :, USER_BASE:] = saved_output_p

    fibre_sums = output_p.sum(axis=2, keepdims=True)
    output_p = np.divide(
        output_p, fibre_sums, out=np.zeros_like(output_p), where=fibre_sums > 0.0
    )

    # Names are synthesized: see the docstring. SymbolTable assigns them
    # USER_BASE, USER_BASE + 1, ... in first-appearance order, which is the
    # renumbering the placement above assumes.
    vocabulary = SymbolTable([f"s{i}" for i in range(alphabet_size)])

    return HMMParams(
        init_state_p=init_state_p,
        transition_p=transition_p,
        output_p=output_p,
        vocabulary=vocabulary,
    )


def corpus_alphabet() -> dict[int, str]:
    """Lush's own code-to-name mapping, which a saved model does not carry.

    `0 -> begin, 1 -> end, 2 -> c, 3 -> d, 4 -> b, 5 -> a`. Read from the corpus
    rather than from any model directory, because that is the only place it
    survives: see :func:`load_params` on the pointer addresses.
    """
    mapping = {}
    for line in (CORPUS / "_alphabet").read_text().splitlines():
        code, name = line.split("\t")
        mapping[int(code)] = name
    return mapping


def load_corpus_codes() -> np.ndarray:
    """Every `.seq` in the corpus concatenated, in numeric order, as Lush codes.

    This is the stream the original's `fprop-all` built and decoded in one pass,
    which is why one `.vpath.xls` covers all 200 sequences rather than one per
    file. Numeric order is not an assumption: the concatenation reproduces each
    saved decode's `Output` column exactly, and a different order would not.
    """
    n_sequences = int((CORPUS / "_size").read_text().split()[0])
    codes: list[int] = []
    for index in range(n_sequences):
        codes.extend(int(t) for t in (CORPUS / f"{index}.seq").read_text().split())
    return np.asarray(codes, dtype=np.int64)


def load_corpus_record() -> SequenceRecord:
    """The corpus as one `SequenceRecord`, renumbered onto the ADR 0011 block.

    `+ USER_BASE` is the same renumbering :func:`load_params` applies to the
    emission axis, and the two must agree or the codes would index fibres
    belonging to other symbols. See that function on why Lush's `begin`/`end`
    become user symbols rather than BOS/EOS.
    """
    return SequenceRecord(load_corpus_codes() + USER_BASE, label="set02a_200")


def load_vpath(model) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read a saved decode: `<model>.vpath.xls`, written by `save-viterbi-path`.

    Returns `(symbols, states, entropies)`. All three columns are `N + 1` long,
    because the file's first data row is the state occupied *before* the first
    symbol -- printed with a literal `-` in the `Output` column, since no symbol
    has been emitted yet. That is ADR 0015's arc-emission geometry written to
    disk by the original itself, which is a stronger statement of it than any
    document here: the author's own file format has one more state than it has
    symbols.

    `entropies` are `state_entropies[states[i]]` as the original computed them,
    printed at four decimals by `round-to`.
    """
    path = FIXTURES / (str(model).removesuffix(".hmm") + ".vpath.xls")
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    if header != ["Output", "States", "Entropy"]:
        raise ValueError(f"{path} is not a saved Viterbi path: header {header}")

    symbols, states, entropies = [], [], []
    for line in lines[1:]:
        symbol, state, entropy = line.split("\t")
        symbols.append(symbol)
        states.append(int(state.strip("[]")))
        entropies.append(float(entropy))
    return (
        symbols,
        np.asarray(states, dtype=np.int64),
        np.asarray(entropies, dtype=np.float64),
    )
