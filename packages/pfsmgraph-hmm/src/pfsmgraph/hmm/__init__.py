"""Hidden Markov models: arc-emission (Mealy), trained by Baum-Welch.

The formulation is the one fact about this package most likely to be lost in
translation, so it is stated first: a symbol is emitted while *crossing* a
transition, not while occupying a state. The emission parameter is
``output_p[i, j, symbol]`` -- source state, destination state, symbol -- and
never ``B[state, symbol]``. A path over ``N`` symbols therefore visits ``N + 1``
states. Every textbook, and every other library, is the state-emission
formulation. See ``docs/design/adr/0015-arc-emission-mealy-formulation.md``.

The package exposes the parameter value, the Viterbi decode over it, and
:func:`baum_welch`, which trains a model's parameters on a fixed topology.

Both take a keyword-only ``backend=``, defaulting to the pure-Python reference,
and :func:`backends` reports which backends can run here
(``docs/design/adr/0021-runtime-backend-selection.md``).
"""

from ._backends import BackendStatus, BackendUnavailableError, backends
from ._baum_welch import BaumWelchResult, baum_welch
from ._params import HMMParams
from ._viterbi import ImpossibleSequenceError, ViterbiPath, viterbi

__all__ = [
    "BackendStatus",
    "BackendUnavailableError",
    "BaumWelchResult",
    "HMMParams",
    "ImpossibleSequenceError",
    "ViterbiPath",
    "backends",
    "baum_welch",
    "viterbi",
]
