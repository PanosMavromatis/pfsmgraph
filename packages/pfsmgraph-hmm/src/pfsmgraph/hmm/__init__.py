"""Hidden Markov models: arc-emission (Mealy), trained by Baum-Welch.

The formulation is the one fact about this package most likely to be lost in
translation, so it is stated first: a symbol is emitted while *crossing* a
transition, not while occupying a state. The emission parameter is
``output_p[i, j, symbol]`` -- source state, destination state, symbol -- and
never ``B[state, symbol]``. A path over ``N`` symbols therefore visits ``N + 1``
states. Every textbook, and every other library, is the state-emission
formulation. See ``docs/design/adr/0015-arc-emission-mealy-formulation.md``.

0.1.0 exposes the parameter value and the Viterbi decode over it. There is no
trainer: the Lush original this package is translated from cannot express a
decode-only use of itself, so revision 02 builds no trainer at all and
Baum-Welch arrives in 0.2.0.
"""

from ._params import HMMParams
from ._viterbi import ImpossibleSequenceError, ViterbiPath, viterbi

__all__ = ["HMMParams", "ImpossibleSequenceError", "ViterbiPath", "viterbi"]
