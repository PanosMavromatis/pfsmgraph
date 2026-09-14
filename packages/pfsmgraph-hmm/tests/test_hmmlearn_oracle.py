"""The numpy reference, checked against ``hmmlearn``: an oracle outside this lineage.

**Every test here reaches a private function**, labelled as ADR 0003 asks, for the
reason ``test_forward_backward.py`` gives: there is no public forward-backward or
trainer yet. ``hmmlearn`` is **not** an ADR 0003 backend either. A backend is
another implementation of the same kernel; ``CategoricalHMM`` is a different
model, a state-emission one, that coincides with ours only on the reduced case
below. It is a test-only dependency in the root ``dev`` group, imported without
``importorskip``, because a dev environment without it is a broken sync.

**Why a second oracle, after exact enumeration.** Enumeration shares no
recurrence and no rounding with the kernel, but it was written here, from the
same reading of the model. ``hmmlearn`` shares no derivation with us at all, so
it is the one check that can catch a misreading of the model itself, the kind
the ``torch`` backend would inherit, being autograd over the same derivation.

**The reduced case** (ADR 0015). An arc-emission model whose emission depends
only on the destination, ``output_p[i, j, USER_BASE + k] = emissionprob_[j, k]``,
is a state-emission model. The two differ at the first step only: our first
symbol is emitted crossing ``s_0 -> s_1``, and hmmlearn's first state is our
``s_1``, so its ``startprob_`` is ``init_state_p @ transition_p``. hmmlearn's
posteriors are therefore our ``gamma[1:]``; our ``gamma[0]``, the state before
any symbol, has no counterpart there.

Comparisons use hmmlearn's ``scaling`` implementation, which does the arithmetic
ADR 0020 prescribes for ours. Its ``log`` implementation rounds differently in
its logsumexp: measured 9.5e-13 on posteriors at ``S = 32``, against 1.4e-15 for
``scaling``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hmmlearn.hmm import CategoricalHMM

from pfsmgraph.dataseq import USER_BASE
from pfsmgraph.hmm._forward_backward import (
    _description_length,
    _forward_backward,
    _state_posteriors,
)

SEED = 20260916

#: Worst measured over these cases: 3.2e-15 on the log-likelihood and 1.4e-15 on
#: a posterior, at ``S = 32``, ``N = 3000``. 1e-12 is the tolerance the other
#: two oracle files use, and still orders of magnitude below what a one-position
#: shift or a wrong start distribution produces (see the cross-checks below).
ORACLE_TOL = 1e-12


def _reduced_model(rng, size, n_user, dead_fraction):
    """``(init, transition, emission)`` with a fraction of arcs exactly dead.

    ``emission`` is hmmlearn's ``(S, n_user)`` state-emission matrix; the
    diagonal stays alive so every transition row normalises.
    """
    init = rng.dirichlet(np.ones(size))
    transition = rng.dirichlet(np.ones(size), size=size)
    dead = rng.random((size, size)) < dead_fraction
    dead[np.arange(size), np.arange(size)] = False
    transition = np.where(dead, 0.0, transition)
    transition /= transition.sum(axis=1, keepdims=True)
    emission = rng.dirichlet(np.ones(n_user), size=size)
    return init, transition, emission


def _arc_output(size, emission):
    """Our ``(S, S, USER_BASE + n_user)`` output with emission on the destination."""
    n_user = emission.shape[1]
    output = np.zeros((size, size, USER_BASE + n_user))
    output[:, :, USER_BASE:] = emission[np.newaxis, :, :]
    return output


def _categorical(init, transition, emission):
    """hmmlearn's model for the same process: nothing initialised, nothing fitted."""
    size, n_user = emission.shape
    model = CategoricalHMM(
        n_components=size, n_features=n_user, implementation="scaling", init_params=""
    )
    model.startprob_ = init @ transition
    model.transmat_ = transition
    model.emissionprob_ = emission
    return model


CASES = [
    # (states, user symbols, record lengths, dead arc fraction)
    (1, 3, (12,), 0.0),
    (2, 2, (1,), 0.0),
    (3, 4, (40,), 0.3),
    (8, 6, (200,), 0.4),
    (16, 12, (1000,), 0.5),
    (32, 20, (3000,), 0.7),
    (5, 7, (25, 1, 60, 9), 0.2),
]


def _case(index):
    rng = np.random.default_rng(SEED + index)
    size, n_user, lengths, dead = CASES[index]
    init, transition, emission = _reduced_model(rng, size, n_user, dead)
    records = [rng.integers(0, n_user, n) for n in lengths]
    return init, transition, emission, records


def _ours(init, transition, emission, records):
    """Summed description length in bits, and ``gamma[1:]`` concatenated."""
    output = _arc_output(len(init), emission)
    bits, posteriors = 0.0, []
    for record in records:
        alpha, beta, scale = _forward_backward(
            init, transition, output, record + USER_BASE
        )
        bits += _description_length(scale)
        posteriors.append(_state_posteriors(alpha, beta, scale)[1:])
    return bits, np.concatenate(posteriors)


def _theirs(model, records):
    """hmmlearn's summed log-likelihood in bits, and its posteriors."""
    X = np.concatenate(records).reshape(-1, 1)
    loglik, posteriors = model.score_samples(X, lengths=[len(r) for r in records])
    return -loglik / math.log(2), posteriors


# --- Forward-backward --------------------------------------------------------


@pytest.mark.parametrize("index", range(len(CASES)))
def test_description_length_matches_hmmlearn_log_likelihood(index):
    init, transition, emission, records = _case(index)
    ours, _ = _ours(init, transition, emission, records)
    theirs, _ = _theirs(_categorical(init, transition, emission), records)
    assert ours == pytest.approx(theirs, rel=ORACLE_TOL)


@pytest.mark.parametrize("index", range(len(CASES)))
def test_state_posteriors_match_hmmlearn(index):
    init, transition, emission, records = _case(index)
    _, ours = _ours(init, transition, emission, records)
    _, theirs = _theirs(_categorical(init, transition, emission), records)
    assert ours.shape == theirs.shape
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=ORACLE_TOL)


def test_the_comparison_can_fail():
    """Each half of the mapping is load-bearing, measured on the 3-state case.

    Handing hmmlearn our ``init_state_p`` unchanged, or reading our posteriors
    one position early, both miss by many orders of magnitude more than
    ``ORACLE_TOL``. A green run above is therefore evidence about the kernel,
    not about a comparison too loose to fail.
    """
    init, transition, emission, records = _case(2)
    bits, gamma = _ours(init, transition, emission, records)

    wrong_start = _categorical(init, transition, emission)
    wrong_start.startprob_ = init
    theirs, _ = _theirs(wrong_start, records)
    assert abs(bits - theirs) > 1e6 * ORACLE_TOL * abs(theirs)

    _, posteriors = _theirs(_categorical(init, transition, emission), records)
    output = _arc_output(len(init), emission)
    alpha, beta, scale = _forward_backward(
        init, transition, output, records[0] + USER_BASE
    )
    shifted = _state_posteriors(alpha, beta, scale)[:-1]
    assert np.abs(shifted - posteriors).max() > 1e6 * ORACLE_TOL
    assert np.abs(gamma - posteriors).max() <= ORACLE_TOL
