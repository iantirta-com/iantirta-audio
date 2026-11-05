from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ------------------
# Helper Method
# ------------------

def is_positive_int(x: float) -> bool:
    """Check that x is a positive integer, i.e. 1 or greater.

    Parameters
    ----------
    x : number
        A scalar value to check

    Returns
    -------
    positive : bool
        True if x is a positive integer, False otherwise
    """
    # Check type first to catch None values.
    return isinstance(x, (int, np.integer)) and (x > 0)


def pad_center(
    data: np.ndarray,
    *,
    size: int,
    axis: int = -1,
    **kwargs: Any
) -> np.ndarray:
    """
    Pad an array to a target length along a target axis.

    This differs from `np.pad` by centering the data prior to padding,
    analogous to `str.center`

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered
    size : int >= len(data) [scalar]
        Length to pad ``data``
    axis : int
        Axis along which to pad and center the data
    **kwargs : additional keyword arguments
        arguments passed to `np.pad`

    Returns
    -------
    data_padded : np.ndarray
        ``data`` centered and padded to length ``size`` along the
        specified axis

    Raises
    ------
    ValueError
        If ``size < data.shape[axis]``

    See Also
    --------
    numpy.pad
    """
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(
            f"Target size ({size:d}) must be at least input size ({n:d})"
        )

    return np.pad(data, lengths, **kwargs)


def get_window(window, Nx: int, *, fftbins: bool = True) -> np.ndarray:
    """
    Compute a window function.

    This is a wrapper for `scipy.signal.get_window` that additionally
    supports callable or pre-computed windows.

    Parameters
    ----------
    window : str, tuple, number, callable, or list-like
        The window specification:

        - If string, it's the name of the window function (e.g., `'hann'`)
        - If tuple, it's the name of the window function and any parameters
          (e.g., `('kaiser', 4.0)`)
        - If numeric, it is treated as the beta parameter of the `'kaiser'`
          window, as in `scipy.signal.get_window`.
        - If callable, it's a function that accepts one integer argument
          (the window length)
        - If list-like, it's a pre-computed window of the correct length `Nx`

    Nx : int > 0
        The length of the window

    fftbins : bool, optional
        If True (default), create a periodic window for use with FFT
        If False, create a symmetric window for filter design applications.

    Returns
    -------
    get_window : np.ndarray
        A window of length `Nx` and type `window`

    See Also
    --------
    scipy.signal.get_window

    Raises
    ------
    ValueError
        If `window` is supplied as a vector of length != `n_fft`,
        or is otherwise mis-specified.
    """
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        import scipy.signal
        win: np.ndarray = scipy.signal.get_window(window, Nx, fftbins=fftbins)  # type: ignore
        return win

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ValueError(f"Window size mismatch: {len(window):d} != {Nx:d}")
    else:
        raise ValueError(f"Invalid window specification: {window!r}")

# ------------------
# viterbi Specific helper Method
# ------------------

try:
    from numba import jit

except ImportError as e:
    jit = None
    warnings.warn(
        f"Numba is unavailable; running without JIT optimization. Error: {e}",
        RuntimeWarning,
        stacklevel=2,
    )

def optional_jit(func):
    if jit is None:
        return func

    return jit(nopython=True, cache=True)(func)

@jit(nopython=True, cache=True)  # type: ignore
def _viterbi(
    log_prob: np.ndarray,
    log_trans: np.ndarray,
    log_p_init: np.ndarray,
    log_trans_threshold: float,
) -> tuple[NDArray[np.uint16], NDArray[np.float64]]:  # pragma: no cover
    """
    Core Viterbi algorithm.

    This is intended for internal use only.

    Parameters
    ----------
    log_prob : np.ndarray [shape=(T, m)]
        ``log_prob[t, s]`` is the conditional log-likelihood
        ``log P[X = X(t) | State(t) = s]``
    log_trans : np.ndarray [shape=(m, m)]
        The log transition matrix
        ``log_trans[i, j] = log P[State(t+1) = j | State(t) = i]``
    log_p_init : np.ndarray [shape=(m,)]
        log of the initial state distribution
    log_trans_threshold : float
        threshold above which transitions are considered feasible.
        Use -np.inf to perform a full search (exact viterbi).
        Otherwise, only transitions with (log) probability above the threshold
        are considered in the loop.

    Returns
    -------
    None
        All computations are performed in-place on ``state, value, ptr``.
    """
    n_steps, n_states = log_prob.shape

    state = np.zeros(n_steps, dtype=np.uint16)
    value = np.zeros((n_steps, n_states), dtype=np.float64)
    ptr = np.zeros((n_steps, n_states), dtype=np.uint16)

    # factor in initial state distribution
    value[0] = log_prob[0] + log_p_init

    if np.isfinite(log_trans_threshold):
        pred_states = []
        for j in range(n_states):
            possible_states = np.flatnonzero(log_trans[:, j] >= log_trans_threshold)
            if len(possible_states) == 0:
                raise ValueError(
                    f"Empty transition matrix detected for state {j} in Viterbi. "
                    f"Try reducing your minimum transition probability threshold."
                )
            pred_states.append(possible_states)

        for t in range(1, n_steps):
            # Want V[t, j] <- p[t, j] * max_k V[t-1, k] * A[k, j]
            #    assume at time t-1 we were in state k
            #    transition k -> j
            for j in range(n_states):
                best_cost = -np.inf
                for k in pred_states[j]:
                    cost = value[t - 1, k] + log_trans[k, j]
                    if cost > best_cost:
                        ptr[t, j] = k
                        best_cost = cost
                value[t, j] = log_prob[t, j] + best_cost
    else:
        # Same as above, but doing a full search over previous states
        for t in range(1, n_steps):
            for j in range(n_states):
                best_cost = -np.inf
                for kj in range(n_states):
                    cost = value[t - 1, kj] + log_trans[kj, j]
                    if cost > best_cost:
                        ptr[t, j] = kj
                        best_cost = cost
                value[t, j] = log_prob[t, j] + best_cost

    # Now roll backward
    # Get the last state
    state[-1] = np.argmax(value[-1])

    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]

    logp = value[-1:, state[-1]]

    return state, logp

# ------------------
# API Method
# ------------------

def transition_loop(n_states: int, prob: float | Iterable[float]):
    """
    Construct a self-loop transition matrix over ``n_states``.

    The transition matrix will have the following properties:

        - ``transition[i, i] = p`` for all ``i``
        - ``transition[i, j] = (1 - p) / (n_states - 1)`` for all ``j != i``

    This type of transition matrix is appropriate when states tend to be
    locally stable, and there is no additional structure between different
    states.  This is primarily useful for de-noising frame-wise predictions.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    prob : float in [0, 1] or iterable, length=n_states
        If a scalar, this is the probability of a self-transition.

        If a vector of length ``n_states``, ``p[i]`` is the probability of self-transition in state ``i``

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix
    """
    if not (is_positive_int(n_states) and (n_states > 1)):
        raise ValueError(f"n_states={n_states} must be a positive integer > 1")

    transition = np.empty((n_states, n_states), dtype=np.float64)

    # if it's a float, make it a vector
    prob = np.asarray(prob, dtype=np.float64)

    if prob.ndim == 0:
        prob = np.tile(prob, n_states)

    if prob.shape != (n_states,):
        raise ValueError(
            f"prob={prob} must have length equal to n_states={n_states}"
        )

    if np.any(prob < 0) or np.any(prob > 1):
        raise ValueError(f"prob={prob} must have values in the range [0, 1]")

    for i, prob_i in enumerate(prob):
        transition[i] = (1.0 - prob_i) / (n_states - 1)
        transition[i, i] = prob_i

    return transition


def transition_local(
    n_states: int,
    width: int | Iterable[int],
    *,
    window: str = "triangle",
    wrap: bool = False,
):
    """Construct a localized transition matrix.

    The transition matrix will have the following properties:

        - ``transition[i, j] = 0`` if ``|i - j| > width``
        - ``transition[i, i]`` is maximal
        - ``transition[i, i - width//2 : i + width//2]`` has shape ``window``

    This type of transition matrix is appropriate for state spaces
    that discretely approximate continuous variables, such as in fundamental
    frequency estimation.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    width : int >= 1 or iterable
        The maximum number of states to treat as "local".
        If iterable, it should have length equal to ``n_states``,
        and specify the width independently for each state.

    window : str, callable, or window specification
        The window function to determine the shape of the "local" distribution.

        Any window specification supported by `filters.get_window` will work here.

        .. note:: Certain windows (e.g., 'hann') are identically 0 at the boundaries,
            so and effectively have ``width-2`` non-zero values.  You may have to expand
            ``width`` to get the desired behavior.

    wrap : bool
        If ``True``, then state locality ``|i - j|`` is computed modulo ``n_states``.
        If ``False`` (default), then locality is absolute.

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix
    """
    if not (is_positive_int(n_states) and n_states > 1):
        raise ValueError(f"n_states={n_states} must be a positive integer > 1")

    width = np.asarray(width, dtype=int)
    if width.ndim == 0:
        width = np.tile(width, n_states)

    if width.shape != (n_states,):
        raise ValueError(
            f"width={width} must have length equal to n_states={n_states}"
        )

    if np.any(width < 1):
        raise ValueError(f"width={width} must be at least 1")

    transition = np.zeros((n_states, n_states), dtype=np.float64)

    # Fill in the widths.  This is inefficient, but simple
    for i, width_i in enumerate(width):
        trans_row = pad_center(
            get_window(window, width_i, fftbins=False), size=n_states
        )
        trans_row = np.roll(trans_row, n_states // 2 + i + 1)

        if not wrap:
            # Knock out the off-diagonal-band elements
            trans_row[min(n_states, i + width_i // 2 + 1) :] = 0
            trans_row[: max(0, i - width_i // 2)] = 0

        transition[i] = trans_row

    # Row-normalize
    transition /= transition.sum(axis=1, keepdims=True)

    return transition


@overload
def viterbi(
    prob: np.ndarray,
    transition: np.ndarray,
    *,
    p_init: np.ndarray | None = ...,
    return_logp: Literal[True],
    transition_min_prob: float | None = ...,
) -> tuple[NDArray[np.uint16], NDArray[np.float64]]: ...


@overload
def viterbi(
    prob: np.ndarray,
    transition: np.ndarray,
    *,
    p_init: np.ndarray | None = ...,
    return_logp: Literal[False] = ...,
    transition_min_prob: float | None = ...,
) -> NDArray[np.uint16]: ...


def viterbi(
    prob: np.ndarray,
    transition: np.ndarray,
    *,
    p_init: np.ndarray | None = None,
    return_logp: bool = False,
    transition_min_prob: float | None = None,
) -> NDArray[np.uint16] | tuple[NDArray[np.uint16], NDArray[np.float64]]:
    """
    Viterbi decoding from observation likelihoods.

    Given a sequence of observation likelihoods ``prob[s, t]``,
    indicating the conditional likelihood of seeing the observation
    at time ``t`` from state ``s``, and a transition matrix
    ``transition[i, j]`` which encodes the conditional probability of
    moving from state ``i`` to state ``j``, the Viterbi algorithm [#]_ computes
    the most likely sequence of states from the observations.

    .. [#] Viterbi, Andrew. "Error bounds for convolutional codes and an
        asymptotically optimum decoding algorithm."
        IEEE transactions on Information Theory 13.2 (1967): 260-269.

    Parameters
    ----------
    prob : np.ndarray [shape=(..., n_states, n_steps), non-negative]
        ``prob[..., s, t]`` is the probability of observation at time ``t``
        being generated by state ``s``.
    transition : np.ndarray [shape=(n_states, n_states), non-negative]
        ``transition[i, j]`` is the probability of a transition from i->j.
        Each row must sum to 1.
    p_init : np.ndarray [shape=(n_states,)]
        Optional: initial state distribution.
        If not provided, a uniform distribution is assumed.
    return_logp : bool
        If ``True``, return the log-likelihood of the state sequence.
    transition_min_prob : float, >=0
        Optional: if provided, the minimum transition probability to consider
        during decoding.

    Returns
    -------
    states : np.ndarray [shape=(..., n_steps,)]
        The most likely state sequence.
        If ``prob`` contains multiple channels of input, then each channel is
        decoded independently.

    logp : scalar [float] or np.ndarray, optional
        If ``return_logp=True``, the log probability of ``states`` given
        the observations.

    See Also
    --------
    viterbi_discriminative : Viterbi decoding from state likelihoods

    Examples
    --------
    Example from https://en.wikipedia.org/wiki/Viterbi_algorithm#Example

    In this example, we have two states ``healthy`` and ``fever``, with
    initial probabilities 60% and 40%.

    We have three observation possibilities: ``normal``, ``cold``, and
    ``dizzy``, whose probabilities given each state are:

    ``healthy => {normal: 50%, cold: 40%, dizzy: 10%}`` and
    ``fever => {normal: 10%, cold: 30%, dizzy: 60%}``

    Finally, we have transition probabilities:

    ``healthy => healthy (70%)`` and
    ``fever => fever (60%)``.

    Over three days, we observe the sequence ``[normal, cold, dizzy]``,
    and wish to know the maximum likelihood assignment of states for the
    corresponding days, which we compute with the Viterbi algorithm below.

    >>> p_init = np.array([0.6, 0.4])
    >>> p_emit = np.array([[0.5, 0.4, 0.1],
    ...                    [0.1, 0.3, 0.6]])
    >>> p_trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> path, logp = viterbi(p_emit, p_trans, p_init=p_init,
    ...                                       return_logp=True)
    >>> print(logp, path)
    -4.19173690823075 [0 0 1]
    """
    from iantirta.audio.audio_extraction_utils import tiny
    
    n_states, _n_steps = prob.shape[-2:]

    if transition.shape != (n_states, n_states):
        raise ValueError(
            f"transition.shape={transition.shape}, must be "
            f"(n_states, n_states)={n_states, n_states}"
        )

    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ValueError(
            "Invalid transition matrix: must be non-negative "
            "and sum to 1 on each row."
        )

    if np.any(prob < 0) or np.any(prob > 1):
        raise ValueError("Invalid probability values: must be between 0 and 1.")

    # Compute log-likelihoods while avoiding log-underflow
    epsilon = tiny(prob)

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(1.0 / n_states)
    elif (
        np.any(p_init < 0)
        or not np.allclose(p_init.sum(), 1)
        or p_init.shape != (n_states,)
    ):
        raise ValueError(f"Invalid initial state distribution: p_init={p_init}")

    log_trans = np.log(transition + epsilon)
    log_prob = np.log(prob + epsilon)
    log_p_init = np.log(p_init + epsilon)

    if transition_min_prob is not None and transition_min_prob > 0:
        log_trans_threshold = np.log(transition_min_prob + epsilon)
    elif transition_min_prob is None or transition_min_prob == 0:
        log_trans_threshold = -np.inf
    else:
        raise ValueError(
            f"Invalid transition_min_prob={transition_min_prob}, must be None or non-negative."
        )

    def _helper(lp) -> tuple[NDArray[np.uint16], NDArray[np.float64]]:
        # Transpose input
        _state, logp = _viterbi(lp.T, log_trans, log_p_init, log_trans_threshold)
        # Transpose outputs for return
        return _state.T, logp

    states: np.ndarray
    logp: np.ndarray

    if log_prob.ndim == 2:
        states, logp = _helper(log_prob)
    else:
        # Vectorize the helper
        __viterbi = np.vectorize(
            _helper, otypes=[np.uint16, np.float64], signature="(s,t)->(t),(1)"
        )

        states, logp = __viterbi(log_prob)

        # Flatten out the trailing dimension introduced by vectorization
        logp = logp[..., 0]

    if return_logp:
        return states, logp

    return states
