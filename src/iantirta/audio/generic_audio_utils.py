
import numpy as np


def slice_np(
    wav: np.ndarray,
    start: float,
    end: float,
    sr: int,
) -> np.ndarray:
    """
    This method will chunk/slice given audio,
    with the new one respecting the start and end ts.

    :param np.ndarray wav:
    :param float start:
        start ts in seconds
    :param float end:
        end ts in seconds
    :param int sr:
    :return: New chunked audio.
    :rtype: np.ndarray
    """
    return wav[int(start*sr):int(end*sr)]
