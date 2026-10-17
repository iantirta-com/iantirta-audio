
from typing import TYPE_CHECKING

from .base_feature import FeatureMixin

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "RMS"
]


class RMS(FeatureMixin):
    """RMS Feature"""

    @property
    def data(self) -> "np.ndarray":
        import librosa
        return librosa.feature.rms(y=self.audio, frame_length=self.framelength, hop_length=self.hoplength)[0]
    