
from typing import TYPE_CHECKING

from .base_feature import FeatureMixin

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "Mel"
]


class Mel(FeatureMixin):
    """Mel Feature"""

    @property
    def data(self) -> "np.ndarray":
        import librosa
        return librosa.feature.melspectrogram(y=self.audio, sr=self.sr, n_mels=256, n_fft=self.framelength, hop_length=self.hoplength)
