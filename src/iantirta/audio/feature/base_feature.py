

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureMixin:
    audio: np.ndarray

    sr: float | int
    framelength: int
    hoplength: int

    _times: np.ndarray | None = None
    _smoothed: np.ndarray | None = None
    _valleys: np.ndarray | None = None
    _threshold: np.float32 | None = None
    _mask: np.ndarray | None = None

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def _uniform_1d(
        arr: np.ndarray,
        *,
        uniform_size_ms: int = 60,
        precision_ms: int = 20,
    ) -> np.ndarray:
        import scipy.ndimage

        framesize = max(
            1,
            int(uniform_size_ms / precision_ms),
        )
        if framesize % 2 == 0:
            framesize += 1

        return scipy.ndimage.uniform_filter1d(arr, size=framesize)

    @property
    def times(self) -> np.ndarray:
        if self._times is None:
            import librosa
            self._times = librosa.times_like(
                self.data,
                sr=self.sr,
                hop_length=self.hoplength
            )

        return self._times

    @property
    def smoothed(self) -> np.ndarray:
        if self._smoothed is None:
            self._smoothed = self._uniform_1d(self.data)

        return self._smoothed

    @property
    def valleys(self) -> np.ndarray:
        """ Local minima of the feature signal """
        if self._valleys is None:
            import scipy.signal
            # Normalized
            _max = np.max(np.abs(self.smoothed))
            norm = (self.smoothed / _max) if _max > 0 else self.smoothed
            inverted = -norm
            self._valleys = scipy.signal.find_peaks(inverted, prominence=0.01)[0]

        return self._valleys

    @property
    def threshold(self) -> np.float32:
        if self._threshold is None:
            noise_floor = np.percentile(self.smoothed, 5)
            self._threshold = noise_floor + (np.std(self.smoothed) * 0.2)
    
        return self._threshold

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            self._mask = (self.smoothed > self.threshold)

        return self._mask


if __name__ == "__main__":
    import librosa
    import scipy.signal
    from pathlib import Path
    from rich import inspect as i
    audio_path = str(Path("../dataset/vocals.wav").expanduser().resolve())
    from iantirta.audio import load_audio
    import matplotlib.pyplot as plt
    from pprint import pprint
    wav, sr = load_audio(audio_path, num_channels=1)

    S = librosa.feature.melspectrogram(y=wav, sr=sr)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)
    librosa.display.colorbar_db(img)
    ax.set(title='Mel-frequency spectrogram')

    # times = librosa.times_like(rms)
    # ax[0].semilogy(times, rms[0], label='RMS Energy')
    # ax[0].set(xticks=[])
    # ax[0].legend()
    # ax[0].label_outer()
    plt.show()
    pprint(S)
    pprint(wav.shape)
    # peak1 = scipy.signal.find_peaks(-rms, prominence=0.01)[0]

    # from pprint import pprint
    # pprint(peak1)

    # peak2 = librosa.util.peak_pick(-rms, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    # pprint(peak2)