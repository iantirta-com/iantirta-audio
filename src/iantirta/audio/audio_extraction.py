


from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from iantirta.audio import AudioSegment, load_audio
from iantirta.audio.audio_extraction_utils import (
    get_mask_starts_ends,
    mask_delete,
    mask_merge_forward,
    mask_merge_lower_duration,
    mask_uniform,
    optimal_fft_length,
    split_long_block_mask,
    uniform_1d,
)

__all__ = [
    "AudioExtraction",
    "detect_audio_activiy",
]

@dataclass(slots=True)
class AudioFeature:
    """
    Base Class that represent audio Features
    """
    arr: np.ndarray

    smooth_distance_ms = 60
    floor_pct = 5
    std_threshold_multipler = 0.2

    peak_prominence: float = 0.01
    hop_duration_ms: int = 16

    @property
    def smoothed(self):
        if self._smoothed is None:
            self._smoothed = uniform_1d(
                self.arr,
                uniform_size_ms=self.smooth_distance_ms,
                hop_duration_ms=self.hop_duration_ms
            )
        return self.smoothed

    @property
    def threshold(self):
        if self._threshold is None:
            noise_floor = np.percentile(self.smoothed, self.floor_pct)
            self._threshold = noise_floor + (
                np.std(self.smoothed) * self.std_threshold_multipler
            )
        return self._threshold

    @property
    def mask(self):
        if self._mask is None:
            self._mask = (self.smoothed > self.threshold)
        return self._mask

    @property
    def valleys(self) -> np.ndarray:
        """
        Local minima of the feature signal.

        a.k.a Inverted peaks.
        """
        if self._valleys is None:
            import scipy.signal

            _max = np.max(np.abs(self.smoothed))
            norm = (self.smoothed / _max) if _max > 0 else self.smoothed
            inverted = -norm
            self._valleys = scipy.signal.find_peaks(
                x=inverted,
                prominence=self.peak_prominence
            )
        return self._valleys


@dataclass(slots=True)
class RMSFeature(AudioFeature):
    """
    Represent RMS Feature
    """


@dataclass(slots=True)
class MelSpectrogramFeature(AudioFeature):
    """
    Represent Log Mel Spectrogram Feature
    """
    S: np.ndarray | None = None


class AudioExtraction:
    """
    Class that will extract audio features.
    """
    _rms = None

    def __init__(
        self,
        wav: np.ndarray,
        sr: int,
        *,
        hop_duration_ms: int = 16,
        win_duration_ms: int = 64,
        mel_floor: float = 1e-10,
        fmin: float = 80,
        fmax: float = 7600,
        num_mel_bins: int = 80,
        reference: float = 1.0,
        min_value: float = 1e-10,
        db_range: float | None = None,
    ):
        self.wav = wav
        self.sr = sr

        self.hop_duration_ms = hop_duration_ms
        self.win_length = win_duration_ms * sr // 1000
        self.hop_length = hop_duration_ms * sr // 1000
        self.n_fft = optimal_fft_length(self.win_length)

        self.n_mels = num_mel_bins
        self.fmin = fmin
        self.fmax = fmax

        self.reference = reference
        self.min_value = min_value
        self.db_range = db_range

    # -----------------------
    # Audio Method Process
    # -----------------------

    def get_rms(self) -> RMSFeature:
        rms = librosa.feature.rms(
            y=self.wav,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )[0]
        self._rms = RMSFeature(arr=rms, hop_duration_ms=self.hop_duration_ms)
        return self._rms

    def get_mel_spectrogram(self) -> MelSpectrogramFeature:
        spec = librosa.feature.melspectrogram(
            y=self.wav,
            sr=self.sr,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            pad_mode='reflect',
        )
        spec = librosa.power_to_db(
            S=spec,
            ref=self.reference,
            amin=self.min_value,
            top_db=self.db_range
        )
        energy = spec.sum(axis=0)
        self._mel_spectrogram = MelSpectrogramFeature(
            S=spec,
            arr=energy,
            hop_duration_ms=self.hop_duration_ms
        )
        return self._mel_spectrogram

    def get_activity_mask(self) -> np.ndarray:
        """
        Method that is used to get audio activity mask.
        based on threshold and features.
        """
        activity_mask = self.rms.mask | self.mel_spectrogram.mask
        
        activity_mask = mask_merge_forward(
            activity_mask,
            self.rms.valleys,
            hop_duration_ms=self.hop_duration_ms
        )
        activity_mask = mask_uniform(
            activity_mask,
            hop_duration_ms=self.hop_duration_ms
        )
        activity_mask = mask_delete(
            activity_mask,
            hop_duration_ms=self.hop_duration_ms
        )
        activity_mask = mask_merge_lower_duration(
            activity_mask,
            hop_duration_ms=self.hop_duration_ms
        )

        self._activity_mask = split_long_block_mask(
            activity_mask,
            target_valleys=self.rms.valleys,
            mel_mask=self.mel_spectrogram.mask,
            hop_duration_ms=self.hop_duration_ms
        )
        return self._activity_mask


    def get_audio_segments(self) -> list[AudioSegment]:
        """
        Method that is used to get audio activity segments.
        """
        audio_segments = []

        starts, ends = get_mask_starts_ends(self.activity_mask)

        for s, e in zip(starts, ends):
            audio_segments.append(AudioSegment(
                start=self.times[s],
                end=self.times[e] if e < len(self.times) else self.times[-1]
            ))

        self._segments = audio_segments
        return self._segments

    # -----------------------
    # Audio Features Properties
    # -----------------------

    @property
    def times(self) -> np.ndarray:
        if self._times is None:
            self._times = librosa.times_like(
                X=self.wav,
                sr=self.sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
            )
        return self._times
    
    @property
    def rms(self) -> RMSFeature:
        if self._rms is None:
            self.get_rms()
        return self._rms

    @property
    def mel_spectrogram(self) -> MelSpectrogramFeature:
        """
        Mel Spectrogram
        """
        if self._mel_spectrogram is None:
            self.get_mel_spectrogram()
        return self._mel_spectrogram

    @property
    def flux(self):
        """
        Also known as onset_strength
        """

    @property
    def activity_mask(self):
        if self._activity_mask is None:
            self.get_activity_mask()
        return self._activity_mask

    @property
    def segments(self) -> list[AudioSegment]:
        if self._segments is None:
            self.get_audio_segments()
        return self._segments


def detect_audio_activiy(
    audio_path: str | Path,
    sr: int = 16000
) -> AudioExtraction:
    wav, sr = load_audio(audio_path, sampling_rate=sr, num_channels=1)

    audio_extraction = AudioExtraction(wav=wav, sr=sr)
    audio_extraction.get_audio_segments()

    return audio_extraction
