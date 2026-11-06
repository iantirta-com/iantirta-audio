


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
    rms as rms_lib,
    melspectrogram,
    pyin,
    ms2frame
)
from iantirta.audio.extraction_utils import spectrogram

__all__ = [
    "AudioExtraction",
    "detect_audio_activity",
]

@dataclass(slots=True)
class AudioFeature:
    """
    Base Class that represent audio Features
    """
    arr: np.ndarray

    smooth_distance_ms: int = 60
    floor_pct: int = 5
    std_threshold_multipler: float = 0.2

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
    _f0 = None
    _voiced_flag = None
    _refined_voiced_flag = None
    _voiced_probs = None
    _rms = None
    _mel_spectrogram = None
    _segments = None
    _times = None

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
        *shape, channels, samples = wav.shape
        self.num_shape = shape
        self.num_channels = channels
        self.num_samples = samples

        self.samples = wav.squeeze()

        if self.samples.ndim != 1:
            raise ValueError(f"Input waveform must have only one dimension, shape is {self.samples.shape}")

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

    def get_pyin(self):
        f0, voiced_flag, voiced_probs = pyin(
            self.samples,
            sr=self.sr,
            fmin=np.float64(65.40639132514966),
            fmax=np.float64(2093.004522404789),
            frame_length=self.win_length,
            hop_length=self.hop_length,
            n_thresholds= 100,
            beta_parameters= (2, 18),
            boltzmann_parameter= 2,
            resolution= 0.1,
            max_transition_rate= 35.92,
            switch_prob= 0.01,
            no_trough_prob= 0.01,
            fill_na= np.nan,
            center=True,
            pad_mode="reflect",
            transition_min_prob=1e-4,
        )

        self._f0 = f0
        self._voiced_flag = voiced_flag
        self._voiced_probs = voiced_probs

        return f0, voiced_flag, voiced_probs

    def get_refined_voiced_flag(self):
        arr = self.voiced_flag.copy()

        arr = mask_uniform(arr, hop_duration_ms=self.hop_duration_ms)
        arr = mask_delete(arr, hop_duration_ms=self.hop_duration_ms)
        arr = mask_merge_lower_duration(arr, hop_duration_ms=self.hop_duration_ms)
        
        max_block_ms: int = 10000
        min_block_ms: int = 1000
        max_frame = ms2frame(max_block_ms, hop_duration_ms=self.hop_duration_ms)
        min_frame = ms2frame(min_block_ms, hop_duration_ms=self.hop_duration_ms)
        voiced_probs = self.voiced_probs

        while True:
            starts, ends = get_mask_starts_ends(arr)

            if len(starts) == 0:
                break

            durations = ends - starts
            long_blocks = np.where(durations >= max_frame)[0]

            if len(long_blocks) == 0:
                break

            split_occurred = False
            for idx in long_blocks:
                s = starts[idx]
                e = ends[idx]

                valid_start = s + min_frame
                valid_end = e - min_frame

                if valid_start >= valid_end:
                    # Segment is longer than max_frame, but not long enough to split into two valid min_frame chunks
                    continue

                valid_probs = voiced_probs[valid_start:valid_end]
                local_min_idx = np.argmin(valid_probs)
                global_min_idx = valid_start + local_min_idx
                arr[global_min_idx] = False
                split_occurred = True

            if not split_occurred:
                break

        self._refined_voiced_flag = arr
        return self._refined_voiced_flag

    def get_rms(self) -> RMSFeature:
        rms = rms_lib(
            self.samples,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )[0]
        self._rms = RMSFeature(arr=rms, hop_duration_ms=self.hop_duration_ms)
        return self._rms

    def get_mel_spectrogram(self) -> MelSpectrogramFeature:
        win_function: str = "hann_window"
        window = spectrogram.window_function(
            window_length=self.win_length,
            name=win_function,
            periodic=True
        )
        spec = melspectrogram(
            self.samples,
            window=window,
            sampling_rate=self.sr,
            fft_length=self.n_fft,
            frame_length=self.win_length,
            hop_length=self.hop_length,
            power=2.0,
            log_mel=None,
            num_mel_filters=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
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

        starts, ends = get_mask_starts_ends(self.refined_voiced_flag)

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
    def f0(self):
        if self._f0 is None:
            self.get_pyin()
        return self._f0

    @property
    def voiced_flag(self):
        if self._voiced_flag is None:
            self.get_pyin()
        return self._voiced_flag

    @property
    def voiced_probs(self):
        if self._voiced_probs is None:
            self.get_pyin()
        return self._voiced_probs

    @property
    def refined_voiced_flag(self):
        if self._refined_voiced_flag is None:
            self.get_refined_voiced_flag()
        return self._refined_voiced_flag

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


def detect_audio_activity(
    audio_path: str | Path,
    sr: int = 16000
) -> AudioExtraction:
    wav, sr = load_audio(audio_path, sampling_rate=sr, num_channels=1)

    audio_extraction = AudioExtraction(wav=wav, sr=sr)
    audio_extraction.get_audio_segments()

    return audio_extraction
