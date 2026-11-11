import builtins
import importlib
import sys
import timeit
from pathlib import Path

import librosa
import numpy as np
import pytest

import iantirta.audio.audio_extraction_utils as audio_mod
from iantirta.audio.audio_extraction_utils import (
    _cumulative_mean_normalized_difference,
    autocorrelate,
    frame,
    melspectrogram,
    pyin,
    rms,
    stft,
)
from iantirta.audio.extraction_utils import spectrogram


def test_compare_rms(make_audio):
    sr, c = 16000, 1
    dummy_audio = make_audio(sr=sr, channels=c)

    hop_duration_ms: int = 16
    win_duration_ms: int = 64
    win_length = win_duration_ms * sr // 1000
    hop_length = hop_duration_ms * sr // 1000

    def rms1():
        return rms(dummy_audio.squeeze(), frame_length=win_length, hop_length=hop_length)

    def rms2():
        return librosa.feature.rms(
            y=dummy_audio.squeeze(),
            frame_length=win_length,
            hop_length=hop_length,
            pad_mode='reflect',
            dtype=np.float64
        )

    res_rms1 = rms1()
    res_rms2 = rms2()
    np.testing.assert_equal(res_rms1, res_rms2)

    t1 = timeit.timeit(rms1, number=100)
    t2 = timeit.timeit(rms2, number=100)

    print()
    print("="*50)
    print("RMS_1 Time:", t1)
    print("Librosa RMS Time:", t2)
    print("="*50)


def test_compare_stft(make_audio):
    sr, c = 16000, 1
    dummy_audio: np.ndarray = make_audio(sr=sr, channels=c)
    dummy_audio = dummy_audio.squeeze()

    hop_duration_ms: int = 16
    win_duration_ms: int = 64
    win_length = win_duration_ms * sr // 1000
    hop_length = hop_duration_ms * sr // 1000

    n_fft = spectrogram.optimal_fft_length(win_length)

    win_function: str = "hann_window"
    window = spectrogram.window_function(window_length=win_length, name=win_function, periodic=True)

    def stft1():
        return stft(
            waveform=dummy_audio,
            window=window,
            frame_length=win_length,
            hop_length=hop_length,
            fft_length=n_fft,
            center=True,
            pad_mode="reflect",
            onesided=True,
            dither=0.0,
            preemphasis=None,
            remove_dc_offset=False,
        )

    def librosa_stft():
        return librosa.stft(
            y=dummy_audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            dtype=None,
            pad_mode="reflect",
            out=None,
        )

    res_stft = stft1()
    res_librosa_stft = librosa_stft()
    np.testing.assert_almost_equal(res_stft, res_librosa_stft)

    t1 = timeit.timeit(stft1, number=100)
    t2 = timeit.timeit(librosa_stft, number=100)

    print()
    print("="*50)
    print(f"STFT {res_stft.shape} ({res_stft.dtype}) Time:", t1)
    print(f"Librosa STFT {res_librosa_stft.shape} ({res_librosa_stft.dtype}) Time:", t2)
    print("="*50)


def test_compare_mel_spectrogram(make_audio):
    sr, c = 16000, 1
    dummy_audio: np.ndarray = make_audio(sr=sr, channels=c)
    dummy_audio = dummy_audio.squeeze()

    hop_duration_ms: int = 16
    win_duration_ms: int = 64
    win_length = win_duration_ms * sr // 1000
    hop_length = hop_duration_ms * sr // 1000

    fmin: float = 80
    fmax: float = 7600
    num_mel_bins: int = 80

    n_fft = spectrogram.optimal_fft_length(win_length)

    win_function: str = "hann_window"
    window = spectrogram.window_function(window_length=win_length, name=win_function, periodic=True)

    def melspec1():
        return melspectrogram(
            waveform=dummy_audio,
            window=window,
            frame_length=win_length,
            hop_length=hop_length,
            fft_length=n_fft,
            power=2.0,
            log_mel=None,
            sampling_rate=sr,
            fmin=fmin,
            fmax=fmax,
            num_mel_filters=num_mel_bins,
        )

    def melspec_librosa():
        kwargs = {
            'n_mels': num_mel_bins,
            'fmin': fmin,
            'fmax': fmax,
            'htk': False,
            'norm': 'slaney',
            'dtype': np.float64
        }
        return librosa.feature.melspectrogram(
            y= dummy_audio,
            sr=sr,
            S=None,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            **kwargs,
        )

    res_melspec1 = melspec1()
    res_melspec_librosa = melspec_librosa()
    np.testing.assert_almost_equal(res_melspec1, res_melspec_librosa, decimal=4)

    t1 = timeit.timeit(melspec1, number=100)
    t2 = timeit.timeit(melspec_librosa, number=100)

    print()
    print("="*50)
    print(f"Mel Spectrogram {res_melspec1.shape} ({res_melspec1.dtype}) Time:", t1)
    print(f"Librosa Mel Spectrogram {res_melspec_librosa.shape} ({res_melspec_librosa.dtype}) Time:", t2)
    print("="*50)

    reference: float = 1.0
    min_value: float = 1e-10
    db_range: float | None = None

    ms_db = spectrogram.power_to_db(
        res_melspec1,
        reference=reference,
        min_value=min_value,
        db_range=db_range,
    )
    librosa_ms_db = librosa.power_to_db(
        res_melspec_librosa,
        ref=reference, # 1.0,
        amin=min_value, # 1e-10,
        top_db=db_range, # 80.0,
        axes="auto",
    )

    np.testing.assert_almost_equal(ms_db, librosa_ms_db, decimal=4)
    print(f"Dummy Audio {dummy_audio.shape} ({dummy_audio.dtype})")
    print(f"Mel Spectrogram (dB) {ms_db.shape} ({ms_db.dtype})")
    print(f"Librosa Mel Spectrogram (dB) {librosa_ms_db.shape} ({librosa_ms_db.dtype})")


def test_compare_make_frames(make_audio):
    sr, c = 16000, 1
    dummy_audio: np.ndarray = make_audio(sr=sr, channels=c)
    dummy_audio = dummy_audio.squeeze()

    hop_duration_ms: int = 16
    win_duration_ms: int = 64
    win_length = win_duration_ms * sr // 1000
    hop_length = hop_duration_ms * sr // 1000

    def frame_1():
        return frame(dummy_audio, frame_length=win_length, hop_length=hop_length)

    def librosa_frame():
        return librosa.util.frame(
            dummy_audio,
            frame_length=win_length,
            hop_length=hop_length,
            axis=-1,
            writeable= False,
            subok=False,
        )

    res_frame_1 = frame_1()
    res_librosa_frame = librosa_frame()
    np.testing.assert_equal(res_frame_1, res_librosa_frame)

    t1 = timeit.timeit(frame_1, number=100)
    t2 = timeit.timeit(librosa_frame, number=100)
    winner = 'Custom'
    if t2 < t1:
        winner = 'Librosa'

    print()
    print("="*50)
    print(f"Frame {res_frame_1.shape} ({res_frame_1.dtype}) Time:", t1)
    print(f"Librosa Frame {res_librosa_frame.shape} ({res_librosa_frame.dtype}) Time:", t2)
    print(">> Winner:", winner)
    print("="*50)


def test_compare_autocorrelate(make_audio):
    from iantirta.audio import load_audio
    wav_path = Path(__file__).parent / 'separated.wav'

    sr, c = 16000, 1
    wav, sr = load_audio(wav_path, sampling_rate=sr, num_channels=c)

    dummy_audio = wav.squeeze()

    hop_duration_ms: int = 16
    win_duration_ms: int = 64
    win_length = win_duration_ms * sr // 1000
    hop_length = hop_duration_ms * sr // 1000

    fmin = librosa.note_to_hz('C2')

    y = dummy_audio

    padding = [(0, 0) for _ in range(y.ndim)]
    padding[-1] = (int(win_length // 2), int(win_length // 2))
    y = np.pad(y, padding, mode='reflect')
    y_frames = frame(dummy_audio, frame_length=win_length, hop_length=hop_length)

    max_period = min(int(np.ceil(sr / fmin)), win_length - 1)

    custom_autocorrelate = autocorrelate(y_frames, max_size=max_period + 1, axis=-2)
    librosa_autocorrelate = librosa.core.audio.autocorrelate(
        y_frames, max_size=max_period + 1, axis=-2
    )

    np.testing.assert_equal(custom_autocorrelate, librosa_autocorrelate)
    # What about non-numba implementation


def test_compare_pyin():
    from iantirta.audio import load_audio
    wav_path = Path(__file__).parent / 'separated.wav'

    sr, c = 16000, 1
    wav, sr = load_audio(wav_path, sampling_rate=sr, num_channels=c)

    dummy_audio = wav.squeeze()

    hop_duration_ms: int = 16
    win_duration_ms: int = 64
    win_length = win_duration_ms * sr // 1000
    hop_length = hop_duration_ms * sr // 1000

    # fmin: float = 80
    # fmax: float = 7600
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')
    num_mel_bins: int = 80

    n_fft = spectrogram.optimal_fft_length(win_length)

    # Test Cumulative
    y = dummy_audio

    padding = [(0, 0) for _ in range(y.ndim)]
    padding[-1] = (int(win_length // 2), int(win_length // 2))
    y = np.pad(y, padding, mode='reflect')
    y_frames = frame(dummy_audio, frame_length=win_length, hop_length=hop_length)

    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), win_length - 1)

    custom_yin_frames = _cumulative_mean_normalized_difference(
        y_frames, min_period, max_period
    )
    librosa_yin_frames = librosa.core.pitch._cumulative_mean_normalized_difference(
        y_frames, min_period, max_period
    )

    np.testing.assert_equal(custom_yin_frames, librosa_yin_frames)

    def custom_pyin():
        return pyin(
            dummy_audio,
            sr=sr,
            fmin=fmin,
            fmax=fmax,
            frame_length=win_length,
            hop_length=hop_length,
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

    def librosa_pyin():
        return librosa.pyin(
            dummy_audio,
            sr=sr,
            fmin=fmin,
            fmax=fmax,
            frame_length=win_length,
            hop_length=hop_length,
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

    custom_f0, custom_voiced_flag, custom_voiced_probs = custom_pyin()
    librosa_f0, librosa_voiced_flag, librosa_voiced_probs = librosa_pyin()

    np.testing.assert_equal(custom_f0, librosa_f0)
    np.testing.assert_equal(custom_voiced_flag, librosa_voiced_flag)
    np.testing.assert_equal(custom_voiced_probs, librosa_voiced_probs)

    t1 = timeit.timeit(custom_pyin, number=1)
    t2 = timeit.timeit(librosa_pyin, number=1)
    winner = 'Custom'
    if t2 < t1:
        winner = 'Librosa'

    print()
    print("="*50)
    print(f"Custom Pyin Time:", t1)
    print(f"Librosa Pyin Time:", t2)
    print(">> Winner:", winner)
    print("="*50)


def test_compare_numba_vs_numpy_fallback(monkeypatch):
    rng = np.random.default_rng(42)
    dummy_data = rng.random((313, 199))
    axis = 0

    numba_funcs = {
        'localmin': {
            'x': dummy_data,
            'axis': axis
        },
        '_parabolic_interpolation': {
            'x': dummy_data,
            'axis': axis
        },
        'abs2': {
            'x': dummy_data.astype(np.complex128)
        },
    }
    numba_results = {}
    fallback_results = {}
    numba_times = {}
    fallback_times = {}

    importlib.reload(audio_mod)
    for name, params in numba_funcs.items():
        func = getattr(audio_mod, name)

        _ = func(**params)

        numba_results[name] = func(**params)
        numba_times[name] = timeit.timeit(lambda: func(**params), number=100)

    original_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("Simulated Numba import failure for testing fallback")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", mock_import)

    monkeypatch.delitem(sys.modules, "numba", raising=False)
    importlib.reload(audio_mod)

    for name, params in numba_funcs.items():
        func = getattr(audio_mod, name)
        fallback_results[name] = func(**params)
        fallback_times[name] = timeit.timeit(lambda: func(**params), number=100)

    print("\n" + "=" * 60)
    print(f"{'Function':<30} | {'Numba (s)':<12} | {'Fallback (s)':<12} | {'Winner':<10}")
    print("-" * 60)

    for name in numba_funcs:
        res_numba = numba_results[name]
        res_fallback = fallback_results[name]

        np.testing.assert_equal(res_numba, res_fallback)

        t_num = numba_times[name]
        t_fall = fallback_times[name]
        winner = 'Numba' if t_num < t_fall else 'Fallback'

        print(f"{name:<30} | {t_num:<12.5f} | {t_fall:<12.5f} | {winner:<10}")

    print("=" * 60)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        __file__
    ]))
