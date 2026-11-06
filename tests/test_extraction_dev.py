import timeit
from pathlib import Path

import librosa
import numpy as np
import pytest
import scipy.fft

from iantirta.audio.audio_extraction_utils import stft, frame, _cumulative_mean_normalized_difference, fast_merge_lower_duration, mask_merge_lower_duration
from iantirta.audio.extraction_utils import spectrogram
from iantirta.audio.audio_extraction import AudioExtraction


@pytest.mark.manual
def test_compare_speed_scipy_numpy(make_audio):
    sr, c = 16000, 1
    dummy_audio = make_audio(sr=sr, channels=c)
    dummy_audio = dummy_audio.squeeze()

    hop_duration_ms: int = 16
    win_duration_ms: int = 64
    win_length = win_duration_ms * sr // 1000
    hop_length = hop_duration_ms * sr // 1000

    n_fft = spectrogram.optimal_fft_length(win_length)

    win_function: str = "hann_window"
    window = spectrogram.window_function(window_length=win_length, name=win_function, periodic=True)

    padding = [(int(win_length // 2), int(win_length // 2))]
    waveform = np.pad(dummy_audio, padding, mode='reflect')

    buffer = np.zeros(n_fft)
    buffer[:win_length] = waveform[0 : 0 + win_length]

    buffer[:win_length] *= window

    def nprfft():
        return np.fft.rfft(buffer)

    def scipyrfft():
        return scipy.fft.rfft(buffer, workers=-1)

    res_nprfft = nprfft()
    res_scipyrfft = scipyrfft()
    np.testing.assert_almost_equal(res_nprfft, res_scipyrfft)

    t1 = timeit.timeit(nprfft, number=1000)
    t2 = timeit.timeit(scipyrfft, number=1000)
    winner = 'numpy'
    if t2 < t1:
        winner = 'scipy'

    print()
    print("="*50)
    print(f"RFFT {res_nprfft.shape} ({res_nprfft.dtype}) Time:", t1)
    print(f"Scipy RFFT {res_scipyrfft.shape} ({res_scipyrfft.dtype}) Time:", t2)
    print(">> Winner:", winner)
    print("="*50)


try:
    import numba


    @numba.stencil  # type: ignore
    def _pi_stencil(x: np.ndarray) -> np.ndarray:
        """Stencil to compute local parabolic interpolation"""
        a = x[1] + x[-1] - 2 * x[0]
        b = (x[1] - x[-1]) / 2

        if np.abs(b) >= np.abs(a):
            # If this happens, we'll shift by more than 1 bin
            # Suppressing types because mypy has no idea about stencils
            return 0  # type: ignore

        return -b / a  # type: ignore


    def _py_pi_stencil(x: np.ndarray) -> np.ndarray:
        """Stencil to compute local parabolic interpolation"""
        out = np.zeros_like(x)

        x_prev = x[..., :-2]
        x_curr = x[..., 1:-1]
        x_next = x[..., 2:]

        a = x_next + x_prev - 2 * x_curr
        b = (x_next - x_prev) / 2

        valid_mask = (np.abs(b) < np.abs(a)) & (a != 0)

        np.divide(-b, a, out=out[..., 1:-1], where=valid_mask)

        return out


    @numba.guvectorize(
        "(n)->(n)",
        cache=True,
        nopython=True,
    )  # type: ignore
    def _pi_wrapper(x: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover
        """Vectorized wrapper for the parabolic interpolation stencil"""
        y[:] = _pi_stencil(x)


    @pytest.mark.manual
    def test_numba_stencil(make_audio):
        sr, c = 16000, 1
        dummy_audio = make_audio(sr=sr, channels=c)
        dummy_audio = dummy_audio.squeeze()
        dummy_audio = dummy_audio.astype(np.float64)

        hop_duration_ms: int = 16
        win_duration_ms: int = 64
        win_length = win_duration_ms * sr // 1000
        hop_length = hop_duration_ms * sr // 1000
        fmin, fmax = 80, 7600

        y = dummy_audio
        padding = [(0, 0) for _ in y.shape]
        padding[-1] = (win_length // 2, win_length // 2)
        y = np.pad(y, padding, mode='reflect')

        # Frame audio.
        y_frames = frame(y, frame_length=win_length, hop_length=hop_length)

        # Calculate minimum and maximum periods
        min_period = int(np.floor(sr / fmax))
        max_period = min(int(np.ceil(sr / fmin)), win_length - 1)

        # Calculate cumulative mean normalized difference function.
        yin_frames = _cumulative_mean_normalized_difference(
            y_frames, min_period, max_period
        )

        axis = -2
        xi = yin_frames.swapaxes(-1, axis)

        def guvectorize():
            out = _py_pi_stencil(xi)
            return out.swapaxes(-1, axis)

        def librosa_guvectorize():
            shifts = np.empty_like(yin_frames)
            shiftsi = shifts.swapaxes(-1, axis)

            _pi_wrapper(xi, shiftsi)

            shiftsi[..., -1] = 0
            shiftsi[..., 0] = 0
    
            return shifts

        res_guvectorize = guvectorize()
        res_librosa_guvectorize = librosa_guvectorize()
        np.testing.assert_equal(res_guvectorize, res_librosa_guvectorize)

        t1 = timeit.timeit(guvectorize, number=100)
        t2 = timeit.timeit(librosa_guvectorize, number=100)
        winner = 'custom'
        if t2 < t1:
            winner = 'Numba'

        print()
        print("="*50)
        print(f"Stencil {res_guvectorize.shape} ({res_guvectorize.dtype}) Time:", t1)
        print(f"Numba Stencil {res_librosa_guvectorize.shape} ({res_librosa_guvectorize.dtype}) Time:", t2)
        print(">> Winner:", winner)
        print("="*50)

except:
    pass



@pytest.mark.manual
def non_test_compare_pyin():
    from iantirta.audio import load_audio
    wav_path = Path(__file__).parent / 'separated.wav'

    sr, c = 16000, 1
    wav, sr = load_audio(wav_path, sampling_rate=sr, num_channels=c)

    dummy_audio = wav.squeeze()

    hop_duration_ms: int = 16
    win_duration_ms: int = 64
    win_length = win_duration_ms * sr // 1000
    hop_length = hop_duration_ms * sr // 1000

    fmin: float = 80
    fmax: float = 7600
    num_mel_bins: int = 80

    n_fft = spectrogram.optimal_fft_length(win_length)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        dummy_audio,
        sr=sr,
        # fmin=fmin, # librosa.note_to_hz('C2'),
        # fmax=fmax, #librosa.note_to_hz('C7'),
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
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
    times = librosa.times_like(
        f0,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        axis=-1,
    )

    print(f0.shape, f0.dtype)
    print(voiced_flag.shape, voiced_flag.dtype)
    print(voiced_probs.shape, voiced_probs.dtype)

    win_function: str = "hann_window"
    window = spectrogram.window_function(window_length=win_length, name=win_function, periodic=True)

    import matplotlib.pyplot as plt
    D = librosa.amplitude_to_db(np.abs(stft(
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
    )), ref=np.max)
    
    fig, (ax_spec, ax_prob, ax_flag) = plt.subplots(
        nrows=3, 
        sharex=True, 
        figsize=(10, 8),
        gridspec_kw={'height_ratios': [3, 1, 1]} # Make the spectrogram taller
    )

    img = librosa.display.specshow(
        D, x_axis='time', y_axis='log', ax=ax_spec,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        win_length= win_length,
        fmin=fmin,
        fmax=fmax,
        top_db=None,
    )
    ax_spec.set(title='pYIN Fundamental Frequency Estimation', ylabel='Frequency (Hz)')
    fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")

    ax_spec.plot(times, f0, label='f0', color='cyan', linewidth=2, path_effects=librosa.display.highlight(ax=ax_spec))
    ax_spec.legend(loc='upper right')

    ax_prob.plot(times, voiced_probs, color='orange')
    ax_prob.set(title='Voicing Probability', ylabel='Probability', ylim=(0, 1.1))
    ax_prob.grid(True)

    ax_flag.step(times, voiced_flag.astype(int), color='green', where='post')
    ax_flag.set(title='Voiced Flag', ylabel='True/False', xlabel='Time (s)', yticks=[0, 1])
    ax_flag.grid(True)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        __file__
    ]))
