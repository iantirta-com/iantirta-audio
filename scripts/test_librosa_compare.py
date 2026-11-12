# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

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
    times_like,
    midi_to_hz
)
from iantirta.audio.extraction_utils import spectrogram


class TestCompareLibrosa:
    """
    This unit test purposes is to help integrate
    and migrating the use of librosa.
    """

    @pytest.fixture(autouse=True)
    def setup_class_vars(self, make_audio):
        self.sr = 16000
        self.channels = 1
        self.wav = make_audio(
            sr=self.sr,
            channels=self.channels
        )
        
        *shape, C, T = self.wav.shape
        assert 16000 == self.sr
        assert C == self.channels
        
        self.samples = self.wav.squeeze()

        self.hop_duration_ms: int = 16
        self.win_duration_ms: int = 64
        self.win_length = self.win_duration_ms * self.sr // 1000
        self.hop_length = self.hop_duration_ms * self.sr // 1000

        self.n_fft = spectrogram.optimal_fft_length(self.win_length)

        win_function: str = "hann_window"
        self.window = spectrogram.window_function(window_length=self.win_length, name=win_function, periodic=True)

        self.fmin: float = 80
        self.fmax: float = 7600
        self.num_mel_bins: int = 80

    def setup_class_separated_vars(self, separated_path):
        """
        Helper to setup sample using actual wav vocals
        """
        from iantirta.audio import load_audio
        wav, sr = load_audio(
            separated_path,
            sampling_rate=self.sr,
            num_channels=self.channels
        )
        *shape, C, T = wav.shape
        assert sr == self.sr
        assert C == self.channels
        self.wav = wav
        self.samples = wav.squeeze()

    def calculate_compare_value(self, fn1, fn2, num: int = 100):
        """
        Helper method to calculate the time it took between both
        """
        t1 = timeit.timeit(fn1, number=num)
        t2 = timeit.timeit(fn2, number=num)

        fn1_str = fn1.__name__.title()
        fn2_str = fn2.__name__.title()

        winner = fn1_str if t1 < t2 else fn2_str

        print("="*50)
        print(f"{fn1_str:<30} | {t1:<12.5f} | {winner:<10}")
        print(f"{fn2_str:<30} | {t2:<12.5f} | {winner:<10}")
        print("="*50)

    def assert_compare(self, fn1, fn2, corr: bool = False, decimal: int = 7):
        res_1 = fn1()
        res_2 = fn2()
        if not corr:
            np.testing.assert_equal(res_1, res_2)
        else:
            import scipy.signal
            corr = scipy.signal.correlate(res_1, res_2)
            print(f">> Correlation between res: {np.max(corr)}")
            np.testing.assert_almost_equal(res_1, res_2, decimal=decimal)

    def test_compare_rms(self):
        """
        Comparing librosa and our implementation.
        """
        def custom_rms():
            return rms(
                self.samples,
                frame_length=self.win_length,
                hop_length=self.hop_length
            )

        def librosa_rms():
            return librosa.feature.rms(
                y=self.samples,
                frame_length=self.win_length,
                hop_length=self.hop_length,
                pad_mode='reflect',
                dtype=np.float64
            )

        self.assert_compare(custom_rms, librosa_rms)
        self.calculate_compare_value(custom_rms, librosa_rms)

    def test_compare_stft(self):
        """
        Comparing stft
        """
        def custom_stft():
            return stft(
                waveform=self.samples,
                window=self.window,
                frame_length=self.win_length,
                hop_length=self.hop_length,
                fft_length=self.n_fft,
                center=True,
                pad_mode="reflect",
                onesided=True,
                dither=0.0,
                preemphasis=None,
                remove_dc_offset=False,
            )

        def librosa_stft():
            return librosa.stft(
                y=self.samples,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                dtype=None,
                pad_mode="reflect",
                out=None,
            )

        self.assert_compare(custom_stft, librosa_stft, corr=True)
        self.calculate_compare_value(custom_stft, librosa_stft)

    def test_compare_mel_spectrogram(self):
        """
        Comparing Mel Spectrogram
        """
        def custom_melspec():
            return melspectrogram(
                waveform=self.samples,
                window=self.window,
                frame_length=self.win_length,
                hop_length=self.hop_length,
                fft_length=self.n_fft,
                power=2.0,
                log_mel=None,
                sampling_rate=self.sr,
                fmin=self.fmin,
                fmax=self.fmax,
                num_mel_filters=self.num_mel_bins,
            )

        def librosa_melspec():
            kwargs = {
                'n_mels': self.num_mel_bins,
                'fmin': self.fmin,
                'fmax': self.fmax,
                'htk': False,
                'norm': 'slaney',
                'dtype': np.float64
            }
            return librosa.feature.melspectrogram(
                y=self.samples,
                sr=self.sr,
                S=None,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window="hann",
                center=True,
                pad_mode="reflect",
                power=2.0,
                **kwargs,
            )

        self.assert_compare(custom_melspec, librosa_melspec, corr=True, decimal=4)
        self.calculate_compare_value(custom_melspec, librosa_melspec)

        reference: float = 1.0
        min_value: float = 1e-10
        db_range: float | None = None

        custom_melspec = custom_melspec()
        librosa_melspec = librosa_melspec()

        def custom_db_mel():
            return spectrogram.power_to_db(
                custom_melspec,
                reference=reference,
                min_value=min_value,
                db_range=db_range,
            )

        def librosa_db_mel():
            return librosa.power_to_db(
                librosa_melspec,
                ref=reference, # 1.0,
                amin=min_value, # 1e-10,
                top_db=db_range, # 80.0,
                axes="auto",
            )

        self.assert_compare(custom_db_mel, librosa_db_mel, corr=True, decimal=4)
        self.calculate_compare_value(custom_db_mel, librosa_db_mel)

    def test_compare_util_frame(self):
        def custom_frame():
            return frame(
                self.samples,
                frame_length=self.win_length,
                hop_length=self.hop_length)

        def librosa_frame():
            return librosa.util.frame(
                self.samples,
                frame_length=self.win_length,
                hop_length=self.hop_length,
                axis=-1,
                writeable= False,
                subok=False,
            )

        self.assert_compare(custom_frame, librosa_frame)
        self.calculate_compare_value(custom_frame, librosa_frame)

    def test_compare_times_like(self):
        """
        Test times like
        """
        def custom_times():
            return times_like(
                x=self.wav,
                sr=self.sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
            )

        def librosa_times():
            return librosa.times_like(
                X=self.wav,
                sr=self.sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
            )

        self.assert_compare(custom_times, librosa_times)
        self.calculate_compare_value(custom_times, librosa_times)

    @pytest.mark.parametrize(
        ("note", "midi"),
        [
            ("C2", 36),
            ("C7", 96)
        ]
    )
    def test_compare_midi_to_hz(self, note, midi):
        """
        With librosa we use note as ``C2``.
        but with ours we use midi int value ``36``
        """
        def custom_midi_hz():
            return midi_to_hz(midi)

        def librosa_midi_hz():
            return librosa.note_to_hz(note)

        self.assert_compare(custom_midi_hz, librosa_midi_hz)
        self.calculate_compare_value(custom_midi_hz, librosa_midi_hz)

    @pytest.mark.parametrize(
        ("midi"),
        [
            [36, 96]
        ]
    )
    def test_compare_midi_to_hz_multi(self, midi):
        hzs = midi_to_hz(midi)
        assert len(hzs) == len(midi)
    
    def _setup_y_frames(self):
        self.fmin = librosa.note_to_hz("C2")
        self.fmax = librosa.note_to_hz('C7')

        y = self.samples

        padding = [(0, 0) for _ in range(y.ndim)]
        padding[-1] = (int(self.win_length // 2), int(self.win_length // 2))

        y = np.pad(y, padding, mode='reflect')

        self.y_frames = frame(
            self.samples,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )

        self.min_period = int(np.floor(self.sr / self.fmax))
        self.max_period = min(int(np.ceil(self.sr / self.fmin)), self.win_length - 1)

    def test_compare_autocorrelate(self, separated_path):
        self.setup_class_separated_vars(separated_path)
        self._setup_y_frames()

        def custom_autocorrelate():
            return autocorrelate(
                self.y_frames, max_size=self.max_period + 1, axis=-2
            )
        def librosa_autocorrelate():
            return librosa.core.audio.autocorrelate(
                self.y_frames, max_size=self.max_period + 1, axis=-2
            )

        self.assert_compare(custom_autocorrelate, librosa_autocorrelate)
        self.calculate_compare_value(custom_autocorrelate, librosa_autocorrelate, num=1)

    def test_compare_pyin(self, separated_path):
        self.setup_class_separated_vars(separated_path)
        self._setup_y_frames()

        def custom_yin_frames():
            return _cumulative_mean_normalized_difference(
                self.y_frames, self.min_period, self.max_period
            )
        def librosa_yin_frames():
            return librosa.core.pitch._cumulative_mean_normalized_difference(
                self.y_frames, self.min_period, self.max_period
            )

        self.assert_compare(custom_yin_frames, librosa_yin_frames)
        self.calculate_compare_value(custom_yin_frames, librosa_yin_frames, num=1)

        def custom_pyin():
            return pyin(
                self.samples,
                sr=self.sr,
                fmin=self.fmin,
                fmax=self.fmax,
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

        def librosa_pyin():
            return librosa.pyin(
                self.samples,
                sr=self.sr,
                fmin=self.fmin,
                fmax=self.fmax,
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

        custom_res_pyin = custom_pyin()
        custom_librosa_pyin = librosa_pyin()
        for (res1, res2) in zip(custom_res_pyin, custom_librosa_pyin):
            np.testing.assert_equal(res1, res2)

        self.calculate_compare_value(custom_pyin, librosa_pyin, num=1)

    def _disable_numba(self, monkeypatch):
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "numba" or name.startswith("numba."):
                raise ImportError("Simulated Numba import failure for testing fallback")
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)

        monkeypatch.delitem(sys.modules, "numba", raising=False)
        importlib.reload(audio_mod)

    def test_compare_numba_vs_numpy(self, monkeypatch):
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
        numpy_results = {}
        numba_times = {}
        numpy_times = {}

        importlib.reload(audio_mod)
        for name, params in numba_funcs.items():
            func = getattr(audio_mod, name)

            _ = func(**params)

            numba_results[name] = func(**params)
            numba_times[name] = timeit.timeit(lambda: func(**params), number=100)

        self._disable_numba(monkeypatch)
        for name, params in numba_funcs.items():
            func = getattr(audio_mod, name)
            numpy_results[name] = func(**params)
            numpy_times[name] = timeit.timeit(lambda: func(**params), number=100)

        print("\n" + "=" * 60)
        print(f"{'Function':<30} | {'Numba (s)':<12} | {'Numpy (s)':<12} | {'Winner':<10}")
        print("-" * 60)

        for name in numba_funcs:
            res_numba = numba_results[name]
            res_fallback = numpy_results[name]

            np.testing.assert_equal(res_numba, res_fallback)

            t_num = numba_times[name]
            t_fall = numpy_times[name]
            winner = 'Numba' if t_num < t_fall else 'Fallback'

            print(f"{name:<30} | {t_num:<12.5f} | {t_fall:<12.5f} | {winner:<10}")

        print("=" * 60)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    conftest_path = (Path(__file__).parent.parent / "tests").resolve()
    
    if str(conftest_path) not in sys.path:
        sys.path.insert(0, str(conftest_path))

    sys.exit(pytest.main([
        '-v', '-s',
        '-p', 'conftest',
        __file__
        + "::TestCompareLibrosa::test_compare_midi_to_hz_multi"
    ]))
