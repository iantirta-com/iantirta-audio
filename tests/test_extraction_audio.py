# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

import pytest

from iantirta.audio import load_audio
from iantirta.audio.audio_extraction import AudioExtraction
from iantirta.audio.audio_extraction_utils import times_like


class TestAudioExtraction:

    def plot_extration(self):
        import librosa
        import matplotlib.pyplot as plt
        # Create a figure with 3 stacked subplots that share the same time axis
        fig, (ax_spec, ax_prob, ax_flag) = plt.subplots(
            nrows=3, 
            sharex=True, 
            figsize=(10, 8),
            gridspec_kw={'height_ratios': [3, 1, 1]} # Make the spectrogram taller
        )
        times = times_like(
            self.aad.f0
        )
    
        # Plot f0 on top of the spectrogram
        ax_spec.plot(times, self.aad.f0, label='f0', color='cyan', linewidth=2, path_effects=librosa.display.highlight(ax=ax_spec))
        ax_spec.legend(loc='upper right')
        
        # --- Middle Plot: Voiced Probabilities ---
        ax_prob.plot(times, self.aad.voiced_probs, color='orange')
        ax_prob.set(title='Voicing Probability', ylabel='Probability', ylim=(0, 1.1))
        ax_prob.grid(True)
        
        # --- Bottom Plot: Voiced Flag (Binary) ---
        # Convert boolean to integer (0 or 1) for plotting
        ax_flag.step(times, self.aad.refined_voiced_flag.astype(int), color='green', where='post')
        ax_flag.set(title='Voiced Flag', ylabel='True/False', xlabel='Time (s)', yticks=[0, 1])
        ax_flag.grid(True)
        
        plt.tight_layout()
        plt.show()

    def test_audio_extraction_all(self, separated_path):
        self.sr = 16000
        self.channels = 1
        wav, sr = load_audio(
            separated_path,
            sampling_rate=self.sr,
            num_channels=self.channels
        )
        *shape, C, T = wav.shape
        assert sr == self.sr
        assert C == self.channels
        self.wav = wav
        self.aad = AudioExtraction(wav, sr)

        # self.plot_extration()

        for aseg in self.aad.segments:
            print(aseg.start, aseg.end, f"Duration ({aseg.end - aseg.start})")
        print(self.aad.segments)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        __file__
    ]))
