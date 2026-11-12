
from pathlib import Path

import pytest

from iantirta.audio.audio_extraction import AudioExtraction


def test_audio_extraction_all():
    """
    test audio extraction needed for making `audio_segments`
    """
    from iantirta.audio import load_audio
    wav_path = Path(__file__).parent.parent / "tests" / 'separated.wav'

    sr, c = 16000, 1
    wav, sr = load_audio(wav_path, sampling_rate=sr, num_channels=c)

    audio_metadata = AudioExtraction(wav, sr)
    print(audio_metadata.voiced_flag)
    

    # import librosa
    # import matplotlib.pyplot as plt
    # # Create a figure with 3 stacked subplots that share the same time axis
    # fig, (ax_spec, ax_prob, ax_flag) = plt.subplots(
    #     nrows=3, 
    #     sharex=True, 
    #     figsize=(10, 8),
    #     gridspec_kw={'height_ratios': [3, 1, 1]} # Make the spectrogram taller
    # )
    # times = librosa.times_like(audio_metadata.f0)

    # # Plot f0 on top of the spectrogram
    # ax_spec.plot(times, audio_metadata.f0, label='f0', color='cyan', linewidth=2, path_effects=librosa.display.highlight(ax=ax_spec))
    # ax_spec.legend(loc='upper right')
    
    # # --- Middle Plot: Voiced Probabilities ---
    # ax_prob.plot(times, audio_metadata.voiced_probs, color='orange')
    # ax_prob.set(title='Voicing Probability', ylabel='Probability', ylim=(0, 1.1))
    # ax_prob.grid(True)
    
    # # --- Bottom Plot: Voiced Flag (Binary) ---
    # # Convert boolean to integer (0 or 1) for plotting
    # ax_flag.step(times, audio_metadata.refined_voiced_flag.astype(int), color='green', where='post')
    # ax_flag.set(title='Voiced Flag', ylabel='True/False', xlabel='Time (s)', yticks=[0, 1])
    # ax_flag.grid(True)
    
    # plt.tight_layout()
    # plt.show()

    for aseg in audio_metadata.segments:
        print(aseg.start, aseg.end, f"Duration ({aseg.end - aseg.start})")
    print(audio_metadata.segments)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        __file__
    ]))
