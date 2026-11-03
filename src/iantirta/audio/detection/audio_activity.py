


# def detect_audio_activiy(
#     audio: str | Path | np.ndarray,
#     sr: int = 16000,
# ) -> AcvitiyDetectionMetadata:
#     import scipy.signal
#     wav, sr = load_audio(audio, sampling_rate=sr, num_channels=1)

#     sos = scipy.signal.butter(10, [200, 5000], btype='bandpass', fs=sr, output='sos')
#     wav = scipy.signal.sosfilt(sos, wav).astype(np.float32)

#     return AcvitiyDetectionMetadata(wav, sr=sr)


if __name__ == '__main__':
    from pathlib import Path
    from iantirta.audio import load_audio
    from rich import inspect as i
    audio_path = str(Path("D:/IANTIRTACOM/dataset/vocals.wav").expanduser().resolve())
    # metadata = detect_audio_activiy(audio_path)
    # i(metadata)
    # print(metadata.final_mask)
    from IPython.display import display, Audio
    wav, sr = load_audio(audio_path)
    display(Audio(wav, rate=sr))
    from huggingface_hub import upload_file