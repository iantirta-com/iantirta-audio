
[![PyPI](https://img.shields.io/pypi/v/iantirta-audio.svg)](https://pypi.org/project/iantirta-audio/)
[![Python](https://img.shields.io/pypi/pyversions/iantirta-audio.svg)](https://pypi.org/project/iantirta-audio/)
[![Tests](https://github.com/iantirta-com/iantirta-audio/actions/workflows/test.yml/badge.svg)](https://github.com/iantirta-com/iantirta-audio/actions/workflows/test.yml)
[![License](https://img.shields.io/github/license/iantirta-com/iantirta-audio.svg)](https://github.com/iantirta-com/iantirta-audio/blob/main/LICENSE)

[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://iantirta-com.github.io/iantirta-audio/)
[![GitHub](https://img.shields.io/badge/GitHub-iantirta--com-181717?logo=github)](https://github.com/iantirta-com/iantirta-audio)

## Support this project

If this library is useful in your work, consider supporting its
continued development and maintenance.

---

# iantirta-audio

Audio loading, encoding, and signal-analysis utilities for Python.

``iantirta-audio`` provides reusable utilities for decoding audio from media files,
exporting processed audio, and extracting time-frequency and vocal features.

The library is built around NumPy, with optional dependencies for audio encoding,
signal processing, and computational optimization.

# Features

Audio loading
---

- Load audio tracks from supported audio and video formats.
- Use ``torchcodec`` when available for supported media formats.
- Automatically select available decoding backends.
- Fall back to an ``FFmpeg``-based loader when necessary.
- Configure the output sampling rate and channel count.
- Retrieve audio as a NumPy array alongside its sampling rate.

Supported formats depend on the available decoder and its installed codec support.
Common formats include:

WAV, MP3, FLAC, AAC, M4A, MP4, MKV, MOV, AVI, and WebM.

Along with video like formats:

3gp, ac3, amr, flv, m4v, mpg, ogv, sox, ts, wma, wmv, wv


Audio encoding
---

Export audio to three formats:

| Format | Encoding | Configuration |
| ------ | -------- | ------------- |
| WAV    | Integer PCM or 32-bit floating point | 16-bit, 24-bit, or float32
| FLAC   | Lossless integer PCM compression | 16-bit or 24-bit
| MP3    | Lossy MPEG audio encoding| Configurable bitrate and quality

Additional controls include clipping prevention,
sample rate, bit depth, MP3 bitrate, and encoder quality.


Audio feature extraction
---

Signal-analysis utilities include:

- Short-Time Fourier Transform (STFT).
- Amplitude and power spectrograms.
- Mel and log-mel spectrograms.
- RMS energy.
- pYIN fundamental frequency estimation.
- Voiced/unvoiced classification.
- Voiced probability estimation.
- Boolean activity masks and audio segmentation.
- Time and frequency conversion utilities.


Computational optimization
---

Numba is supported as an optional dependency for
JIT compilation of selected computational routines.

The library remains usable without Numba,
using the regular Python/NumPy implementations.

---

# Installation

Install the base package:

```bash
pip install iantirta-audio
```


Optional dependencies
---

Install individual features as needed:

#### # MP3 encoding
```bash
pip install "iantirta-audio[mp3]"
```

#### # Audio feature extraction and signal processing
```bash
pip install "iantirta-audio[extraction]"
```

#### # Optional Numba optimization
```bash
pip install "iantirta-audio[optimization]"
```

#### # Combined MP3, extraction dependencies, and loading audio optional deps
```bash
pip install "iantirta-audio[all]"
```


The available extras are:
---

| Extra | Dependency | Purpose |
| ----- | ---------- | ------- |
| ``mp3`` | ``lameenc``    | MP3 encoding |
| ``extraction`` | ``scipy`` | Signal processing and feature extraction |
| ``optimization`` | ``numba`` | Optional JIT optimization |
| ``all`` | mp3, extraction, ``torchcodec``, ``torchaudio`` | Combined optional features |
| ``docs`` | ``mkdocs-material`` | Documentation development |
| ``test`` | ``pytest``, ``iantirta-media``, optimization | Testing |
| ``dev`` | test, all | Development environment |

Numba is not included in the current ``all`` extra. Install ``optimization`` separately when needed.


Runtime requirements
---

Some functionality requires dependencies that are not installed by the base package:

| Dependency | Required for |
| ---------- | ------------ |
| FFmpeg and ffprobe | Built-in decoding fallback and FLAC encoding |
| SciPy | Audio feature extraction |
| Numba | Optional computational optimization |
| lameenc | MP3 encoding |

FFmpeg must be installed separately and
accessible through the system ``PATH``.

TorchCodec is optional. A compatible PyTorch
installation is required when using TorchCodec.

---

# Examples

Audio loading
---

Use ``load_audio`` to decode an audio file into a NumPy array.

```py
from iantirta.audio import load_audio

wav, sr = load_audio(
    "input.mp4",
    sampling_rate=16000,
    num_channels=1,
)

print(wav.shape)
# (1, <sample_size>)
print(sr)
# 16000
```

The returned waveform uses a channel-first layout:

`(channels, samples)`

The ``sampling_rate`` and ``num_channels`` arguments
control the output audio.

- `sampling_rate=None` preserves the original sampling rate.
- `num_channels=1` converts the audio to mono.
- `backend="auto"` selects the available decoding backend.

The supported backend selectors include:

#### # Automatic backend selection
```py
wav, sr = load_audio("input.wav", backend="auto")
```

#### # TorchCodec
```py
wav, sr = load_audio("input.mp4", backend="torchcodec")
```

#### # Built-in FFmpeg loader
```py
wav, sr = load_audio("input.wav", backend="audioloader")
```

The automatic backend can use TorchCodec or librosa when available, with the built-in FFmpeg loader as a fallback.

The built-in loader also provides lower-level
stream selection and audio metadata through ``AudioLoader``.

---

Audio saving
---

Use ``save_audio`` to export NumPy arrays or PyTorch tensors.

#### - WAV

Save as integer PCM:

```py
from iantirta.audio import save_audio

save_audio(
    wav,
    "output.wav",
    samplerate=sr,
    bits_per_sample=24,
)
```

Save as 32-bit floating-point WAV:

```py
save_audio(
    wav,
    "output_float.wav",
    samplerate=sr,
    as_float=True,
)
```

Floating-point WAV avoids integer PCM quantization
and is suitable when preserving in-range float32 sample values matters.

The writer converts the data to float32 and clamps samples to the range `[-1, 1]`.


#### - FLAC

```py
save_audio(
    wav,
    "output.flac",
    samplerate=sr,
    bits_per_sample=24,
)
```

FLAC uses FFmpeg for encoding.
The current implementation supports 16-bit and 24-bit integer PCM input.

Although FLAC compression is lossless, converting floating-point samples to integer PCM can introduce quantization before encoding.


#### - MP3

```py
save_audio(
    wav,
    "output.mp3",
    samplerate=sr,
    bitrate=320,
    preset=2,
)
```

MP3 encoding uses `lameenc`.

Parameters:

- `bitrate`: target encoding bitrate in kbps.
- `preset`: encoder quality setting from `2` to `7`.
  - `2`: highest quality setting.
  - `7`: fastest encoding setting.

Install the `mp3` extra to enable MP3 encoding.


### Clipping control

``save_audio`` provides several strategies for
handling samples outside the normal audio range:

| Mode | Behavior |
| ---- | -------- |
| `rescale` | Rescales the waveform when its peak exceeds 1 `(default)` |
| `clamp` | Clamps samples to approximately -0.99 to 0.99 |
| `tanh` | Applies the hyperbolic tangent function |
| `none` | Skips the initial clipping-prevention step |

The default is `rescale`.

Note that the individual encoders also constrain
samples to their supported output range.

---

Audio feature extraction
---

``AudioExtraction`` provides a collection of features
for analyzing mono audio, particularly isolated speech
and vocal recordings.

Install the extraction dependency first:

```py
pip install "iantirta-audio[extraction]"
```

### Basic usage

```py
from iantirta.audio import load_audio, AudioExtraction

wav, sr = load_audio(
    "vocals.wav",
    sampling_rate=16000,
    num_channels=1,
)

features = AudioExtraction(wav, sr)
```

The input should contain a single mono channel.


### pYIN pitch estimation

```py
f0 = features.f0

voiced_flag = features.voiced_flag

voiced_probs = features.voiced_probs
```

The pYIN implementation provides:

| Property | Description |
| -------- | ----------- |
| `f0` | Estimated fundamental frequency in Hz |
| `voiced_flag` | Boolean voiced/unvoiced classification |
| `voiced_probs` | Estimated probability of voiced frames |
| `refined_voiced_flag` | Refined voiced/unvoiced mask |

Unvoiced frames can contain `NaN` values in the fundamental frequency output.


### RMS and Mel features

```py
rms = features.rms

mel = features.mel_spectrogram

rms_values = rms.arr

mel_values = mel.S
```

Both feature objects also provide smoothing,
thresholding, and mask-related utilities.

The RMS feature describes short-term signal energy.

The Mel feature represents the distribution of
spectral energy across Mel frequency bands.


### Activity masks and segmentation

```py
activity_mask = features.activity_mask

segments = features.segments

for segment in segments:
    print(
        segment.start,
        segment.end,
        segment.duration,
    )
    # 2.0 5.0 3.0
```

The activity mask is derived from RMS and Mel energy features.

The ``segments`` property uses the refined pYIN
voiced mask to generate time-based ``AudioSegment`` objects.

Mask refinement includes merging short gaps,
removing short regions, and splitting sufficiently
long regions using duration-based heuristics.

The default analysis configuration uses
a 16 ms hop duration and a 64 ms analysis window.


### Convenience function

For direct analysis of an audio file:

```py
from iantirta.audio import detect_audio_activity

features = detect_audio_activity(
    "vocals.wav",
    sr=16000,
)

segments = features.segments
```

This function loads the audio as mono,
resamples it to the requested sample rate,
and returns the corresponding ``AudioExtraction`` object.


### Processing considerations

The feature extraction and segmentation routines
are primarily intended for isolated audio.

They are not designed to replace a dedicated
denoising or source-separation system.

Strong background noise, overlapping speakers,
instrumental accompaniment, and reverberation
can affect pitch estimation, voiced probabilities,
and segmentation accuracy.

---

Spectrogram utilities
---

The spectrogram module provides configurable
Short-Time Fourier Transform processing and decibel conversion.

```py
from iantirta.audio.extraction_utils.spectrogram import (
    spectrogram,
    window_function,
    amplitude_to_db,
    power_to_db,
    mel_filter_bank,
)
```

### Spectrogram types

The ``spectrogram`` function supports the following output types:

| Output | Configuration |
| ------ | ------------- |
| Amplitude spectrogram | `power=1.0` |
| Power spectrogram | `power=2.0` |
| Complex-valued spectrogram | `power=None` |
| Log spectrogram | Set `log_mel` |
| Mel spectrogram | Provide `mel_filters` |
| Log-Mel spectrogram | Provide `mel_filters` and `log_mel` |

Available logarithmic conversion modes are:

- `"log"`: natural logarithm.
- `"log10"`: base-10 logarithm.
- `"dB"`: decibel conversion.


### Example: Mel spectrogram

```py
from iantirta.audio.extraction_utils.spectrogram import (
    spectrogram,
    window_function,
    mel_filter_bank,
)

frame_length = 1024
fft_length = 1024
hop_length = 256

window = window_function(
    window_length=frame_length,
    name="hann",
)

mel_filters = mel_filter_bank(
    num_frequency_bins=fft_length // 2 + 1,
    num_mel_filters=80,
    min_frequency=80,
    max_frequency=7600,
    sampling_rate=sr,
    norm="slaney",
    mel_scale="slaney",
)

mel_spec = spectrogram(
    waveform=wav[0],
    window=window,
    frame_length=frame_length,
    hop_length=hop_length,
    fft_length=fft_length,
    power=2.0,
    mel_filters=mel_filters,
    log_mel="dB",
)
```

The returned feature matrix has shape:

`(num_frequency_bins, num_frames)`

For Mel spectrograms, the first dimension
corresponds to the number of Mel filters.

### Decibel conversion

Two utilities are provided for converting spectrogram values to decibels.

```py
from iantirta.audio.extraction_utils.spectrogram import (
    amplitude_to_db,
    power_to_db,
)

amplitude_db = amplitude_to_db(
    amplitude_spectrogram,
    reference=1.0,
    db_range=80,
)

power_db = power_to_db(
    power_spectrogram,
    reference=1.0,
    db_range=80,
)
```

Use `amplitude_to_db` for amplitude values
and `power_to_db` for squared-magnitude power values.

Both utilities support reference levels, minimum values, and dynamic range limits.

---

Development
---

Clone the repository:

```bash
git clone https://github.com/iantirta-com/iantirta-audio.git

cd iantirta-audio
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

Tests requiring external services or real media downloads are marked as `manual`.

---

Support
---

For bug reports, feature requests,
or implementation questions, use the `GitHub issue tracker`
[GitHub issue tracker](https://github.com/iantirta-com/iantirta-audio/issues).

For other enquiries, contact:

[ian@iantirta.com](mailto:ian@iantirta.com)


License
---

This project is licensed under the Apache License 2.0.

See [LICENSE](LICENSE) for the full license text.


Project
---

Maintained by [Iantirta.com](https://iantirta.com).

[GitHub Repository](https://github.com/iantirta-com/iantirta-audio) · [PyPI Package](https://pypi.org/project/iantirta-audio/) · [Changelog](https://github.com/iantirta-com/iantirta-audio/blob/main/CHANGELOG.md)