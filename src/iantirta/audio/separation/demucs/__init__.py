# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from iantirta.audio.separation.base import Separator, SeparationOptions, SeparationResult

__all__ = [
    "load_demucs_model"
]


@dataclass(frozen=True, slots=True)
class DemucsOptions(SeparationOptions):
    model_name: str = "mdx_extra_q"
    shifts: int = 1
    overlap: float = 0.25
    split: bool = True,
    segment: float | None = None

    progress: bool = False,
    callback: t.Callable[[dict], None] | None = None,
    callback_arg: dict | None = None,

    num_workers: int = 0


class DemucsSeparator(Separator):
    def __init__(self, options: DemucsOptions):
        self.config = options
        self._load_model(options.model_name)

    def _load_model(self, name: str):
        self._model = load_demucs_model(name=name)
        if self._model is None:
            raise RuntimeError("Failed to load model")
        self._audio_channels = self._model.audio_channels
        self._samplerate = self._model.samplerate

    def separate_tensor(self, wav: torch.Tensor, sr: int | None = None):
        """
        Separate a loaded tensor.

        Parameters
        ----------
        wav: Waveform of the audio. Should have 2 dimensions, the first is each audio channel, \
            while the second is the waveform of each channel. Type should be float32. \
            e.g. `tuple(wav.shape) == (2, 884000)` means the audio has 2 channels.
        sr: Sample rate of the original audio, the wave will be resampled if it doesn't match the \
            model.

        Returns
        -------
        A tuple, whose first element is the original wave and second element is a dict, whose keys
        are the name of stems and values are separated waves. The original wave will have already
        been resampled.

        Notes
        -----
        Use this function with cautiousness. This function does not provide data verifying.
        """
        if sr is not None:
            assert sr == self._samplerate

        from .vendor.demucs.apply import _replace_dict, apply_model

        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std() + 1e-8
        out = apply_model(
                self._model,
                ((wav - mean) / std)[None],
                segment=self.config.segment,
                shifts=self.config.shifts,
                split=self.config.split,
                overlap=self.config.overlap,
                device=self.config.device,
                num_workers=self.config.num_workers,
                callback=self.config.callback,
                callback_arg=_replace_dict(
                    self.config.callback_arg, ("audio_length", wav.shape[1])
                ),
                progress=self.config.progress,
            )
        out = out * std + mean
        return (wav, dict(zip(self._model.sources, out[0])))


    def separate(
        self,
        tracks: str | list[str] | Path | list[Path],
        *,
        output_dir = None,
        options = None
    ):
        import torch

        from iantirta.audio.files import load_audio

        if not isinstance(tracks, list):
            tracks = [tracks]

        for track in tracks:
            if not (track := Path(track)).is_file():
                continue
            audio = load_audio(track, self._samplerate)
            wav = torch.from_numpy(audio.samples)
            self.separate_tensor(wav, self._samplerate)

        return super().separate(input, output_dir=output_dir, options=options)
        

def load_demucs_model(name: str,):
    """Load a pretrained Demucs separation model.

    Parameters
    ----------
    name:
        Registered separation model name, for example ``"htdemucs"``.

    device:
        Torch device on which the model should run. If ``None``,
        an available device is selected automatically.

    Returns
    -------
    _DemucsModel
        Loaded separation model.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    ValueError
        If the requested model is not registered.

    RuntimeError
        If the model cannot be loaded.
    """

    from ._core import _MODELS

    try:
        spec = _MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown Demucs model: {name!r}"
        ) from None

    from .vendor.demucs.apply import BagOfModels

    try:
        model = BagOfModels(spec.models, spec.weights, spec.segment)
    except ImportError as exc:
        if 'diffq' in exc.args[0]:
            raise ImportError(
                "Demucs Audio separation requires diffq. "
                "Install it with `pip install iantirta-audio[separation]`."
            ) from exc
        raise
    
    model.eval()
    return model
