# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from iantirta.models.demucs.apply import _replace_dict, apply_model
from iantirta.models.vendor.transformers.audio_utils import load_audio

from iantirta.audio.files import AudioFile


def _get_device(device: str | None = None):
    """Select the torch device used for separation.

    Parameters
    ----------
    device:
        Explicit device name such as ``"cpu"`` or ``"cuda"``.
        If ``None``, an available accelerator is selected automatically.

    Returns
    -------
    torch.device
        Selected computation device.

    Raises
    ------
    RuntimeError
        If the requested device is unavailable.
    """
    import torch

    if device is not None:
        result = torch.device(device)

        if result.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        if result.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available.")

        return result

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


@dataclass(frozen=True, slots=True)
class SeparationConfig:
    """Configuration for audio source separation.

    Parameters
    ----------
    model_name:
        Name of the separation model to use.
    
    shifts:
        Number of random shifts used during inference. Higher values can
        improve separation quality at the cost of additional computation.
    
    overlap:
        Fraction of overlap between adjacent processing segments.
        Must be between 0 and 1.
    
    segment:
        Processing segment length in seconds. If ``None``, the model's
        default segment length is used.
    
    num_workers:
        Number of CPU workers used for parallel segment processing.
        ``0`` disables parallel workers.
    
    device:
        Device used for model inference. If ``None``, an available
        accelerator is selected automatically.
    """

    model_name: str = "mdx_extra_q"
    device: str | None = field(default_factory=_get_device)

    shifts: int = 1
    overlap: float = 0.25
    split: bool = True
    segment: float | None = None

    progress: bool = False
    callback: Callable[[dict], None] | None = None
    callback_arg: dict | None = None

    num_workers: int = 0

    @classmethod
    def from_dict(cls, options: dict) -> SeparationConfig:
        return cls(**options)


@dataclass(frozen=True, slots=True)
class SeparationResult:
    """Result of separating an audio file into its source components.

    Parameters
    ----------
    vocals:
        Separated vocal track.
    
    instrumental:
        Remaining instrumental accompaniment.
    """

    vocals: AudioFile
    instrumental: AudioFile


class Separator:
    """Protocol implemented by audio separation backends."""

    def __init__(
        self,
        *,
        config: dict | SeparationConfig | None = None,
        **kwargs,
    ):
        if not isinstance(config, SeparationConfig):
            if config and isinstance(config, dict):
                config = SeparationConfig.from_dict(**config)
            elif kwargs and isinstance(kwargs, dict):
                config = SeparationConfig.from_dict(**kwargs)
            else:
                config = SeparationConfig()
        self.config = config

        from iantirta.models.demucs import DemucsBagOfModel
        self.model = DemucsBagOfModel.from_pretrained(self.config.model_name)
        self.model.eval()

        self.samplerate = self.model.samplerate
        self.audio_channels = self.model.audio_channels

    
    def separate_tensor(self, wav, sr):
        if sr is not None and self.samplerate != sr:
            raise ValueError()
        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std() + 1e-8
        out = apply_model(
            self.model,
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
        return (wav, dict(zip(self.model.sources, out[0])))

    
    def separate_audio_file(self, track):
        audionp = load_audio(str(track))
        return self.separate_tensor(
            torch.from_numpy(audionp), self.samplerate
        )

    
    def separate(
        self,
        tracks: str | Path | list[str] | list[Path],
        *,
        output_dir: str | Path | None = None,
    ) -> SeparationResult:
        """Separate an audio file into vocals and instrumental audio.

        Parameters
        ----------
        input:
            Input audio file.

        output_dir:
            Directory where separated files are written. If ``None``,
            the backend chooses the output location.

        options:
            Separation configuration. If ``None``, default options are used.

        Returns
        -------
        SeparationResult
            Separated vocal and instrumental audio files.
        """
        if not isinstance(tracks, list):
            tracks = [tracks]

        results = []
        for track in tracks:
            origin, res = self.separate_audio_file(track)
            results.append(SeparationResult(
                instrumental = origin - res["vocals"],
                vocals = res["vocals"]
            ))
            del origin, res
        return results
