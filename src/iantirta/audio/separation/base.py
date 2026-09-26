# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

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
class SeparationOptions:
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
    device: str | None = field(_default_factory=_get_device)

    @classmethod
    def from_dict(cls, options: dict) -> SeparationOptions:
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


class Separator(Protocol):
    """Protocol implemented by audio separation backends."""

    def separate(
        self,
        input: str | Path | AudioFile,
        *,
        output_dir: str | Path | None = None,
        options: SeparationOptions | None = None,
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
        ...