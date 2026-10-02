# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

from pathlib import Path

from .base import SeparationConfig, SeparationResult, Separator

__all__ = [
    "SeparationConfig",
    "SeparationResult",
    "separate",
]


def separate(
    tracks: str | Path | list[str] | list[Path],
    *,
    output_dir: str | Path | None = None,
    options: SeparationConfig | dict | None = None,
    **kwargs,
) -> SeparationResult:
    """Separate an audio file into vocals and instrumental audio.

    Parameters
    ----------
    input:
        Path to the input audio file.

    output_dir:
        Directory where separated files are written. If ``None``, an
        output directory is created next to the input file.

    options:
        Separation configuration. If omitted, the default configuration
        is used.

    Returns
    -------
    SeparationResult
        The separated vocal and instrumental audio files.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.

    RuntimeError
        If the separation backend cannot be loaded.

    ValueError
        If the separation options are invalid.
    """

    if not isinstance(options, SeparationConfig):
        if options and isinstance(options, dict):
            options = SeparationConfig.from_dict(options)
        elif kwargs and isinstance(kwargs, dict):
            options = SeparationConfig.from_dict(kwargs)
        else:
            options = SeparationConfig()

    separator = Separator(options=options)
    return separator.separate(
        tracks,
        output_dir=output_dir,
    )
