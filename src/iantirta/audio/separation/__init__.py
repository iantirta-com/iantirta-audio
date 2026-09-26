# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

from pathlib import Path

from .base import SeparationOptions, SeparationResult

__all__ = [
    "SeparationOptions",
    "SeparationResult",
    "separate",
]


def separate(
    input: str | Path,
    *,
    output_dir: str | Path | None = None,
    options: SeparationOptions | dict | None = None,
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
    from .demucs import DemucsSeparator, DemucsOptions

    if not isinstance(options, DemucsOptions):
        options = DemucsOptions.from_dict(options)

    separator = DemucsSeparator(options=options)
    return separator.separate(
        input,
        output_dir=output_dir,
    )
