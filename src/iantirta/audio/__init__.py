# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from importlib.metadata import version

from .files import AudioData, AudioFile, get_audio_info, load_audio

__all__ = [
    "AudioData",
    "AudioFile",
    "get_audio_info",
    "load_audio",
]

__version__ = version("iantirta-audio")
