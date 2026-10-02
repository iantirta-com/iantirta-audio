
import pytest

from iantirta.audio.separation import Separator

def test_separator():
    separator = Separator()
    assert separator is not None