
import pytest
from pathlib import Path


@pytest.fixture
def audio():
    return str(Path("../dataset/vocals.wav").expanduser().resolve())
