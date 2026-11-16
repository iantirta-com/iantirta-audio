
import logging

import pytest

from iantirta.audio.separation import ModelAudioSeparation


class TestSeparationModel:
    def test_init_class(self):
        model = ModelAudioSeparation(
            "adefosezz/Demucs-mdx_extra_q"
        )
        assert model.logger is not None

    def test_init_with_logger(self):
        logger = logging.getLogger("TestSeparation")
        model = ModelAudioSeparation(
            "adefosezz/Demucs-mdx_extra_q",
            logger=logger
        )
        assert model.logger is not None
        assert model.logger.name == "TestSeparation"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        '--log-cli-level=DEBUG',
        __file__
    ]))
