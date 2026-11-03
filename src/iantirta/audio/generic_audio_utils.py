

from dataclasses import dataclass

__all__ = [
    "AudioSegment",
    "HumanTiming",
    "TimingMixin",
]


@dataclass
class HumanTiming:
    pass

@dataclass(slots=True)
class TimingMixin:
    """
    Class that represent timing in seconds
    
    :param float start:
    :param float end:
    """
    start: float
    end: float

    @property
    def duration(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return float(
            round(self.end - self.start, 2)
        )


@dataclass(slots=True)
class AudioSegment(TimingMixin):
    pass
