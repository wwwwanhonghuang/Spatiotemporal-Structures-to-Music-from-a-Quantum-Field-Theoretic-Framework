
from dataclasses import dataclass


@dataclass
class MIDINote:
    pitch: int
    velocity: int
    channel: int
    start_beats: float
    end_beats: float

    def is_active_at(self, t: float) -> bool:
        return self.start_beats <= t < self.end_beats
