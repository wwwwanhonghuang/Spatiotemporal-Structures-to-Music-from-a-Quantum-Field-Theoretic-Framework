from dataclasses import dataclass

@dataclass(frozen=True)
class NoteKey:
    pitch: int
    velocity: int
    channel: int