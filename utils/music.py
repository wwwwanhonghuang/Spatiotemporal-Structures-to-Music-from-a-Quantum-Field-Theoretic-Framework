from typing import Tuple, List, Dict, Set
from spacetime_ir.midi_ir.midi_note import MIDINote
import mido

def pitch_to_name(pitch: int) -> str:
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    name = note_names[pitch % 12]
    return f"{name}{octave}"

def detect_state_change_times(notes: List[MIDINote]) -> List[float]:
    times: Set[float] = set()
    for n in notes:
        times.add(n.start_beats)
        times.add(n.end_beats)
    return sorted(times)

def parse_midi_file(path: str) -> Tuple[List[MIDINote], float, int]:
    midi = mido.MidiFile(path)
    tpb = midi.ticks_per_beat
    tempo = 500000  # default 120 BPM

    completed: List[MIDINote] = []
    active: Dict[Tuple[int, int], Tuple[float, int]] = {}  # (pitch,ch)->(start_beats,vel)

    for track in midi.tracks:
        t = 0.0
        for msg in track:
            t += msg.time / tpb
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                active[(msg.note, msg.channel)] = (t, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in active:
                    t0, vel = active[key]
                    completed.append(MIDINote(msg.note, vel, msg.channel, t0, t))
                    del active[key]

    # close dangling notes at last time
    end_t = max((n.end_beats for n in completed), default=0.0)
    for (pitch, ch), (t0, vel) in active.items():
        completed.append(MIDINote(pitch, vel, ch, t0, end_t))

    completed.sort(key=lambda n: (n.start_beats, n.pitch, n.channel))
    bpm = 60_000_000 / tempo
    return completed, bpm, tpb