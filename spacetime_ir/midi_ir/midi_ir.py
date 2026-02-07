#!/usr/bin/env python3
"""
MIDI to Spin Foam Compiler - Philosophy: "Only Changes Are Events"
"""

import json
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
import mido
from collections import defaultdict
import math


@dataclass
class MIDINote:
    """A complete note with start and end times"""
    pitch: int
    velocity: int
    channel: int
    start_time: float  # in beats
    end_time: float  # in beats
    note_id: int  # Unique identifier for this note instance
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def __repr__(self):
        return f"N{self.note_id}:{self.pitch_name()}@{self.start_time:.2f}-{self.end_time:.2f}"
    
    def pitch_name(self) -> str:
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (self.pitch // 12) - 1
        name = note_names[self.pitch % 12]
        return f"{name}{octave}"


@dataclass 
class ChangeEvent:
    """
    A vertex in the spin foam representing a REAL change in musical state.
    
    Core philosophy: If nothing changes, no event exists.
    """
    event_id: int
    time: float  # When this change occurs
    active_notes: Set[int]  # Set of note_ids active AFTER this event
    notes_born: Set[int]    # Note_ids that start at this event
    notes_died: Set[int]    # Note_ids that end at this event
    duration_to_next: Optional[float] = None  # Time until next change
    next_event_id: Optional[int] = None  # ID of next change event
    
    def __repr__(self):
        born_str = f"+{len(self.notes_born)}" if self.notes_born else ""
        died_str = f"-{len(self.notes_died)}" if self.notes_died else ""
        change_str = f"{born_str}{died_str}".strip()
        return f"E{self.event_id}@{self.time:.2f}[{change_str}]→{self.next_event_id}"


@dataclass
class PersistenceEdge:
    """
    An edge representing a note's persistence between change events.
    
    In spin foam: A face's worldline between vertices.
    """
    edge_id: int
    note_id: int
    note: MIDINote
    start_event_id: int  # Event where this note begins
    end_event_id: Optional[int]  # Event where this note ends (None if continues beyond scope)
    duration: float  # Duration in beats
    
    # Spin representation
    spin_j: float  # Total spin (encodes pitch)
    spin_m: float  # Magnetic quantum number (encodes timbre/expression)
    
    def __repr__(self):
        return f"Edge{self.edge_id}:N{self.note_id}({self.note.pitch_name()}) E{self.start_event_id}→E{self.end_event_id}"


@dataclass
class MIDISpinFoamIR:
    """
    MIDI Spin Foam IR that truly follows "only changes are events" philosophy.
    """
    # Core spin foam components
    events: List[ChangeEvent] = field(default_factory=list)  # Vertices (ONLY real changes)
    edges: List[PersistenceEdge] = field(default_factory=list)  # Faces' worldlines
    
    # Metadata
    tempo_bpm: float = 120.0
    total_duration: float = 0.0
    title: str = ""
    
    # Event chain (causal structure)
    first_event_id: Optional[int] = None
    last_event_id: Optional[int] = None
    
    def add_event(self, event: ChangeEvent):
        """Add a change event and update the causal chain"""
        self.events.append(event)
        
        if len(self.events) == 1:
            self.first_event_id = event.event_id
        else:
            # Update previous event's next pointer
            prev_event = self.events[-2]
            prev_event.next_event_id = event.event_id
            prev_event.duration_to_next = event.time - prev_event.time
        
        self.last_event_id = event.event_id
    
    def add_edge(self, edge: PersistenceEdge):
        """Add a persistence edge"""
        self.edges.append(edge)
    
    def get_event(self, event_id: int) -> Optional[ChangeEvent]:
        """Get event by ID"""
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None
    
    def get_causal_chain(self) -> List[ChangeEvent]:
        """Get events in causal order"""
        chain = []
        current_id = self.first_event_id
        
        while current_id is not None:
            event = self.get_event(current_id)
            if event is None:
                break
            chain.append(event)
            current_id = event.next_event_id
        
        return chain
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "version": "2.0",
            "philosophy": "Only changes constitute events. Time emerges from causal chain.",
            "title": self.title,
            "tempo_bpm": self.tempo_bpm,
            "total_duration": self.total_duration,
            "first_event_id": self.first_event_id,
            "last_event_id": self.last_event_id,
            "events": [
                {
                    "event_id": e.event_id,
                    "time": e.time,
                    "active_notes": list(e.active_notes),
                    "notes_born": list(e.notes_born),
                    "notes_died": list(e.notes_died),
                    "duration_to_next": e.duration_to_next,
                    "next_event_id": e.next_event_id
                }
                for e in self.events
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "note_id": e.note_id,
                    "note": {
                        "pitch": e.note.pitch,
                        "pitch_name": e.note.pitch_name(),
                        "velocity": e.note.velocity,
                        "channel": e.note.channel,
                        "start_time": e.note.start_time,
                        "end_time": e.note.end_time
                    },
                    "start_event_id": e.start_event_id,
                    "end_event_id": e.end_event_id,
                    "duration": e.duration,
                    "spin_j": e.spin_j,
                    "spin_m": e.spin_m
                }
                for e in self.edges
            ]
        }
    
    def serialize(self, filepath: str):
        """Save to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"✓ Compiled Spin Foam saved to {filepath}")
    
    @classmethod
    def deserialize(cls, filepath: str) -> "MIDISpinFoamIR":
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct notes and edges
        foam = cls(
            tempo_bpm=data["tempo_bpm"],
            total_duration=data["total_duration"],
            title=data["title"],
            first_event_id=data["first_event_id"],
            last_event_id=data["last_event_id"]
        )
        
        # Reconstruct events
        event_dicts = data["events"]
        events = []
        for e_dict in event_dicts:
            event = ChangeEvent(
                event_id=e_dict["event_id"],
                time=e_dict["time"],
                active_notes=set(e_dict["active_notes"]),
                notes_born=set(e_dict["notes_born"]),
                notes_died=set(e_dict["notes_died"]),
                duration_to_next=e_dict.get("duration_to_next"),
                next_event_id=e_dict.get("next_event_id")
            )
            events.append(event)
        
        foam.events = events
        
        # Reconstruct edges
        edge_dicts = data["edges"]
        edges = []
        for e_dict in edge_dicts:
            note_data = e_dict["note"]
            note = MIDINote(
                pitch=note_data["pitch"],
                velocity=note_data["velocity"],
                channel=note_data["channel"],
                start_time=note_data["start_time"],
                end_time=note_data["end_time"],
                note_id=e_dict["note_id"]
            )
            
            edge = PersistenceEdge(
                edge_id=e_dict["edge_id"],
                note_id=e_dict["note_id"],
                note=note,
                start_event_id=e_dict["start_event_id"],
                end_event_id=e_dict.get("end_event_id"),
                duration=e_dict["duration"],
                spin_j=e_dict["spin_j"],
                spin_m=e_dict["spin_m"]
            )
            edges.append(edge)
        
        foam.edges = edges
        
        return foam


def pitch_to_spin(pitch: int, velocity: int) -> Tuple[float, float]:
    """
    Convert MIDI pitch and velocity to spin representation (j, m).
    
    j (total spin): encodes pitch class and octave
    m (projection): encodes timbre/expression (using velocity)
    """
    # j: combines pitch class and octave
    octave = pitch // 12 - 1
    pitch_class = pitch % 12
    
    # j increases with octave and varies with pitch class
    j_base = octave * 2.0  # 2.0 per octave
    j_variation = pitch_class / 6.0  # 0 to 2.0 within octave
    j = j_base + j_variation
    
    # m: uses velocity and fine pitch variation
    # Normalize velocity to [-1, 1] range
    m_velocity = (velocity - 64) / 64.0  # 64 is typical mezzo-forte
    
    # Add micro-variation based on pitch class
    m_pitch_variation = (pitch_class % 3) / 3.0 * 0.1  # Small variation
    
    m = m_velocity + m_pitch_variation
    
    return round(j, 3), round(m, 3)


def parse_midi_file(midi_file_path: str) -> Tuple[List[MIDINote], float]:
    """Parse MIDI file and extract complete notes"""
    midi_file = mido.MidiFile(midi_file_path)
    ticks_per_beat = midi_file.ticks_per_beat
    
    tempo = 500000  # Default: 120 BPM
    current_time = 0.0
    active_notes = {}  # (pitch, channel) -> (start_time, velocity, note_id)
    completed_notes = []
    note_id_counter = 0
    
    for track in midi_file.tracks:
        current_time = 0.0
        for msg in track:
            delta_beats = msg.time / ticks_per_beat
            current_time += delta_beats
            
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                key = (msg.note, msg.channel)
                active_notes[key] = (current_time, msg.velocity, note_id_counter)
                note_id_counter += 1
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in active_notes:
                    start_time, velocity, nid = active_notes[key]
                    note = MIDINote(
                        pitch=msg.note,
                        velocity=velocity,
                        channel=msg.channel,
                        start_time=start_time,
                        end_time=current_time,
                        note_id=nid
                    )
                    completed_notes.append(note)
                    del active_notes[key]
    
    # Handle notes that never ended
    end_time = current_time
    for key, (start_time, velocity, nid) in active_notes.items():
        pitch, channel = key
        note = MIDINote(
            pitch=pitch,
            velocity=velocity,
            channel=channel,
            start_time=start_time,
            end_time=end_time,
            note_id=nid
        )
        completed_notes.append(note)
    
    # Calculate BPM
    bpm = 60_000_000 / tempo
    
    return completed_notes, bpm, end_time


def detect_real_changes(notes: List[MIDINote]) -> List[ChangeEvent]:
    """
    Detect ONLY times when the set of active notes actually changes.
    
    Returns events sorted by time, with NO "no change" events.
    """
    # Collect ALL unique event times (note starts and ends)
    event_times = set()
    note_start_times = {}  # note_id -> start_time
    note_end_times = {}    # note_id -> end_time
    
    for note in notes:
        event_times.add(note.start_time)
        event_times.add(note.end_time)
        note_start_times[note.note_id] = note.start_time
        note_end_times[note.note_id] = note.end_time
    
    # Sort times and filter out times where nothing changes
    sorted_times = sorted(event_times)
    change_events = []
    event_id_counter = 0
    
    # We'll process in pairs: (prev_time, current_time)
    # Only create event at current_time if active notes changed since prev_time
    prev_active_notes = set()
    
    for i, current_time in enumerate(sorted_times):
        # Determine which notes are active at current_time
        current_active = set()
        notes_starting = set()
        notes_ending = set()
        
        for note in notes:
            if math.isclose(note.start_time, current_time, abs_tol=1e-9):
                current_active.add(note.note_id)
                notes_starting.add(note.note_id)
            elif math.isclose(note.end_time, current_time, abs_tol=1e-9):
                notes_ending.add(note.note_id)
                # Note ending: it was active just before this time
                if note.note_id in prev_active_notes:
                    notes_ending.add(note.note_id)
            elif note.start_time < current_time < note.end_time:
                current_active.add(note.note_id)
        
        # Check if this is a REAL change from previous state
        # A change occurs if:
        # 1. Notes are born (notes_starting not empty)
        # 2. Notes die (notes_ending not empty)
        # 3. The set of active notes changed
        
        is_real_change = (
            len(notes_starting) > 0 or
            len(notes_ending) > 0 or
            current_active != prev_active_notes
        )
        
        if is_real_change:
            # Create a change event
            event = ChangeEvent(
                event_id=event_id_counter,
                time=current_time,
                active_notes=current_active,
                notes_born=notes_starting,
                notes_died=notes_ending,
                duration_to_next=None,  # Will be filled later
                next_event_id=None
            )
            change_events.append(event)
            event_id_counter += 1
            
            # Update for next iteration
            prev_active_notes = current_active
    
    # Now update duration_to_next and next_event_id
    for i in range(len(change_events) - 1):
        current = change_events[i]
        next_event = change_events[i + 1]
        
        current.duration_to_next = next_event.time - current.time
        current.next_event_id = next_event.event_id
    
    return change_events


def build_persistence_edges(notes: List[MIDINote], events: List[ChangeEvent]) -> List[PersistenceEdge]:
    """
    Build edges representing note persistence between change events.
    
    Each edge corresponds to a note's existence between two change events.
    """
    edges = []
    edge_id_counter = 0
    
    # Create a map from time to event for quick lookup
    time_to_event = {event.time: event.event_id for event in events}
    
    for note in notes:
        # Find the event where this note starts
        start_event_id = None
        for event in events:
            if math.isclose(event.time, note.start_time, abs_tol=1e-9):
                if note.note_id in event.notes_born:
                    start_event_id = event.event_id
                    break
        
        # Find the event where this note ends
        end_event_id = None
        for event in events:
            if math.isclose(event.time, note.end_time, abs_tol=1e-9):
                if note.note_id in event.notes_died:
                    end_event_id = event.event_id
                    break
        
        # If we found both start and end events, create an edge
        if start_event_id is not None:
            # Calculate spin values
            spin_j, spin_m = pitch_to_spin(note.pitch, note.velocity)
            
            edge = PersistenceEdge(
                edge_id=edge_id_counter,
                note_id=note.note_id,
                note=note,
                start_event_id=start_event_id,
                end_event_id=end_event_id,
                duration=note.duration,
                spin_j=spin_j,
                spin_m=spin_m
            )
            edges.append(edge)
            edge_id_counter += 1
    
    return edges


def compile_midi_to_spinfoam(midi_file_path: str, output_json_path: Optional[str] = None) -> MIDISpinFoamIR:
    """
    Compile MIDI file to spin foam IR following "only changes are events" philosophy.
    """
    print(f"🔧 Compiling MIDI: {midi_file_path}")
    
    # 1. Parse MIDI
    notes, bpm, total_duration = parse_midi_file(midi_file_path)
    print(f"   Parsed {len(notes)} notes, {bpm:.1f} BPM, duration: {total_duration:.2f} beats")
    
    # 2. Detect ONLY real changes
    events = detect_real_changes(notes)
    print(f"   Found {len(events)} REAL change events (no 'no change' events)")
    
    # 3. Build persistence edges
    edges = build_persistence_edges(notes, events)
    print(f"   Built {len(edges)} persistence edges")
    
    # 4. Create spin foam IR
    foam = MIDISpinFoamIR(
        tempo_bpm=bpm,
        total_duration=total_duration,
        title=midi_file_path
    )
    
    # Add events in order
    for event in sorted(events, key=lambda e: e.time):
        foam.add_event(event)
    
    # Add edges
    for edge in edges:
        foam.add_edge(edge)
    
    # 5. Save if requested
    if output_json_path:
        foam.serialize(output_json_path)
    
    return foam


def print_spinfoam_summary(foam: MIDISpinFoamIR, max_events: int = 20):
    """Print a summary of the spin foam"""
    print(f"\n{'='*80}")
    print(f"MIDI SPIN FOAM IR - 'Only Changes Are Events'")
    print(f"{'='*80}")
    print(f"Title: {foam.title}")
    print(f"Tempo: {foam.tempo_bpm:.1f} BPM")
    print(f"Total duration: {foam.total_duration:.2f} beats")
    print(f"Change events (vertices): {len(foam.events)}")
    print(f"Persistence edges: {len(foam.edges)}")
    
    print(f"\n{'='*80}")
    print(f"CAUSAL CHAIN OF EVENTS (Time emerges from this chain):")
    print(f"{'='*80}")
    
    chain = foam.get_causal_chain()
    
    # Print limited number of events
    display_events = chain[:max_events] if len(chain) > max_events else chain
    
    for event in display_events:
        # Get note names for born and died notes
        born_names = []
        died_names = []
        
        for edge in foam.edges:
            if edge.note_id in event.notes_born:
                born_names.append(edge.note.pitch_name())
            if edge.note_id in event.notes_died:
                died_names.append(edge.note.pitch_name())
        
        born_str = f"+{','.join(born_names)}" if born_names else ""
        died_str = f"-{','.join(died_names)}" if died_names else ""
        change_str = f"{born_str}{died_str}".strip()
        
        print(f"E{event.event_id:3d} @ {event.time:6.2f} beats | "
              f"Active notes: {len(event.active_notes):2d} | "
              f"Change: {change_str:15s} | "
              f"→ E{event.next_event_id if event.next_event_id is not None else 'END':3} "
              f"(after {event.duration_to_next if event.duration_to_next else 0:.2f} beats)")
    
    if len(chain) > max_events:
        print(f"  ... and {len(chain) - max_events} more events")
    
    print(f"\n{'='*80}")
    print(f"PERSISTENCE EDGES (Note Worldlines):")
    print(f"{'='*80}")
    
    # Group edges by note
    edges_by_note = defaultdict(list)
    for edge in foam.edges:
        edges_by_note[edge.note.pitch_name()].append(edge)
    
    for pitch_name, note_edges in list(edges_by_note.items())[:10]:  # Show first 10
        for edge in note_edges:
            print(f"  Edge{edge.edge_id:3d}: {pitch_name:4s} "
                  f"(N{edge.note_id:3d}) "
                  f"spin=({edge.spin_j:.2f}, {edge.spin_m:+.2f}) "
                  f"E{edge.start_event_id:3d}→E{edge.end_event_id if edge.end_event_id else 'END':3} "
                  f"({edge.duration:.2f} beats)")
    
    if len(edges_by_note) > 10:
        print(f"  ... and {len(edges_by_note) - 10} more notes")


def create_test_midi() -> str:
    """Create our example MIDI file programmatically"""
    from mido import MidiFile, MidiTrack, Message
    
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Set tempo
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    
    # Our example: C4,D3,E1 (2 beats) → C4,D3 (2 beats) → D#3,E2,F5 (2 beats) → D#3,E2 (1 beat) → E2 (1 beat)
    
    # Beat 1: C4(60), D3(50), E1(28) start
    track.append(Message('note_on', note=60, velocity=80, time=0))
    track.append(Message('note_on', note=50, velocity=75, time=0))
    track.append(Message('note_on', note=28, velocity=70, time=0))
    
    # Beat 3: E1 ends (after 2 beats = 960 ticks)
    track.append(Message('note_off', note=28, velocity=0, time=960))
    
    # Beat 5: C4 and D3 end, D#3(51), E2(40), F5(77) start
    track.append(Message('note_off', note=60, velocity=0, time=960))
    track.append(Message('note_off', note=50, velocity=0, time=0))
    track.append(Message('note_on', note=51, velocity=75, time=0))
    track.append(Message('note_on', note=40, velocity=80, time=0))
    track.append(Message('note_on', note=77, velocity=85, time=0))
    
    # Beat 7: F5 ends (after 2 beats)
    track.append(Message('note_off', note=77, velocity=0, time=960))
    
    # Beat 8: D#3 ends (after 1 beat = 480 ticks)
    track.append(Message('note_off', note=51, velocity=0, time=480))
    
    # Beat 9: E2 ends (after 1 beat)
    track.append(Message('note_off', note=40, velocity=0, time=480))
    
    test_path = "philosophy_example.mid"
    midi.save(test_path)
    print(f"✓ Created test MIDI: {test_path}")
    
    return test_path

