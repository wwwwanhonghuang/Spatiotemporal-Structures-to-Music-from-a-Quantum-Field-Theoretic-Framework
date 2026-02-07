#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SpinFoam v4 Decoder
- Accept input as:
    (1) JSON path
    (2) dict (already json-loaded)
    (3) MIDISpinFoamIR instance (dataclass)
- Decode edges' embedding (start_beats/end_beats) back to MIDI.

Dependencies:
  pip install mido
"""

import json
from dataclasses import is_dataclass
from typing import Any, Dict, List, Optional, Union

import mido
from mido import MidiFile, MidiTrack, Message


# ----------------------------
# IR adapters (dict <-> dataclass)
# ----------------------------

def _is_pathlike(x: Any) -> bool:
    return isinstance(x, str) and (x.endswith(".json") or x.endswith(".spinfoam.v4.json") or x.endswith(".ir.json") or x.endswith(".spinfoam.json"))

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _as_dict(ir_or_path_or_dict: Any) -> Dict[str, Any]:
    """
    Normalize input to a plain dict with keys:
      - tempo_bpm, ticks_per_beat
      - edges: list of edges with note + embedding
    """
    if _is_pathlike(ir_or_path_or_dict):
        return _load_json(ir_or_path_or_dict)

    if isinstance(ir_or_path_or_dict, dict):
        return ir_or_path_or_dict

    # dataclass instance
    if is_dataclass(ir_or_path_or_dict):
        # your IR has .to_dict(include_foliation=...)
        if hasattr(ir_or_path_or_dict, "to_dict"):
            return ir_or_path_or_dict.to_dict(include_foliation=True)
        # or fallback to vars()
        return vars(ir_or_path_or_dict)

    raise TypeError("Unsupported input type. Provide a JSON path, dict, or MIDISpinFoamIR instance.")


# ----------------------------
# Decoder
# ----------------------------

class SpinFoamDecoder:
    """
    Decode SpinFoam IR (v4 schema) to MIDI, using edge embedding as the decoding gauge.
    """

    def __init__(
        self,
        ir: Union[str, Dict[str, Any], Any],  # path | dict | IR instance
        *,
        default_tpb: int = 480,
        default_bpm: float = 120.0,
        allow_missing_embedding: bool = False,
    ):
        self._raw = _as_dict(ir)
        self.tpb = int(self._raw.get("ticks_per_beat", default_tpb))
        self.bpm = float(self._raw.get("tempo_bpm", default_bpm))
        self.tempo_us_per_beat = int(round(60_000_000 / self.bpm))
        self.allow_missing_embedding = allow_missing_embedding

        # light schema check
        if "edges" not in self._raw or not isinstance(self._raw["edges"], list):
            raise ValueError("IR missing 'edges' list. Cannot decode.")

    @staticmethod
    def beats_to_ticks(beats: float, tpb: int) -> int:
        return int(round(beats * tpb))

    def _extract_note_events(self) -> List[Dict[str, Any]]:
        """
        Return event list:
          [{"type":"on"/"off", "tick":int, "note":int, "vel":int, "ch":int}, ...]
        """
        events: List[Dict[str, Any]] = []

        for e in self._raw["edges"]:
            note = e.get("note", {})
            emb = e.get("embedding", {})

            pitch = note.get("pitch", None)
            vel = note.get("velocity", 64)
            ch = note.get("channel", 0)

            if pitch is None:
                continue

            if not emb:
                if self.allow_missing_embedding:
                    # If embedding missing, we skip (or you could synthesize).
                    continue
                raise ValueError(f"Edge has no embedding; cannot decode losslessly. Edge(note={note}).")

            t0 = emb.get("start_beats", None)
            t1 = emb.get("end_beats", None)

            if t0 is None or t1 is None:
                if self.allow_missing_embedding:
                    continue
                raise ValueError(f"Edge embedding missing start/end beats; cannot decode. Edge(note={note}, embedding={emb}).")

            start_tick = self.beats_to_ticks(float(t0), self.tpb)
            end_tick = self.beats_to_ticks(float(t1), self.tpb)

            # Guard: ensure end >= start (else clamp)
            if end_tick < start_tick:
                end_tick = start_tick

            events.append({"type": "on", "tick": start_tick, "note": int(pitch), "vel": int(vel), "ch": int(ch)})
            events.append({"type": "off", "tick": end_tick, "note": int(pitch), "vel": 0, "ch": int(ch)})

        # Sorting rule:
        # - earlier tick first
        # - at same tick: note_off before note_on (avoid stuck notes when re-attacking)
        # - then by pitch for stability
        def key(ev):
            pri = 0 if ev["type"] == "off" else 1
            return (ev["tick"], pri, ev["note"], ev["ch"])

        events.sort(key=key)
        return events

    def decode_to_midi(
        self,
        out_path: str,
        *,
        add_tempo: bool = True,
        program_changes: Optional[Dict[int, int]] = None,  # channel -> program
    ) -> None:
        """
        Write reconstructed MIDI.
        - program_changes: optional mapping channel->program number (0..127).
        """
        midi = MidiFile(ticks_per_beat=self.tpb)
        track = MidiTrack()
        midi.tracks.append(track)

        if add_tempo:
            track.append(mido.MetaMessage("set_tempo", tempo=self.tempo_us_per_beat, time=0))

        # Optional program changes at time 0
        if program_changes:
            # keep deterministic order
            for ch in sorted(program_changes.keys()):
                prog = int(program_changes[ch])
                prog = max(0, min(127, prog))
                track.append(Message("program_change", program=prog, channel=int(ch), time=0))

        events = self._extract_note_events()

        last_tick = 0
        for ev in events:
            tick = ev["tick"]
            delta = tick - last_tick
            if delta < 0:
                delta = 0
            last_tick = tick

            if ev["type"] == "on":
                track.append(Message("note_on", note=ev["note"], velocity=ev["vel"], channel=ev["ch"], time=delta))
            else:
                track.append(Message("note_off", note=ev["note"], velocity=0, channel=ev["ch"], time=delta))

        midi.save(out_path)
        print(f"✓ Decoded MIDI saved to {out_path}")


# ----------------------------
# Convenience function
# ----------------------------

def decode_spinfoam_to_midi(
    ir_or_json_path: Union[str, Dict[str, Any], Any],
    out_path: str,
    *,
    default_tpb: int = 480,
    default_bpm: float = 120.0,
    allow_missing_embedding: bool = False,
    program_changes: Optional[Dict[int, int]] = None,
) -> None:
    dec = SpinFoamDecoder(
        ir_or_json_path,
        default_tpb=default_tpb,
        default_bpm=default_bpm,
        allow_missing_embedding=allow_missing_embedding,
    )
    dec.decode_to_midi(out_path, program_changes=program_changes)

