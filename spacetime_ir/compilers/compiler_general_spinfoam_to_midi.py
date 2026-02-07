#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mido


# =============================================================================
# Data models
# =============================================================================

@dataclass
class GeneralFoamVertex:
    vid: int
    label: str = ""


@dataclass
class GeneralFoamEdge:
    eid: int
    v_start: int
    v_end: Optional[int]
    incident_faces: List[int] = field(default_factory=list)


@dataclass
class GeneralFoamFace:
    fid: int
    boundary_edges: List[int] = field(default_factory=list)
    j: Optional[float] = None
    m: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneralSpinFoamIR:

    version: str = "general-2.0"

    vertices: List[GeneralFoamVertex] = field(default_factory=list)
    edges: List[GeneralFoamEdge] = field(default_factory=list)
    faces: List[GeneralFoamFace] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GeneralSpinFoamIR":
        return GeneralSpinFoamIR(
            version=str(d.get("version", "general-2.0")),
            vertices=[GeneralFoamVertex(**v) for v in d.get("vertices", [])],
            edges=[GeneralFoamEdge(**e) for e in d.get("edges", [])],
            faces=[GeneralFoamFace(**f) for f in d.get("faces", [])],
        )


@dataclass
class SemanticField:

    version: str = "semantic-2.0"

    tempo_bpm: Optional[float] = None
    ticks_per_beat: Optional[int] = None

    edge_fields: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SemanticField":

        return SemanticField(
            version=str(d.get("version", "semantic-2.0")),
            tempo_bpm=d.get("tempo_bpm"),
            ticks_per_beat=d.get("ticks_per_beat"),
            edge_fields={int(k): v for k, v in d.get("edge_fields", {}).items()},
        )


# =============================================================================
# Loading
# =============================================================================

def load_general_ir(path: str) -> GeneralSpinFoamIR:
    with open(path, "r", encoding="utf-8") as f:
        return GeneralSpinFoamIR.from_dict(json.load(f))


def load_semantic_field(path: str) -> SemanticField:
    with open(path, "r", encoding="utf-8") as f:
        return SemanticField.from_dict(json.load(f))


# =============================================================================
# Heuristic mapping (ONLY if semantic absent)
# =============================================================================

def clamp_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(x))))


def edge_spin_stats(edge, face_by_id):
    js, ms = [], []
    for fid in edge.incident_faces:
        f = face_by_id.get(fid)
        if not f:
            continue
        if f.j is not None:
            js.append(float(f.j))
        if f.m is not None:
            ms.append(abs(float(f.m)))

    avg_j = sum(js) / len(js) if js else 1.0
    avg_abs_m = sum(ms) / len(ms) if ms else 0.0

    return avg_j, avg_abs_m, len(js)


def j_to_pitch(avg_j, base_pitch, pitch_span):
    return clamp_int(base_pitch + avg_j * pitch_span, 0, 127)


def spin_to_velocity(avg_j, avg_abs_m, count_j):
    return clamp_int(35 + 10 * avg_j + 6 * avg_abs_m + 2 * count_j, 1, 127)


# =============================================================================
# Decoder
# =============================================================================

def decode_general_ir_to_midi(
    ir: GeneralSpinFoamIR,
    out_path: str,
    *,
    semantic_field: Optional[SemanticField] = None,
    default_bpm: float = 120.0,
    default_tpb: int = 480,
    base_pitch: int = 60,
    pitch_span: int = 6,
    default_channel: int = 0,
):

    reconstruction_mode = semantic_field is not None and len(semantic_field.edge_fields) > 0

    ticks_per_beat = (
        semantic_field.ticks_per_beat
        if reconstruction_mode and semantic_field.ticks_per_beat
        else default_tpb
    )

    tempo_bpm = (
        semantic_field.tempo_bpm
        if reconstruction_mode and semantic_field.tempo_bpm
        else default_bpm
    )

    face_by_id = {f.fid: f for f in ir.faces}

    events: List[Tuple[int, int, str, int, int, int]] = []

    # =====================================================
    # ⭐ Reconstruction mode
    # =====================================================

    if reconstruction_mode:

        for e in ir.edges:

            sf = semantic_field.edge_fields.get(e.eid)

            if not sf:
                continue

            start = sf.get("start_beats")
            duration = sf.get("duration_beats")

            if start is None:
                continue

            if duration is None:
                end = sf.get("end_beats")
                if end is None:
                    continue
                duration = end - start

            pitch = int(sf.get("pitch", base_pitch))
            velocity = int(sf.get("velocity", 80))
            channel = int(sf.get("channel", default_channel))

            start_tick = int(round(start * ticks_per_beat))
            end_tick = int(round((start + duration) * ticks_per_beat))

            events.append((start_tick, 1, "on", pitch, velocity, channel))
            events.append((end_tick, 0, "off", pitch, 0, channel))

    # =====================================================
    # ⭐ Generative mode
    # =====================================================

    else:

        print("⚠ No semantic field — using generative decoding")

        ordering = sorted(v.vid for v in ir.vertices)
        v_index = {vid: i for i, vid in enumerate(ordering)}

        for e in ir.edges:

            if e.v_start not in v_index:
                continue

            si = v_index[e.v_start]
            ei = v_index.get(e.v_end, si + 1)

            start_tick = si * ticks_per_beat
            end_tick = max(start_tick + ticks_per_beat, ei * ticks_per_beat)

            avg_j, avg_abs_m, count_j = edge_spin_stats(e, face_by_id)

            pitch = j_to_pitch(avg_j, base_pitch, pitch_span)
            velocity = spin_to_velocity(avg_j, avg_abs_m, count_j)

            events.append((start_tick, 1, "on", pitch, velocity, default_channel))
            events.append((end_tick, 0, "off", pitch, 0, default_channel))

    # =====================================================

    events.sort(key=lambda x: (x[0], x[1]))

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo_us = int(round(60_000_000 / tempo_bpm))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

    cur = 0
    for tick, _prio, typ, pitch, vel, ch in events:

        dt = max(0, tick - cur)
        cur = tick

        if typ == "on":
            track.append(mido.Message("note_on", note=pitch, velocity=vel, channel=ch, time=dt))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, channel=ch, time=dt))

    mid.save(out_path)

    print("\n✓ MIDI saved:", out_path)
    print("  reconstruction_mode =", reconstruction_mode)
    print("  bpm =", tempo_bpm)
    print("  ticks_per_beat =", ticks_per_beat)
