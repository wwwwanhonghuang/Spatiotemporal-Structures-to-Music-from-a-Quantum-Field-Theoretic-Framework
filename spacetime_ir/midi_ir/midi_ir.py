#!/usr/bin/env python3
"""
Geometric / Physics-faithful (as a *model*) MIDI -> SpinFoam IR compiler

What this implements (complete, runnable):
1) SpinFoam intrinsic IR = {vertices, edges, faces} with NO time as ontology.
2) Optional foliation Γ_t (slicing) stored separately for debugging/decoding.
3) Note is NOT primitive:
   - each note-edge (worldline) carries a 4-valent "note-3cell" made of 4 component faces.
4) Each note-component face has (j,m) and a flux vector J in R^3.
   - j,m are quantized to half-integers
   - |m| <= j enforced (projected if needed)
5) Intertwiner types (switchable):
   - "vector_closure": true vector closure ||Σ J_i|| small
   - "fusion_4valent": SU(2)-like admissibility via intermediate k channels
   - "constraint_gate": physically motivated gates across the 4 channels
   - "hybrid": combines all above (recommended)
6) Same-slice geometry = spin-network-like Γ slice:
   - nodes = active note-edges
   - links = sparse glue edges (top-k by harmonic score)
7) Cross-slice coupling = glue-worldsheet faces:
   - persistent glue links sweep faces between rewrite-vertices.

Dependencies:
  pip install mido
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict

import mido

from spacetime_ir.spinfoam_ir.foam_vertex import FoamVertex
from spacetime_ir.spinfoam_ir.foam_face import FoamFace
from spacetime_ir.spinfoam_ir.foam_edge import FoamEdge
from spacetime_ir.spinfoam_ir.foliation_slice import FoliationSlice
from spacetime_ir.midi_ir.spin_physics import *
from spacetime_ir.midi_ir.intertwiner import *
from spacetime_ir.midi_ir.gluing import *
from spacetime_ir.midi_ir.intrinsic_vertex import *

from utils.math_and_physics.math_and_physics import *
from utils.music import pitch_to_name

@dataclass
class MIDISpinFoamIR:
    version: str = "4.0"
    title: str = ""
    tempo_bpm: float = 120.0
    ticks_per_beat: int = 480

    vertices: List[FoamVertex] = field(default_factory=list)
    edges: List[FoamEdge] = field(default_factory=list)
    faces: List[FoamFace] = field(default_factory=list)

    # optional foliation Γ_t + embedding times
    foliation: List[FoliationSlice] = field(default_factory=list)

    def to_dict(self, include_foliation: bool = True) -> Dict:
        d = {
            "version": self.version,
            "title": self.title,
            "tempo_bpm": self.tempo_bpm,
            "ticks_per_beat": self.ticks_per_beat,
            "num_vertices": len(self.vertices),
            "num_edges": len(self.edges),
            "num_faces": len(self.faces),
            "vertices": [vars(v) for v in self.vertices],
            "edges": [vars(e) for e in self.edges],
            "faces": [vars(f) for f in self.faces],
        }
        if include_foliation:
            d["foliation"] = [vars(s) for s in self.foliation]
        return d

    def serialize(self, path: str, include_foliation: bool = True) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_foliation=include_foliation), f, indent=2, ensure_ascii=False)
        print(f"✓ Saved IR: {path} (foliation={'on' if include_foliation else 'off'})")


def faces_of_edge(ir: MIDISpinFoamIR, eid: int) -> List[FoamFace]:
    e = ir.edges[eid]
    return [ir.faces[fid] for fid in e.component_faces]

def glue_faces_between(ir: MIDISpinFoamIR, a: int, b: int) -> List[FoamFace]:
    aa, bb = (a, b) if a < b else (b, a)
    out = []
    for f in ir.faces:
        if f.face_type != "glue_worldsheet":
            continue
        if f.edge_a is None or f.edge_b is None:
            continue
        x, y = (f.edge_a, f.edge_b)
        x, y = (x, y) if x < y else (y, x)
        if x == aa and y == bb:
            out.append(f)
    return out

def print_summary(ir: MIDISpinFoamIR, max_edges: int = 8):
    print("\n" + "="*90)
    print("SpinFoam IR v4 SUMMARY (intrinsic: vertices/edges/faces; optional foliation)")
    print("="*90)
    print(f"Title: {ir.title}")
    print(f"Tempo: {ir.tempo_bpm:.2f} BPM | TPB: {ir.ticks_per_beat}")
    print(f"Vertices: {len(ir.vertices)} | Edges(notes): {len(ir.edges)} | Faces: {len(ir.faces)} | Γ_slices: {len(ir.foliation)}")

    print("\nVertices (rewrite regions, no time):")
    for v in ir.vertices[:min(10, len(ir.vertices))]:
        print(f"  v{v.vid}: {v.label}")

    print("\nSample note-edges with component faces (j,m) and intertwiner:")
    for e in ir.edges[:max_edges]:
        n = e.note
        name = pitch_to_name(n["pitch"])
        print(f"\n  e{e.eid}: {name} ch={n['channel']} vel={n['velocity']}  v{e.v_start}->{('v'+str(e.v_end)) if e.v_end is not None else 'OPEN'}")
        print(f"    intertwiner: mode={e.intertwiner.get('mode')} closed={e.intertwiner.get('closed')} sum_m={e.intertwiner.get('sum_m'):.2f}")
        if "vector_closure" in e.intertwiner:
            vc = e.intertwiner["vector_closure"]
            print(f"    vector_closure: err={vc['closure_error']:.3f} tol={vc['tol']} closed={vc['closed']}")
        if "fusion" in e.intertwiner:
            fu = e.intertwiner["fusion"]
            print(f"    fusion_4valent: exists={fu['exists']} k_candidates={fu['k_candidates'][:5]}")
        if "gate" in e.intertwiner:
            gt = e.intertwiner["gate"]
            print(f"    constraint_gate: admissible={gt['admissible']} reasons={gt['reasons']}")

        for f in faces_of_edge(ir, e.eid):
            print(f"    f{f.fid} [{f.channel}] (j={f.j}, m={f.m}) | |J|={v_norm(f.flux):.3f} meta={f.meta}")

    if ir.foliation:
        s0 = ir.foliation[0]
        print("\nFoliation (external slicing; not intrinsic):")
        print(f"  Γ[0] @beats={s0.embedding_time_beats:.2f} nodes={len(s0.gamma_nodes)} links={len(s0.gamma_links)}")


# =============================================================================
# Test MIDI generator
# =============================================================================

def create_test_midi() -> str:
    from mido import MidiFile, MidiTrack, Message
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

    # Beat 1: C4(60), D3(50), E1(28)
    track.append(Message('note_on', note=60, velocity=80, time=0))
    track.append(Message('note_on', note=50, velocity=75, time=0))
    track.append(Message('note_on', note=28, velocity=70, time=0))

    # after 2 beats: E1 off at beat 3
    track.append(Message('note_off', note=28, velocity=0, time=960))

    # after 2 beats: C4 off and D3 off at beat 5
    track.append(Message('note_off', note=60, velocity=0, time=960))
    track.append(Message('note_off', note=50, velocity=0, time=0))

    # Beat 5: D#3(51), E2(40), F5(77)
    track.append(Message('note_on', note=51, velocity=75, time=0))
    track.append(Message('note_on', note=40, velocity=80, time=0))
    track.append(Message('note_on', note=77, velocity=85, time=0))

    # after 2 beats: F5 off at beat 7
    track.append(Message('note_off', note=77, velocity=0, time=960))

    # after 1 beat: D#3 off at beat 8
    track.append(Message('note_off', note=51, velocity=0, time=480))

    # after 1 beat: E2 off at beat 9
    track.append(Message('note_off', note=40, velocity=0, time=480))

    path = "test_example.mid"
    midi.save(path)
    print(f"✓ Created test MIDI: {path}")
    return path

