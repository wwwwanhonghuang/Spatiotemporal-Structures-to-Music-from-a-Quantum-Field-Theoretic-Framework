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
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

import mido  # pip install mido


# =============================================================================
# Utilities: quantization & small linear algebra
# =============================================================================

def q_half(x: float) -> float:
    """Quantize to nearest half-integer."""
    return round(x * 2.0) / 2.0

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def v_add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def v_scale(a, s: float):
    return (a[0]*s, a[1]*s, a[2]*s)

def v_norm(a) -> float:
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def sph_to_vec(r: float, theta: float, phi: float):
    """
    theta: polar [0, pi], phi: azimuth [0, 2pi)
    """
    return (
        r * math.sin(theta) * math.cos(phi),
        r * math.sin(theta) * math.sin(phi),
        r * math.cos(theta),
    )

def pitch_to_name(pitch: int) -> str:
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    name = note_names[pitch % 12]
    return f"{name}{octave}"


# =============================================================================
# MIDI parsing (embedding only; ontology-free)
# =============================================================================

@dataclass(frozen=True)
class NoteKey:
    pitch: int
    velocity: int
    channel: int

@dataclass
class MIDINote:
    pitch: int
    velocity: int
    channel: int
    start_beats: float
    end_beats: float

    def is_active_at(self, t: float) -> bool:
        return self.start_beats <= t < self.end_beats

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

def detect_state_change_times(notes: List[MIDINote]) -> List[float]:
    times: Set[float] = set()
    for n in notes:
        times.add(n.start_beats)
        times.add(n.end_beats)
    return sorted(times)


# =============================================================================
# SpinFoam IR datatypes (intrinsic) + optional foliation
# =============================================================================

@dataclass
class FoamVertex:
    vid: int
    label: str = ""  # intrinsic label only (no time)

@dataclass
class FoamFace:
    fid: int
    face_type: str  # "note_component" | "glue_worldsheet"

    # note_component fields
    owner_edge: Optional[int] = None
    channel: Optional[str] = None  # harmonic|spectral|energy|phase
    j: Optional[float] = None
    m: Optional[float] = None
    flux: Optional[Tuple[float, float, float]] = None
    meta: Dict = field(default_factory=dict)

    # glue_worldsheet fields
    edge_a: Optional[int] = None
    edge_b: Optional[int] = None
    v_start: Optional[int] = None
    v_end: Optional[int] = None

@dataclass
class FoamEdge:
    eid: int
    note: Dict[str, int]  # pitch, velocity, channel

    v_start: int
    v_end: Optional[int]  # open if None

    component_faces: List[int] = field(default_factory=list)

    intertwiner: Dict = field(default_factory=dict)

    # optional embedding (not intrinsic)
    embedding: Dict = field(default_factory=dict)

@dataclass
class FoliationSlice:
    sid: int
    embedding_time_beats: float
    gamma_nodes: List[int]              # active note-edges
    gamma_links: List[Tuple[int, int]]  # glue links (spin-network edges)

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


# =============================================================================
# (j,m) + flux design for note-component faces (4 channels)
# =============================================================================

CHANNELS = ("harmonic", "spectral", "energy", "phase")

def harmonic_stability_j(pitch: int) -> float:
    """
    Toy but structured: map pitch class to 'stability class' => j.
    Replace freely.
    """
    pc = pitch % 12
    # consonant-ish classes (C, E, F, G, A) as lower j
    stable = {0, 4, 5, 7, 9}
    return 1.0 if pc in stable else 2.5

def spectral_register_j(pitch: int) -> float:
    """
    Low register => larger coupling complexity.
    """
    # bins: low/mid/high/very_high
    if pitch < 36:
        return 3.0
    if pitch < 60:
        return 2.5
    if pitch < 84:
        return 2.0
    return 1.5

def energy_j(vel: int) -> float:
    # bins into half-integers
    if vel < 20:
        return 0.5
    if vel < 50:
        return 1.5
    if vel < 80:
        return 2.5
    if vel < 110:
        return 3.5
    return 4.5

def phase_coherence_j(onset_beats: float) -> float:
    """
    Coherence proxy: closer to integer beat => smaller j (more coherent).
    This is embedding-derived but used only to label the face; you may remove or replace.
    """
    frac = abs(onset_beats - round(onset_beats))
    if frac < 0.05:
        return 0.5
    if frac < 0.20:
        return 1.5
    return 2.5

def m_from_direction(pitch: int, ref: int = 60) -> float:
    """
    Orientation proxy: above reference -> positive, below -> negative.
    """
    return 0.5 if pitch >= ref else -0.5

def m_from_channel(ch: int) -> float:
    return (-0.5, 0.0, 0.5)[ch % 3]

def m_from_attack(vel: int) -> float:
    return 0.5 if vel >= 80 else -0.5

def m_from_phase(onset_beats: float) -> float:
    return -0.5 if (onset_beats - round(onset_beats)) < 0 else 0.5

def face_flux_from_jm(j: float, m: float, channel: str, note: NoteKey, onset_beats: float) -> Tuple[float, float, float]:
    """
    Turn (j,m) into a 3D flux vector J.
    In SU(2) intuition, |J| ~ sqrt(j(j+1)). We also use channel to set a preferred direction (anisotropy).
    """
    # magnitude
    r = math.sqrt(max(0.0, j * (j + 1.0)))

    # orientation encoding:
    # - use m/j as cos(theta) (projection fraction) clamped
    if j <= 0:
        theta = math.pi / 2
    else:
        cos_theta = clamp(m / j, -1.0, 1.0)
        theta = math.acos(cos_theta)

    # phi depends on channel and note attributes (deterministic, not random)
    base = {
        "harmonic": 0.0,
        "spectral": 2.0,
        "energy": 4.0,
        "phase": 5.0,
    }[channel]
    phi = (base + 0.07 * (note.pitch % 12) + 0.11 * (note.channel) + 0.13 * (onset_beats % 1.0)) % (2 * math.pi)

    return sph_to_vec(r, theta, phi)

def make_note_component_faces(note: NoteKey, onset_beats: float) -> Dict[str, Dict]:
    """
    Return dict channel -> {j,m,flux,meta}, with j,m half-integer and |m|<=j.
    This implements all 4 channels.
    """
    pitch, vel, ch = note.pitch, note.velocity, note.channel

    raw = {}
    raw["harmonic"] = (harmonic_stability_j(pitch), m_from_direction(pitch), {"pitch_class": pitch % 12})
    raw["spectral"] = (spectral_register_j(pitch), m_from_channel(ch), {"register": "low" if pitch < 60 else "high"})
    raw["energy"] = (energy_j(vel), m_from_attack(vel), {"velocity": vel})
    raw["phase"] = (phase_coherence_j(onset_beats), m_from_phase(onset_beats), {"onset_frac": float(abs(onset_beats - round(onset_beats)))})

    out = {}
    for channel, (j, m, meta) in raw.items():
        j = q_half(j)
        # ensure j >= 0
        j = max(0.0, j)
        # quantize m and project to |m|<=j
        m = q_half(m)
        m = clamp(m, -j, j)
        flux = face_flux_from_jm(j, m, channel, note, onset_beats)
        out[channel] = {"j": j, "m": m, "flux": flux, "meta": meta}

    return out


# =============================================================================
# Intertwiner models (more "geometric / physical")
# =============================================================================

def admissible_triple(j1: float, j2: float, j3: float) -> bool:
    """
    SU(2) triangle inequalities + integrality (half-integer parity consistency)
    Conditions:
      |j1-j2| <= j3 <= j1+j2
      (j1+j2+j3) is integer (i.e., 2*(sum) is even)
    """
    if j3 < abs(j1 - j2) - 1e-9:
        return False
    if j3 > (j1 + j2) + 1e-9:
        return False
    # parity: 2*(j1+j2+j3) integer and even
    s2 = int(round(2 * (j1 + j2 + j3)))
    return (s2 % 2) == 0

def possible_k_range(j1: float, j2: float) -> List[float]:
    """
    Allowed intermediate spins k in j1 ⊗ j2, step 1.
    """
    lo = abs(j1 - j2)
    hi = j1 + j2
    # k increments by 1 in SU(2) fusion
    # but if j1,j2 are half-integers, lo/hi share parity; step 1 is correct.
    k = lo
    out = []
    # nudge to nearest half-integer
    k = q_half(k)
    while k <= hi + 1e-9:
        # fusion parity is automatically satisfied by step-1 if endpoints are consistent,
        # but we keep it simple here
        out.append(k)
        k += 1.0
    return out

def intertwiner_fusion_4valent(js: List[float]) -> Dict:
    """
    4-valent intertwiner existence test:
    There exists k such that:
      k ∈ (j1⊗j2) AND k ∈ (j3⊗j4)
    plus parity constraints.
    """
    j1, j2, j3, j4 = js
    k12 = set(possible_k_range(j1, j2))
    k34 = set(possible_k_range(j3, j4))
    inter = sorted(list(k12.intersection(k34)))
    return {
        "type": "fusion_4valent",
        "exists": len(inter) > 0,
        "k_candidates": inter[:10],  # keep short
    }

def intertwiner_vector_closure(fluxes: List[Tuple[float, float, float]], tol: float = 0.75) -> Dict:
    """
    Vector closure: Σ J_i ≈ 0
    """
    s = (0.0, 0.0, 0.0)
    for J in fluxes:
        s = v_add(s, J)
    err = v_norm(s)
    return {
        "type": "vector_closure",
        "closure_vector": s,
        "closure_error": float(err),
        "closed": err <= tol,
        "tol": tol,
    }

def intertwiner_constraint_gate(face_data: Dict[str, Dict]) -> Dict:
    """
    Physically motivated gates across 4 channels (a 'reality filter'):
    - If phase incoherent (high j_phase) AND spectral complexity high (high j_spectral),
      then reject closure (note can't stabilize).
    - If energy too low but harmonic tension high, reject (weak + tense cannot sustain).
    These are modeling choices, but they make geometry nontrivial.
    """
    j_h = face_data["harmonic"]["j"]
    j_s = face_data["spectral"]["j"]
    j_e = face_data["energy"]["j"]
    j_p = face_data["phase"]["j"]

    reasons = []
    ok = True

    if (j_p >= 2.5) and (j_s >= 2.5):
        ok = False
        reasons.append("phase_incoherent_and_spectral_complex")

    if (j_e <= 1.0) and (j_h >= 2.5):
        ok = False
        reasons.append("energy_too_low_for_harmonic_tension")

    return {
        "type": "constraint_gate",
        "admissible": ok,
        "reasons": reasons
    }

def build_intertwiner(face_data: Dict[str, Dict], mode: str = "hybrid") -> Dict:
    """
    Build an intertwiner for a note-edge from its 4 component faces.
    mode ∈ {"vector_closure","fusion_4valent","constraint_gate","hybrid"}
    """
    js = [face_data[ch]["j"] for ch in CHANNELS]
    ms = [face_data[ch]["m"] for ch in CHANNELS]
    fluxes = [face_data[ch]["flux"] for ch in CHANNELS]

    # m constraint (projection consistency)
    # In an invariant intertwiner, total m can be 0 in suitable coupling.
    # Here we enforce it approximately (model choice).
    sum_m = float(sum(ms))
    m_ok = abs(sum_m) <= 1.0  # half-integer tolerant gate

    out = {"mode": mode, "sum_m": sum_m, "m_ok": m_ok}

    if mode == "vector_closure":
        out["vector_closure"] = intertwiner_vector_closure(fluxes)
        out["closed"] = out["vector_closure"]["closed"] and m_ok
        return out

    if mode == "fusion_4valent":
        out["fusion"] = intertwiner_fusion_4valent(js)
        out["closed"] = out["fusion"]["exists"] and m_ok
        return out

    if mode == "constraint_gate":
        out["gate"] = intertwiner_constraint_gate(face_data)
        out["closed"] = out["gate"]["admissible"] and m_ok
        return out

    # hybrid (recommended)
    out["gate"] = intertwiner_constraint_gate(face_data)
    out["fusion"] = intertwiner_fusion_4valent(js)
    out["vector_closure"] = intertwiner_vector_closure(fluxes)
    out["closed"] = (
        out["gate"]["admissible"]
        and out["fusion"]["exists"]
        and out["vector_closure"]["closed"]
        and m_ok
    )
    return out


# =============================================================================
# Same-slice glue links (spin network) and glue-worldsheet faces
# =============================================================================

def harmonic_score(p1: int, p2: int) -> float:
    """
    Higher score => stronger glue.
    """
    interval = abs(p1 - p2) % 12
    # consonance map
    consonant = {
        0: 3.0, 7: 2.6, 5: 2.3, 4: 2.2, 3: 1.9, 9: 1.8, 8: 1.7,
        2: 1.2, 10: 1.2, 1: 0.6, 11: 0.6, 6: 0.2
    }
    return consonant.get(interval, 1.0)

def build_glue_links(active_eids: List[int], edges: List[FoamEdge], top_k: int = 2) -> List[Tuple[int, int]]:
    """
    Sparse top-k per node by harmonic_score.
    """
    if len(active_eids) < 2:
        return []
    scores: Dict[Tuple[int, int], float] = {}
    for i in range(len(active_eids)):
        for j in range(i + 1, len(active_eids)):
            a = active_eids[i]
            b = active_eids[j]
            p1 = edges[a].note["pitch"]
            p2 = edges[b].note["pitch"]
            scores[(a, b)] = harmonic_score(p1, p2)

    incident = defaultdict(list)
    for (a, b), sc in scores.items():
        incident[a].append((sc, (a, b)))
        incident[b].append((sc, (a, b)))

    kept: Set[Tuple[int, int]] = set()
    for node, lst in incident.items():
        lst.sort(key=lambda x: x[0], reverse=True)
        for _, e in lst[:top_k]:
            kept.add(e)

    out = []
    for a, b in kept:
        out.append((a, b) if a < b else (b, a))
    return sorted(set(out))

def glue_face_label(edges: List[FoamEdge], a: int, b: int) -> Dict:
    """
    (j,m,flux) for glue-worldsheet face (rep label on face in spin foam sense).
    We use:
      j = quantized consonance score
      m = orientation by register direction
      flux = encode in 3D using the same SU(2) magnitude rule
    """
    p1 = edges[a].note["pitch"]
    p2 = edges[b].note["pitch"]
    sc = harmonic_score(p1, p2)
    j = q_half(sc)          # half-integer label
    m = 0.5 if p1 >= p2 else -0.5
    m = clamp(m, -j, j)
    # channel is "glue"; use deterministic phi from pitches
    theta = math.acos(clamp(m / j, -1.0, 1.0)) if j > 0 else (math.pi / 2)
    phi = (0.21 * (p1 % 12) + 0.37 * (p2 % 12)) % (2 * math.pi)
    flux = sph_to_vec(math.sqrt(max(0.0, j * (j + 1.0))), theta, phi)
    return {"j": j, "m": m, "flux": flux, "meta": {"interval_mod12": abs(p1 - p2) % 12, "score": sc}}


# =============================================================================
# Intrinsic vertex definition: rewrite points (not ticks)
# =============================================================================

@dataclass(frozen=True)
class SliceSignature:
    active_edges: Tuple[int, ...]
    glue_links: Tuple[Tuple[int, int], ...]

def signature_of_slice(active_eids: List[int], links: List[Tuple[int, int]]) -> SliceSignature:
    return SliceSignature(
        active_edges=tuple(sorted(active_eids)),
        glue_links=tuple(sorted((min(a,b), max(a,b)) for a,b in links))
    )


# =============================================================================
# Compiler
# =============================================================================

def compile_midi_to_spinfoam_ir(
    midi_path: str,
    output_json: Optional[str] = None,
    include_foliation: bool = True,
    glue_top_k: int = 2,
    intertwiner_mode: str = "hybrid",
    closure_tol: float = 0.75,
) -> MIDISpinFoamIR:
    """
    Full compiler:
      - edges: one per MIDI note instance
      - note-component faces: 4 per edge with (j,m,flux)
      - edge intertwiner: hybrid gate + fusion + vector closure + m-sum
      - optional foliation: Γ slices
      - intrinsic vertices: rewrite regions (compressed identical signatures)
      - glue-worldsheet faces between rewrite regions
    """
    notes, bpm, tpb = parse_midi_file(midi_path)
    times = detect_state_change_times(notes)

    ir = MIDISpinFoamIR(
        title=midi_path,
        tempo_bpm=bpm,
        ticks_per_beat=tpb
    )

    # Build edges (note worldlines): one edge per MIDINote instance
    for eid, n in enumerate(notes):
        ir.edges.append(FoamEdge(
            eid=eid,
            note={"pitch": n.pitch, "velocity": n.velocity, "channel": n.channel},
            v_start=-1,
            v_end=None,
            component_faces=[],
            intertwiner={},
            embedding={"start_beats": n.start_beats, "end_beats": n.end_beats}
        ))

    # Slice -> active edges
    def active_edges_in_slice(sid: int) -> List[int]:
        t = times[sid]
        out = []
        for eid, n in enumerate(notes):
            if n.is_active_at(t):
                out.append(eid)
        return sorted(out)

    # Build optional foliation and slice signatures
    slice_links: List[List[Tuple[int, int]]] = []
    slice_signs: List[SliceSignature] = []
    for sid, t in enumerate(times):
        active_eids = active_edges_in_slice(sid)
        links = build_glue_links(active_eids, ir.edges, top_k=glue_top_k)
        slice_links.append(links)
        slice_signs.append(signature_of_slice(active_eids, links))
        if include_foliation:
            ir.foliation.append(FoliationSlice(
                sid=sid,
                embedding_time_beats=t,
                gamma_nodes=active_eids,
                gamma_links=links
            ))

    # Intrinsic vertices as rewrite regions (compress identical consecutive slice signatures)
    region_starts = [0]
    for sid in range(1, len(slice_signs)):
        if slice_signs[sid] != slice_signs[sid - 1]:
            region_starts.append(sid)

    # Create vertices (one per region)
    for rid in range(len(region_starts)):
        ir.vertices.append(FoamVertex(vid=rid, label=f"rewrite_region#{rid}"))

    # Map each slice sid -> region id
    region_of_slice: Dict[int, int] = {}
    rid = 0
    next_idx = 1
    next_boundary = region_starts[next_idx] if next_idx < len(region_starts) else None
    for sid in range(len(times)):
        if next_boundary is not None and sid >= next_boundary:
            rid += 1
            next_idx += 1
            next_boundary = region_starts[next_idx] if next_idx < len(region_starts) else None
        region_of_slice[sid] = rid

    # Assign edge endpoints in region vertex ids (intrinsic)
    for e in ir.edges:
        t0 = e.embedding["start_beats"]
        t1 = e.embedding["end_beats"]
        s_start = min(range(len(times)), key=lambda i: abs(times[i] - t0))
        s_end = min(range(len(times)), key=lambda i: abs(times[i] - t1))

        e.v_start = region_of_slice[s_start]
        e.v_end = region_of_slice[s_end] if s_end < len(times) - 1 else None

    # Build note-component faces (4 per edge) + intertwiner
    fid = 0
    for e in ir.edges:
        nk = NoteKey(e.note["pitch"], e.note["velocity"], e.note["channel"])
        onset = e.embedding["start_beats"]

        face_data = make_note_component_faces(nk, onset)
        # write component faces to IR
        for ch in CHANNELS:
            j = face_data[ch]["j"]
            m = face_data[ch]["m"]
            flux = face_data[ch]["flux"]
            meta = face_data[ch]["meta"]
            ir.faces.append(FoamFace(
                fid=fid,
                face_type="note_component",
                owner_edge=e.eid,
                channel=ch,
                j=j,
                m=m,
                flux=flux,
                meta=meta
            ))
            e.component_faces.append(fid)
            fid += 1

        # build intertwiner (hybrid by default)
        # allow user to set vector closure tolerance
        inter = build_intertwiner(face_data, mode=intertwiner_mode)
        # overwrite closure tol if using vector closure in out
        if "vector_closure" in inter:
            inter["vector_closure"]["tol"] = closure_tol
            inter["vector_closure"]["closed"] = inter["vector_closure"]["closure_error"] <= closure_tol
            # recompute closed for hybrid
            if intertwiner_mode == "hybrid":
                inter["closed"] = (
                    inter["gate"]["admissible"]
                    and inter["fusion"]["exists"]
                    and inter["vector_closure"]["closed"]
                    and inter["m_ok"]
                )
            elif intertwiner_mode == "vector_closure":
                inter["closed"] = inter["vector_closure"]["closed"] and inter["m_ok"]
        e.intertwiner = inter

    # Build region representative glue links (first slice in region)
    region_rep_slice: Dict[int, int] = {}
    for sid in range(len(times)):
        r = region_of_slice[sid]
        if r not in region_rep_slice:
            region_rep_slice[r] = sid

    region_links: Dict[int, Set[Tuple[int, int]]] = {}
    for r, sid in region_rep_slice.items():
        region_links[r] = set(slice_links[sid])

    # Glue-worldsheet faces between consecutive regions: persistent links sweep faces
    for r in range(len(ir.vertices) - 1):
        common = region_links.get(r, set()).intersection(region_links.get(r + 1, set()))
        for (a, b) in sorted(common):
            lbl = glue_face_label(ir.edges, a, b)
            ir.faces.append(FoamFace(
                fid=fid,
                face_type="glue_worldsheet",
                edge_a=a,
                edge_b=b,
                v_start=r,
                v_end=r + 1,
                j=lbl["j"],
                m=lbl["m"],
                flux=lbl["flux"],
                meta=lbl["meta"]
            ))
            fid += 1

    if output_json:
        ir.serialize(output_json, include_foliation=include_foliation)

    return ir


# =============================================================================
# IR search helpers (how to find j,m and coupling in IR)
# =============================================================================

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

