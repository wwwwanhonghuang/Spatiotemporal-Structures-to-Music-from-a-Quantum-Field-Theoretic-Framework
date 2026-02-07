# decode_spacetime_to_music_v4.py
# ------------------------------------------------------------
# Topology-first decoder (NO metric time in IR):
# - "Time" is an observer gauge: user chooses duration/tempo/grid.
# - Ordering (proto-time) is derived from graph topology (triangles -> edges)
#   using either:
#     (A) spectral ordering (Fiedler vector), or
#     (B) biased random walk (stochastic traversal).
#
# Supported JSON schemas:
#   A) {"points": [...], "triangles": [...], ...}
#   B) {"vertex_positions": {...}, "triangles": [...], "vertex_weights": {...}}
#
# CLI examples:
#   python decode_spacetime_to_music_v4.py --json in.json --out out.mid --seconds 30 --bpm 120 --grid 16 --mode spectral
#   python decode_spacetime_to_music_v4.py --json in.json --out out.mid --bars 16 --bpm 90 --grid 16 --mode walk --seed 7
#
# ------------------------------------------------------------

import json
import math
import random
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
import mido
from mido import MidiFile, MidiTrack, Message

# -------------------------
# Music theory helpers
# -------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)

_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}

def cosine_sim(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def rotate_profile(profile, k):
    return np.roll(profile, k)

def pc_histogram(pitches_midi: List[int]) -> np.ndarray:
    h = np.zeros(12, dtype=float)
    for p in pitches_midi:
        h[int(p) % 12] += 1.0
    if h.sum() > 0:
        h /= h.sum()
    return h

def detect_key(pitches_midi: List[int]) -> Tuple[int, str, float]:
    if not pitches_midi:
        return 0, "major", 0.0
    hist = pc_histogram(pitches_midi)
    maj = _MAJOR_PROFILE / _MAJOR_PROFILE.sum()
    minr = _MINOR_PROFILE / _MINOR_PROFILE.sum()
    best_score = -1e9
    best_tonic = 0
    best_mode = "major"
    for tonic in range(12):
        smaj = cosine_sim(hist, rotate_profile(maj, tonic))
        smin = cosine_sim(hist, rotate_profile(minr, tonic))
        if smaj > best_score:
            best_score, best_tonic, best_mode = smaj, tonic, "major"
        if smin > best_score:
            best_score, best_tonic, best_mode = smin, tonic, "minor"
    return int(best_tonic), best_mode, float(best_score)

def snap_pitch_to_scale(pitch_midi: int, tonic_pc: int, mode: str) -> int:
    scale = _MAJOR_SCALE if mode == "major" else _MINOR_SCALE
    allowed = {(tonic_pc + s) % 12 for s in scale}
    pc = pitch_midi % 12
    if pc in allowed:
        return int(pitch_midi)
    best = pitch_midi
    best_d = 1e9
    for delta in range(-6, 7):
        cand = pitch_midi + delta
        if (cand % 12) in allowed:
            d = abs(delta)
            if d < best_d:
                best_d, best = d, cand
    return int(best)

def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def quantize_to_grid(ticks: int, grid_step: int) -> int:
    return int(round(ticks / grid_step) * grid_step)

# -------------------------
# Data model
# -------------------------

@dataclass
class SpacetimeData:
    points: np.ndarray
    triangles: List[Tuple[int, int, int]]
    raw: Dict[str, Any]
    vertex_weights: Optional[Dict[str, float]] = None

@dataclass
class NoteEvent:
    tick: int
    pitch: int
    vel: int
    dur: int
    salience: float = 0.0

# -------------------------
# JSON loading
# -------------------------

def load_spacetime_json(path: str) -> SpacetimeData:
    with open(path, "r") as f:
        data = json.load(f)

    if "points" in data:
        points = np.array(data["points"], dtype=float)
    elif "vertex_positions" in data:
        vp = data["vertex_positions"]
        idxs = sorted([int(k) for k in vp.keys()])
        points = np.array([vp[str(i)] for i in idxs], dtype=float)
    else:
        raise KeyError("JSON must include 'points' or 'vertex_positions'.")

    if "triangles" not in data:
        raise KeyError("JSON must include 'triangles'.")

    triangles: List[Tuple[int, int, int]] = []
    for tri in data["triangles"]:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        triangles.append((a, b, c))

    return SpacetimeData(
        points=points,
        triangles=triangles,
        raw=data,
        vertex_weights=data.get("vertex_weights", None),
    )

# -------------------------
# Graph building
# -------------------------

def build_graph(n: int, triangles: List[Tuple[int, int, int]]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, c in triangles:
        G.add_edge(a, b)
        G.add_edge(b, c)
        G.add_edge(a, c)
    return G

def degrees(G: nx.Graph) -> np.ndarray:
    deg = np.zeros(G.number_of_nodes(), dtype=float)
    for i, d in G.degree():
        deg[i] = float(d)
    return deg

# -------------------------
# Ordering = proto-time
# -------------------------

def spectral_ordering(G: nx.Graph) -> List[int]:
    """
    Use the Fiedler vector (2nd smallest eigenvector of Laplacian)
    to order vertices along an emergent 1D flow coordinate.
    """
    n = G.number_of_nodes()
    if n <= 2:
        return list(range(n))

    # If graph is disconnected, handle components separately then concatenate
    comps = [list(c) for c in nx.connected_components(G)]
    if len(comps) > 1:
        ordered = []
        for comp in sorted(comps, key=len, reverse=True):
            sub = G.subgraph(comp)
            ordered.extend(_spectral_ordering_connected(sub))
        return ordered

    return _spectral_ordering_connected(G)

def _spectral_ordering_connected(G: nx.Graph) -> List[int]:
    nodes = list(G.nodes())
    n = len(nodes)
    if n <= 2:
        return nodes

    # Laplacian matrix
    L = nx.laplacian_matrix(G, nodelist=nodes).astype(float).toarray()

    # eigen-decomp
    vals, vecs = np.linalg.eigh(L)
    if len(vals) < 2:
        return nodes

    fiedler = vecs[:, 1]
    order_idx = np.argsort(fiedler)
    return [nodes[i] for i in order_idx]

def biased_random_walk_ordering(
    G: nx.Graph,
    *,
    steps: int,
    seed: int,
    bias_degree: float = 1.0
) -> List[int]:
    """
    Stochastic traversal. Produces a visitation order (with repeats),
    then compress to first-visit order for scheduling.
    bias_degree > 0 biases toward high degree nodes.
    """
    rng = random.Random(seed)
    n = G.number_of_nodes()
    if n == 0:
        return []

    nodes = list(G.nodes())
    deg = dict(G.degree())

    # start from a high-degree node (stable)
    start = max(nodes, key=lambda x: deg.get(x, 0))
    cur = start

    visited_seq = []
    for _ in range(max(1, steps)):
        visited_seq.append(cur)
        neigh = list(G.neighbors(cur))
        if not neigh:
            cur = rng.choice(nodes)
            continue

        # weights proportional to (deg^bias)
        weights = []
        for v in neigh:
            w = (deg.get(v, 1) ** bias_degree)
            weights.append(max(1e-6, float(w)))
        s = sum(weights)
        r = rng.random() * s
        acc = 0.0
        nxt = neigh[-1]
        for v, w in zip(neigh, weights):
            acc += w
            if acc >= r:
                nxt = v
                break
        cur = nxt

    # compress to first-visit order, then append any unvisited nodes
    seen = set()
    order = []
    for v in visited_seq:
        if v not in seen:
            seen.add(v)
            order.append(v)
    for v in nodes:
        if v not in seen:
            order.append(v)
    return order

# -------------------------
# Decode to MIDI events
# -------------------------

def decode_topology_to_events(
    sp: SpacetimeData,
    *,
    mode: str,
    ticks_per_beat: int,
    bpm: int,
    grid_div: int,
    seconds: Optional[float],
    bars: Optional[int],
    pitch_range: Tuple[int, int],
    max_polyphony: int,
    canonicalize_key_to_C: bool,
    enable_scale_snap: bool,
    target_median_pitch: Optional[int],
    seed: int,
) -> Tuple[List[NoteEvent], Dict[str, Any]]:
    points = sp.points
    n = len(points)
    G = build_graph(n, sp.triangles)
    deg = degrees(G)

    # ----------------------
    # 1) Observer time gauge
    # ----------------------
    if seconds is None and bars is None:
        raise ValueError("Provide either --seconds or --bars (duration is an observer parameter).")

    if seconds is not None:
        total_ticks = int(round(seconds * ticks_per_beat * (bpm / 60.0)))
        duration_desc = f"{seconds:.3f}s"
    else:
        total_ticks = int(ticks_per_beat * 4 * int(bars))
        duration_desc = f"{int(bars)} bars"

    grid_step = max(1, int(ticks_per_beat * 4 / max(1, grid_div)))

    # ----------------------
    # 2) Proto-time ordering
    # ----------------------
    if mode == "spectral":
        order = spectral_ordering(G)
        order_mode = "spectral_fiedler"
    elif mode == "walk":
        # steps slightly larger than n for nicer mixing
        steps = max(n * 3, 1000)
        order = biased_random_walk_ordering(G, steps=steps, seed=seed, bias_degree=1.0)
        order_mode = f"biased_walk(steps={steps},seed={seed})"
    else:
        raise ValueError("mode must be 'spectral' or 'walk'.")

    # Map rank -> tick (uniformly spread across duration)
    # (This is the “emergent clock”.)
    ranks = np.arange(len(order), dtype=float)
    if len(order) <= 1:
        tick_of_rank = np.array([0], dtype=int)
    else:
        tick_of_rank = (ranks / (len(order) - 1) * total_ticks).astype(int)
    tick_of_rank = np.array([quantize_to_grid(int(t), grid_step) for t in tick_of_rank], dtype=int)

    # ----------------------
    # 3) Pitch/velocity gauge (from latent coords/weights, not time)
    # ----------------------
    lo, hi = pitch_range

    # Use points[:,1] as "pitch-like latent" if available else degree
    if points.shape[1] >= 2:
        p_lat = points[:, 1]
        p_norm = normalize01(p_lat)
    else:
        p_norm = normalize01(deg)

    pitch_midi = (lo + p_norm * (hi - lo)).astype(int)

    # velocity from weights if present else points[:,2] else degree
    if sp.vertex_weights:
        w = np.zeros(n, dtype=float)
        for i in range(n):
            w[i] = float(sp.vertex_weights.get(str(i), 0.5))
        w = normalize01(w)
    elif points.shape[1] >= 3:
        w = normalize01(points[:, 2])
    else:
        w = normalize01(deg)

    vel = np.clip((w * 90 + 30).astype(int), 20, 120)

    # salience = log(deg) + weight
    sal = np.log1p(deg) + 0.8 * w

    # ----------------------
    # 4) Schedule note candidates by onset, control polyphony
    # ----------------------
    # We will form onsets by bucketed ticks, then choose top salience per onset.
    by_tick: Dict[int, List[int]] = defaultdict(list)
    for r, v in enumerate(order):
        t = int(tick_of_rank[r])
        by_tick[t].append(v)

    events: List[NoteEvent] = []
    for t in sorted(by_tick.keys()):
        vs = by_tick[t]
        # sort vertices by salience
        vs.sort(key=lambda i: float(sal[i]), reverse=True)
        chosen = vs[:max(1, int(max_polyphony))]
        # duration: one grid step, maybe slightly longer for salient notes
        for i in chosen:
            ext = int(min(2 * grid_step, float(sal[i]) * 0.6 * grid_step))
            dur = max(1, grid_step + ext)
            events.append(NoteEvent(
                tick=t,
                pitch=int(pitch_midi[i]),
                vel=int(vel[i]),
                dur=int(dur),
                salience=float(sal[i]),
            ))

    # ----------------------
    # 5) Gauge fixing: key / register (optional)
    # ----------------------
    # Key detection on produced pitches (observer may canonicalize)
    det_tonic, det_mode, det_score = detect_key([e.pitch for e in events])

    if canonicalize_key_to_C:
        shift = (-det_tonic) % 12
        for e in events:
            e.pitch += shift
        det_tonic = 0

    if enable_scale_snap:
        for e in events:
            e.pitch = snap_pitch_to_scale(int(e.pitch), det_tonic, det_mode)

    if target_median_pitch is not None and events:
        med = int(np.median([e.pitch for e in events]))
        shift_reg = int(target_median_pitch - med)
        for e in events:
            e.pitch = int(np.clip(e.pitch + shift_reg, 0, 127))
    else:
        shift_reg = 0

    # clamp final
    for e in events:
        e.pitch = int(np.clip(e.pitch, 0, 127))
        e.vel = int(np.clip(e.vel, 1, 127))
        e.dur = int(max(1, e.dur))

    # sort events
    events.sort(key=lambda x: (x.tick, -x.salience))

    meta = {
        "decoder": "topology_first_v4",
        "ordering": order_mode,
        "duration": duration_desc,
        "tempo_bpm": int(bpm),
        "ticks_per_beat": int(ticks_per_beat),
        "grid_div": int(grid_div),
        "grid_step": int(grid_step),
        "max_polyphony": int(max_polyphony),
        "pitch_range": [int(lo), int(hi)],
        "detected_key": f"{_NOTE_NAMES[det_tonic]} {det_mode}",
        "key_fit_score": float(det_score),
        "canonicalize_key_to_C": bool(canonicalize_key_to_C),
        "scale_snap": bool(enable_scale_snap),
        "target_median_pitch": target_median_pitch,
        "register_shift": int(shift_reg),
        "num_vertices": int(n),
        "num_triangles": int(len(sp.triangles)),
        "num_events": int(len(events)),
    }
    return events, meta

# -------------------------
# MIDI writing
# -------------------------

def write_midi(events: List[NoteEvent], out_path: str, *, ticks_per_beat: int, bpm: int):
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    flat = []
    for e in events:
        t0 = int(e.tick)
        t1 = int(e.tick + max(1, e.dur))
        flat.append((t0, 1, int(e.pitch), int(e.vel)))  # on
        flat.append((t1, 0, int(e.pitch), 0))          # off

    # off before on at same tick
    flat.sort(key=lambda x: (x[0], x[1]))

    last = 0
    for tick, typ, pitch, vel in flat:
        delta = max(0, tick - last)
        if typ == 1:
            track.append(Message("note_on", note=pitch, velocity=vel, time=delta))
        else:
            track.append(Message("note_off", note=pitch, velocity=0, time=delta))
        last = tick

    midi.save(out_path)
    print(f"✅ MIDI saved: {out_path}")

# -------------------------
# Simple diagnostics (optional)
# -------------------------

def music_likeness(events: List[NoteEvent]) -> Dict[str, float]:
    if not events:
        return {"total": 0.0, "rhythm": 0.0, "melody": 0.0, "tonal": 0.0}

    # rhythm: IOI entropy
    ticks = sorted(set(e.tick for e in events))
    iois = [ticks[i+1] - ticks[i] for i in range(len(ticks)-1)]
    c = Counter(iois)
    total = sum(c.values()) or 1
    ent = 0.0
    for v in c.values():
        p = v / total
        ent -= p * math.log(p + 1e-12)
    rhythm = max(0.0, 1.0 - ent / 2.5)

    # melody: take top salience per onset
    by = defaultdict(list)
    for e in events:
        by[e.tick].append(e)
    mel = []
    for t in sorted(by.keys()):
        by[t].sort(key=lambda x: (x.vel, x.salience), reverse=True)
        mel.append(by[t][0].pitch)
    if len(mel) >= 2:
        leaps = [abs(mel[i+1] - mel[i]) for i in range(len(mel)-1)]
        big = sum(1 for d in leaps if d >= 8) / len(leaps)
        melody = max(0.0, 1.0 - 1.2 * big)
    else:
        melody = 0.5

    # tonal: key-fit score
    tonic, mode, key_score = detect_key([e.pitch for e in events])
    tonal = max(0.0, min(1.0, key_score))

    total_score = 0.35 * rhythm + 0.35 * melody + 0.30 * tonal
    return {"total": float(total_score), "rhythm": float(rhythm), "melody": float(melody), "tonal": float(tonal)}

# -------------------------
# Pipeline
# -------------------------

def decode_json_to_midi_v4(
    json_path: str,
    out_midi: str,
    *,
    mode: str,
    bpm: int,
    ticks_per_beat: int,
    grid_div: int,
    seconds: Optional[float],
    bars: Optional[int],
    pitch_lo: int,
    pitch_hi: int,
    max_polyphony: int,
    canonicalize_key_to_C: bool,
    enable_scale_snap: bool,
    target_median_pitch: Optional[int],
    seed: int,
):
    sp = load_spacetime_json(json_path)

    events, meta = decode_topology_to_events(
        sp,
        mode=mode,
        ticks_per_beat=ticks_per_beat,
        bpm=bpm,
        grid_div=grid_div,
        seconds=seconds,
        bars=bars,
        pitch_range=(pitch_lo, pitch_hi),
        max_polyphony=max_polyphony,
        canonicalize_key_to_C=canonicalize_key_to_C,
        enable_scale_snap=enable_scale_snap,
        target_median_pitch=target_median_pitch,
        seed=seed,
    )

    score = music_likeness(events)

    print("\n--- Decode v4 summary (topology-first) ---")
    print(f"Vertices: {meta['num_vertices']} | Triangles: {meta['num_triangles']}")
    print(f"Ordering: {meta['ordering']}")
    print(f"Duration: {meta['duration']} | BPM={meta['tempo_bpm']} | TPB={meta['ticks_per_beat']}")
    print(f"Grid: 1/{meta['grid_div']} (step={meta['grid_step']} ticks) | Polyphony<= {meta['max_polyphony']}")
    print(f"Key: {meta['detected_key']} (fit={meta['key_fit_score']:.3f}) | canon_C={meta['canonicalize_key_to_C']} | snap={meta['scale_snap']}")
    print(f"Events: {meta['num_events']}")
    print(f"Music-likeness: total={score['total']:.3f} | rhythm={score['rhythm']:.3f} | melody={score['melody']:.3f} | tonal={score['tonal']:.3f}")

    write_midi(events, out_midi, ticks_per_beat=ticks_per_beat, bpm=bpm)

    report = {"meta": meta, "music_likeness": score}
    report_path = out_midi.rsplit(".", 1)[0] + "_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📝 Report saved: {report_path}")

# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--out", default="decoded_v4.mid")

    p.add_argument("--mode", choices=["spectral", "walk"], default="spectral")
    p.add_argument("--bpm", type=int, default=120)
    p.add_argument("--tpb", type=int, default=480)
    p.add_argument("--grid", type=int, default=16, help="Grid division per whole note (16 => 1/16)")

    # duration gauge (choose one)
    p.add_argument("--seconds", type=float, default=None, help="Observer-chosen duration in seconds")
    p.add_argument("--bars", type=int, default=None, help="Observer-chosen duration in bars")

    p.add_argument("--pitch_lo", type=int, default=36)
    p.add_argument("--pitch_hi", type=int, default=84)
    p.add_argument("--polyphony", type=int, default=4)

    p.add_argument("--canon_C", action="store_true", help="Gauge-fix transposition by canonicalizing to C")
    p.add_argument("--no_snap", action="store_true", help="Disable scale snapping")
    p.add_argument("--target_median", type=int, default=60, help="Register gauge (median pitch target). Use -1 to disable.")
    p.add_argument("--seed", type=int, default=0, help="Seed for walk mode")

    args = p.parse_args()

    target_median = None if args.target_median < 0 else int(args.target_median)

    decode_json_to_midi_v4(
        json_path=args.json,
        out_midi=args.out,
        mode=args.mode,
        bpm=args.bpm,
        ticks_per_beat=args.tpb,
        grid_div=args.grid,
        seconds=args.seconds,
        bars=args.bars,
        pitch_lo=args.pitch_lo,
        pitch_hi=args.pitch_hi,
        max_polyphony=args.polyphony,
        canonicalize_key_to_C=args.canon_C,
        enable_scale_snap=(not args.no_snap),
        target_median_pitch=target_median,
        seed=args.seed,
    )
