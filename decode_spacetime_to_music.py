import json
import math
from collections import defaultdict, Counter

import numpy as np
import networkx as nx
import mido
from mido import MidiFile, MidiTrack, Message


# ============================================================
# 1) Load spacetime in multiple schemas
# ============================================================

def load_spacetime(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if "points" in data:
        points = np.array(data["points"], dtype=float)
    elif "vertex_positions" in data:
        vp = data["vertex_positions"]
        # keys are strings of ints
        idxs = sorted([int(k) for k in vp.keys()])
        points = np.array([vp[str(i)] for i in idxs], dtype=float)
    else:
        raise ValueError("JSON must contain 'points' or 'vertex_positions'.")

    if "triangles" not in data:
        raise ValueError("JSON must contain 'triangles'.")

    triangles_raw = data["triangles"]
    triangles = []
    for tri in triangles_raw:
        # tri entries might be strings
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        triangles.append((a, b, c))

    return points, triangles, data


# ============================================================
# 2) Build topology graph and degrees
# ============================================================

def build_graph(n, triangles):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, c in triangles:
        G.add_edge(a, b)
        G.add_edge(b, c)
        G.add_edge(a, c)
    return G


# ============================================================
# 3) Helpers: normalization + entropy
# ============================================================

def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def shannon_entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


# ============================================================
# 4) Key detection (simple K-S style profiles)
# ============================================================

# Krumhansl-Kessler-like pitch-class profiles (normalized later)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def rotate_profile(profile, k):
    return np.roll(profile, k)

def pc_histogram(pitches_midi):
    h = np.zeros(12, dtype=float)
    for p in pitches_midi:
        h[int(p) % 12] += 1.0
    if h.sum() > 0:
        h = h / h.sum()
    return h

def cosine_sim(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def detect_key(pitches_midi):
    """
    Returns (tonic_pc, mode_str, score)
    mode_str in {"major", "minor"}
    """
    hist = pc_histogram(pitches_midi)

    maj = _MAJOR_PROFILE / _MAJOR_PROFILE.sum()
    minr = _MINOR_PROFILE / _MINOR_PROFILE.sum()

    best = (-1e9, 0, "major")  # (score, tonic_pc, mode)
    for tonic in range(12):
        smaj = cosine_sim(hist, rotate_profile(maj, tonic))
        smin = cosine_sim(hist, rotate_profile(minr, tonic))
        if smaj > best[0]:
            best = (smaj, tonic, "major")
        if smin > best[0]:
            best = (smin, tonic, "minor")

    return best[1], best[2], best[0]


# ============================================================
# 5) Scale snapping
# ============================================================

_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}  # natural minor

def snap_pitch_to_scale(pitch_midi, tonic_pc, mode):
    """
    Keep pitch close but force pitch-class into the detected scale.
    """
    pc = pitch_midi % 12
    scale = _MAJOR_SCALE if mode == "major" else _MINOR_SCALE
    scale_pcs = {(tonic_pc + s) % 12 for s in scale}

    if pc in scale_pcs:
        return pitch_midi

    # find nearest pitch (up/down) that lands in scale
    best_pitch = pitch_midi
    best_dist = 1e9
    for delta in range(-6, 7):
        candidate = pitch_midi + delta
        if (candidate % 12) in scale_pcs:
            d = abs(delta)
            if d < best_dist:
                best_dist = d
                best_pitch = candidate
    return best_pitch


# ============================================================
# 6) Beat grid quantization + onset grouping
# ============================================================

def quantize_to_grid(ticks, grid_step):
    # nearest grid multiple
    return int(round(ticks / grid_step) * grid_step)

def group_by_onset(notes):
    """
    notes: list of dict with "tick" and "salience"
    returns dict tick -> list of notes at that tick
    """
    buckets = defaultdict(list)
    for n in notes:
        buckets[n["tick"]].append(n)
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))


# ============================================================
# 7) Geometry -> candidate notes
# ============================================================

def geometry_to_candidate_notes(points, triangles, *,
                                ticks_per_beat=480,
                                bars=8,
                                pitch_range=(36, 84),
                                grid_div=16,
                                topk=3,
                                time_dim=0, pitch_dim=1, weight_dim=2,
                                enable_key_detect=True,
                                canonicalize_key_to_C=True,
                                target_median_pitch=60):
    """
    Returns finalized note events ready for MIDI writing.
    """

    n = len(points)
    G = build_graph(n, triangles)
    degrees = dict(G.degree())

    # raw coords
    t_raw = points[:, time_dim]
    if bars is None:
        # Estimate bars from max time value
        # Assuming t_raw is in ticks originally
        max_ticks = float(np.max(t_raw))
        bars = max(4, int(np.ceil(max_ticks / (ticks_per_beat * 4))))
    p_raw = points[:, pitch_dim]

    t = normalize01(t_raw)
    p = normalize01(p_raw)

    if points.shape[1] > weight_dim:
        w_raw = points[:, weight_dim]
        w = normalize01(w_raw)
    else:
        w = np.ones(n, dtype=float) * 0.5

    # # map time into ticks
    # total_ticks = int(ticks_per_beat * bars * 4)  # 4 beats per bar
    # ticks = (t * total_ticks).astype(int)
    
    # DON'T normalize time - use it directly if in ticks
    # Or scale proportionally while preserving duration
    t_scale = (bars * 4 * ticks_per_beat) / (np.max(t_raw) - np.min(t_raw) + 1e-6)
    ticks = ((t_raw - np.min(t_raw)) * t_scale).astype(int)

    grid_step = max(1, int(ticks_per_beat * 4 / grid_div))  # grid_div=16 => 1/16 note

    # map pitch into midi range
    # lo, hi = pitch_range
    # pitches = lo + p * (hi - lo)
    
     # For pitch: preserve octave relationships
    # Instead of normalize01, preserve the original pitch structure
    p_raw_normalized = (p_raw - np.min(p_raw)) / (np.max(p_raw) - np.min(p_raw) + 1e-6)
    lo, hi = pitch_range
    pitches = lo + p_raw_normalized * (hi - lo)
    
    pitches = pitches.astype(int)
    
    

    # velocity from weight
    velocities = np.clip((w * 90) + 30, 20, 120).astype(int)

    # salience from topology + weight (explainable!)
    # log1p stabilizes degree scale
    sal = np.array([math.log1p(degrees[i]) for i in range(n)], dtype=float) + 0.8 * w

    # build candidate notes
    candidates = []
    for i in range(n):
        tick_q = quantize_to_grid(int(ticks[i]), grid_step)
        candidates.append({
            "vid": i,
            "tick": tick_q,
            "pitch": int(pitches[i]),
            "vel": int(velocities[i]),
            "salience": float(sal[i]),
        })

    # group by onset and keep top-K
    buckets = group_by_onset(candidates)
    selected = []
    for tick, items in buckets.items():
        items.sort(key=lambda x: x["salience"], reverse=True)
        for it in items[:max(1, topk)]:
            selected.append(it)

    # key detect + scale snapping (based on selected notes)
    selected_pitches = [n["pitch"] for n in selected]

    if enable_key_detect and len(selected_pitches) >= 12:
        tonic, mode, key_score = detect_key(selected_pitches)
    else:
        tonic, mode, key_score = 0, "major", 0.0

    # optionally canonicalize key to C (gauge-fix for transposition symmetry)
    # tonic_pc -> 0 means C
    if canonicalize_key_to_C:
        shift_pc = (-tonic) % 12
        for n in selected:
            n["pitch"] += shift_pc
        tonic = 0  # now in C or A-min like space (mode preserved)

    # snap to scale
    for n in selected:
        n["pitch"] = int(snap_pitch_to_scale(int(n["pitch"]), tonic, mode))

    # second gauge-fix: set median register to target_median_pitch (keeps things in playable range)
    med = int(np.median([n["pitch"] for n in selected])) if selected else target_median_pitch
    shift_reg = target_median_pitch - med
    for n in selected:
        n["pitch"] = int(np.clip(n["pitch"] + shift_reg, 24, 108))

    # duration: one grid step, with mild extension from salience
    base_dur = grid_step
    for n in selected:
        # extend up to ~3 steps based on salience
        ext = int(min(2 * grid_step, n["salience"] * 0.8 * grid_step))
        n["dur"] = int(base_dur + ext)

    # sort final notes by time
    selected.sort(key=lambda x: (x["tick"], -x["salience"]))

    meta = {
        "ticks_per_beat": ticks_per_beat,
        "bars": bars,
        "grid_div": grid_div,
        "grid_step": grid_step,
        "topk": topk,
        "pitch_range": pitch_range,
        "detected_key": f"{_NOTE_NAMES[tonic]} {mode}",
        "key_fit_score": float(key_score),
        "canonicalize_key_to_C": bool(canonicalize_key_to_C),
        "register_shift": int(shift_reg),
    }

    return selected, meta


# ============================================================
# 8) Music-likeness score (lightweight + explainable)
# ============================================================

def compute_music_score(selected_notes, *, tonic_pc=0, mode="major"):
    """
    selected_notes: list with fields tick, pitch, dur
    returns dict of sub-scores + total
    """

    if not selected_notes:
        return {"total": 0.0, "rhythm": 0.0, "melody": 0.0, "tonal": 0.0, "repeat": 0.0}

    # rhythm: IOI entropy (lower entropy is more regular -> better),
    # but too-low (all same) also can be boring; we keep a simple target.
    ticks = sorted({n["tick"] for n in selected_notes})
    iois = [ticks[i+1] - ticks[i] for i in range(len(ticks)-1)]
    ioi_counts = Counter(iois)
    ent = shannon_entropy(list(ioi_counts.values()))
    # map entropy to score (heuristic)
    rhythm = float(max(0.0, 1.0 - ent / 2.5))  # 2.5 is a rough scale

    # melody: take top1 per tick and score leap smoothness
    by_tick = defaultdict(list)
    for n in selected_notes:
        by_tick[n["tick"]].append(n)
    melody = []
    for t in sorted(by_tick.keys()):
        # choose highest velocity as melody proxy
        by_tick[t].sort(key=lambda x: (x["vel"], x["salience"]), reverse=True)
        melody.append(by_tick[t][0]["pitch"])
    leaps = [abs(melody[i+1] - melody[i]) for i in range(len(melody)-1)]
    if leaps:
        big = sum(1 for d in leaps if d >= 8) / len(leaps)
        melody_score = float(max(0.0, 1.0 - 1.2 * big))
    else:
        melody_score = 0.5

    # tonal: key-fit of pitch class histogram (C major/minor after canonicalization if you used it)
    pcs = pc_histogram([n["pitch"] for n in selected_notes])
    maj = _MAJOR_PROFILE / _MAJOR_PROFILE.sum()
    minr = _MINOR_PROFILE / _MINOR_PROFILE.sum()
    tonal = max(cosine_sim(pcs, rotate_profile(maj, tonic_pc)),
                cosine_sim(pcs, rotate_profile(minr, tonic_pc)))
    tonal = float(max(0.0, tonal))

    # repetition: interval bigram repetition ratio (transposition-invariant)
    intervals = [(melody[i+1] - melody[i]) for i in range(len(melody)-1)]
    if len(intervals) >= 4:
        bigrams = [(intervals[i], intervals[i+1]) for i in range(len(intervals)-1)]
        c = Counter(bigrams)
        rep = sum(v for v in c.values() if v >= 2) / max(1, len(bigrams))
        repeat = float(min(1.0, rep))
    else:
        repeat = 0.0

    # weighted total (tune later)
    total = 0.30 * rhythm + 0.30 * melody_score + 0.25 * tonal + 0.15 * repeat

    return {
        "total": float(total),
        "rhythm": float(rhythm),
        "melody": float(melody_score),
        "tonal": float(tonal),
        "repeat": float(repeat),
    }


# ============================================================
# 9) MIDI writing (polyphonic, simple)
# ============================================================

def write_midi(selected_notes, output_path, *, ticks_per_beat=480):
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)

    # Build a flat event list: (tick, type, pitch, velocity)
    events = []
    for n in selected_notes:
        t0 = int(n["tick"])
        t1 = int(n["tick"] + n["dur"])
        pitch = int(n["pitch"])
        vel = int(n["vel"])
        events.append((t0, "on", pitch, vel))
        events.append((t1, "off", pitch, 0))

    # Sort events, note_off before note_on at same tick
    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

    last_tick = 0
    for tick, typ, pitch, vel in events:
        delta = max(0, tick - last_tick)
        if typ == "on":
            track.append(Message("note_on", note=pitch, velocity=vel, time=delta))
        else:
            track.append(Message("note_off", note=pitch, velocity=0, time=delta))
        last_tick = tick

    midi.save(output_path)
    print(f"✅ MIDI saved to: {output_path}")


# ============================================================
# 10) Pipeline
# ============================================================

def decode_spacetime_json_to_midi(
    json_path,
    out_midi="decoded_v2.mid",
    *,
    ticks_per_beat=480,
    bars=None,
    pitch_range=(36, 84),
    grid_div=16,
    topk=3,
    enable_key_detect=True,
    canonicalize_key_to_C=True,
    target_median_pitch=60,
):
    points, triangles, data = load_spacetime(json_path)

    selected, meta = geometry_to_candidate_notes(
        points, triangles,
        ticks_per_beat=ticks_per_beat,
        bars=bars,
        pitch_range=pitch_range,
        grid_div=grid_div,
        topk=topk,
        enable_key_detect=enable_key_detect,
        canonicalize_key_to_C=canonicalize_key_to_C,
        target_median_pitch=target_median_pitch,
    )

    # score (tonic_pc=0 if canonicalized)
    score = compute_music_score(selected, tonic_pc=0, mode="major")

    print("\n--- Decoding summary ---")
    print(f"Vertices: {len(points)} | Triangles: {len(triangles)}")
    print(f"Selected notes: {len(selected)}  (topK={meta['topk']}, grid=1/{grid_div})")
    print(f"Key detect: {meta['detected_key']}  (fit={meta['key_fit_score']:.3f})")
    print(f"Register shift: {meta['register_shift']}")
    print("Music-likeness score:")
    print(f"  total={score['total']:.3f} | rhythm={score['rhythm']:.3f} | melody={score['melody']:.3f} | tonal={score['tonal']:.3f} | repeat={score['repeat']:.3f}")

    write_midi(selected, out_midi, ticks_per_beat=ticks_per_beat)

    return {"meta": meta, "score": score}


# ============================================================
# 11) CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Input spacetime JSON")
    parser.add_argument("--out", default="decoded_v2.mid", help="Output MIDI path")

    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--tpb", type=int, default=480)
    parser.add_argument("--grid", type=int, default=16, help="Grid division per whole note (16 => 1/16 note)")
    parser.add_argument("--topk", type=int, default=3, help="Max notes per onset (polyphony)")
    parser.add_argument("--pitch_lo", type=int, default=36)
    parser.add_argument("--pitch_hi", type=int, default=84)
    parser.add_argument("--no_key_detect", action="store_true")
    parser.add_argument("--no_canon_C", action="store_true")
    parser.add_argument("--target_median", type=int, default=60)

    args = parser.parse_args()

    decode_spacetime_json_to_midi(
        args.json,
        out_midi=args.out,
        ticks_per_beat=args.tpb,
        bars=args.bars,
        pitch_range=(args.pitch_lo, args.pitch_hi),
        grid_div=args.grid,
        topk=args.topk,
        enable_key_detect=(not args.no_key_detect),
        canonicalize_key_to_C=(not args.no_canon_C),
        target_median_pitch=args.target_median,
    )
