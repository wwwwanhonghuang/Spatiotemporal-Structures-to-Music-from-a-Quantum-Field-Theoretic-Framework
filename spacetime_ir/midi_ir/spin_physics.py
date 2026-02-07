
# =============================================================================
# (j,m) + flux design for note-component faces (4 channels)
# =============================================================================
import math
from typing import Tuple, Dict
from spacetime_ir.midi_ir.note_key import NoteKey
from utils.math_and_physics.math_and_physics import clamp, sph_to_vec, q_half
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
