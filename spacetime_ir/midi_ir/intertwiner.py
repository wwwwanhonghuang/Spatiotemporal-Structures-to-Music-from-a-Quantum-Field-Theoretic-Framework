
# =============================================================================
# Intertwiner models (more "geometric / physical")
# =============================================================================
from spacetime_ir.midi_ir.spin_physics import CHANNELS
from typing import List, Dict, Tuple
from utils.math_and_physics.math_and_physics import v_add, v_norm, q_half

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