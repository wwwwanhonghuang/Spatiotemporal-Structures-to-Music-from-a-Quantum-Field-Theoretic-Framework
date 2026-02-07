
# =============================================================================
# Same-slice glue links (spin network) and glue-worldsheet faces
# =============================================================================
from typing import List, Dict, Tuple, Set
import math
from spacetime_ir.spinfoam_ir.foam_edge import FoamEdge
from utils.math_and_physics.math_and_physics import sph_to_vec, clamp, q_half
from collections import defaultdict


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
