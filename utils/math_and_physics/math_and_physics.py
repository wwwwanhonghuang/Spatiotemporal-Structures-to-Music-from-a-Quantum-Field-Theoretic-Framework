import math

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