import math

_MAX_BINS = 64


def q_linear(v: float, n_bins: int, lo: float = 0.0, hi: float = 1.0) -> int:
    n = max(1, int(n_bins))
    if not math.isfinite(float(v)) or hi <= lo or n <= 1:
        return 0
    x = (float(v) - float(lo)) / (float(hi) - float(lo))
    b = int(math.floor(max(0.0, min(1.0 - 1e-12, x)) * n))
    return max(0, min(n - 1, b))


def _q_linear_safe(v: float, n_bins: int, lo: float = 0.0, hi: float = 1.0) -> int:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return 0
    return q_linear(float(v), int(n_bins), float(lo), float(hi))


def _q_unsigned(v: float, step: float, max_bins: int = 9) -> int:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or step <= 0.0:
        return 0
    b = int(math.floor(max(0.0, float(v)) / float(step)))
    return max(0, min(int(max_bins), b))


def _q_signed(v: float, max_bins: int = 16) -> int:
    n = max(1, int(max_bins))
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        x = 0.0
    else:
        x = max(-1.0, min(1.0, float(v)))
    b = int(math.floor(((x + 1.0) / 2.0) * n))
    return max(0, min(n - 1, b))


def _aang(a: float, b: float) -> float:
    d = abs((float(a) - float(b)) % math.pi)
    return min(d, math.pi - d)


def carrier_to_tangent(theta: float) -> float:
    return float(theta) % math.pi


def _opens_toward(det, delta_xy, n_theta: int = 12) -> bool:
    dx, dy = float(delta_xy[0]), float(delta_xy[1])
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return False
    q = getattr(getattr(det, "sigma", None), "quantized", ()) or ()
    th_bin = float(q[0]) if len(q) >= 1 else 0.0
    t = carrier_to_tangent((th_bin + 0.5) * math.pi / max(1, int(n_theta)))
    dot = math.cos(t) * (dx / n) + math.sin(t) * (dy / n)
    side = float(q[4]) if len(q) >= 5 else 1.0
    if abs(side - 1.0) < 0.5:
        return abs(dot) > 0.5
    return (dot > 0.25) if side > 1.0 else (dot < -0.25)
