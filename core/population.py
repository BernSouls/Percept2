from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PopAxis:
    name: str
    n_bins: int
    lo: float = 0.0
    hi: float = 1.0
    circular: bool = False
    sigma_bins: float = 0.85


def soft_bump_bins(center_bin: float, axis: PopAxis) -> Tuple[float, ...]:
    n = max(1, int(axis.n_bins))
    if center_bin is None or not math.isfinite(float(center_bin)):
        return tuple(1.0 / n for _ in range(n))
    sig = max(0.25, float(axis.sigma_bins))
    vals = []
    for i in range(n):
        c = i + 0.5
        d = abs(float(center_bin) - c)
        if axis.circular:
            d = min(d, max(0.0, n - d))
        vals.append(math.exp(-0.5 * (d / sig) ** 2))
    s = sum(vals)
    return tuple(v / s for v in vals) if s > 1e-12 else tuple(1.0 / n for _ in range(n))


def soft_bump_from_value(value: Optional[float], axis: PopAxis) -> Tuple[float, ...]:
    n = max(1, int(axis.n_bins))
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return tuple(1.0 / n for _ in range(n))
    span = max(1e-9, float(axis.hi) - float(axis.lo))
    pos = ((float(value) - float(axis.lo)) / span) * n
    if axis.circular:
        pos = pos % n
    else:
        pos = min(float(n), max(0.0, pos))
    return soft_bump_bins(pos, axis)


def mixture_bins(centers: Sequence[float], axis: PopAxis) -> Tuple[float, ...]:
    n = max(1, int(axis.n_bins))
    if not centers:
        return tuple(1.0 / n for _ in range(n))
    acc = [0.0] * n
    k = 0
    for c in centers:
        if c is not None and math.isfinite(float(c)):
            v = soft_bump_bins(float(c), axis)
            for i in range(n):
                acc[i] += v[i]
            k += 1
    if k == 0:
        return tuple(1.0 / n for _ in range(n))
    s = sum(acc)
    return tuple(v / s for v in acc) if s > 1e-12 else tuple(1.0 / n for _ in range(n))


def merge_populations(
    pop_a: Any,
    w_a: Any = 1.0,
    pop_b: Any = None,
    w_new: float = 1.0,
    axes: Optional[Sequence[PopAxis]] = None,
) -> Tuple[Tuple[float, ...], ...]:
    # Real 5-arg call in concept_graph.py: merge_populations(prev_pop, n_prev, pop, 1.0, axes)
    if pop_b is not None:
        if not pop_a:
            return tuple(tuple(float(x) for x in v) for v in pop_b)
        if not pop_b:
            return tuple(tuple(float(x) for x in v) for v in pop_a)
        wa = float(w_a)
        wb = float(w_new)
        n_ax = min(len(pop_a), len(pop_b))
        out = []
        for i in range(n_ax):
            va, vb = pop_a[i], pop_b[i]
            n = min(len(va), len(vb))
            acc = [wa * float(va[b]) + wb * float(vb[b]) for b in range(n)]
            s = sum(acc)
            out.append(tuple(v / s for v in acc) if s > 1e-12 else tuple(1.0 / n for _ in range(n)))
        return tuple(out)
    # Fallback for sequence-of-populations call
    pops = pop_a
    if not pops or not pops[0]:
        return ()
    ref = pops[0]
    ws = list(w_a) if isinstance(w_a, (list, tuple)) else [1.0] * len(pops)
    out = []
    for ax_i in range(len(ref)):
        n = len(ref[ax_i])
        acc = [0.0] * n
        for p, w in zip(pops, ws):
            if ax_i < len(p) and len(p[ax_i]) == n:
                for b in range(n):
                    acc[b] += float(w) * float(p[ax_i][b])
        s = sum(acc)
        out.append(tuple(v / s for v in acc) if s > 1e-12 else tuple(1.0 / n for _ in range(n)))
    return tuple(out)


def population_centre_bins(pop: Tuple[Tuple[float, ...], ...]) -> Tuple[float, ...]:
    out = []
    for vec in pop:
        if not vec:
            out.append(0.0)
        else:
            s = sum(vec)
            out.append(sum((i + 0.5) * v for i, v in enumerate(vec)) / s if s > 1e-12 else 0.0)
    return tuple(out)


def validate_population(pop: Tuple[Tuple[float, ...], ...], axes: Sequence[PopAxis]) -> bool:
    if len(pop) != len(axes):
        return False
    return all(len(v) == ax.n_bins and all(math.isfinite(x) for x in v) for v, ax in zip(pop, axes))
