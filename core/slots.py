import math
from typing import Any, Sequence, Tuple
from core.registries import POPULATION_SLOTS


def slot_names(class_: str) -> Tuple[str, ...]:
    return tuple(s.name for s in POPULATION_SLOTS.get(class_, ()))


def slot_slice(sigma: Any, class_: str, names: Sequence[str] = ()) -> Tuple[float, ...]:
    q = getattr(sigma, "quantized", ()) or ()
    specs = POPULATION_SLOTS.get(class_, ())
    idx_map = {s.name: i for i, s in enumerate(specs)}
    out = []
    for nm in names:
        i = idx_map.get(nm)
        if i is not None and i < len(q):
            out.append(float(q[i]))
        else:
            out.append(0.0)
    return tuple(out)


def slot_value(sigma: Any, class_: str, name: str, default: Any = None) -> Any:
    q = getattr(sigma, "quantized", ()) or ()
    specs = POPULATION_SLOTS.get(class_, ())
    for i, s in enumerate(specs):
        if s.name == name and i < len(q):
            return float(q[i])
    return default


def slot_values(sigma: Any, class_: str, names: Sequence[str] = ()) -> Tuple[float, ...]:
    if names:
        return slot_slice(sigma, class_, names)
    return tuple(float(x) for x in (getattr(sigma, "quantized", ()) or ()))


def orientation_theta(sigma: Any) -> float:
    cls = getattr(sigma, "class_", "")
    v = slot_value(sigma, cls, "theta", default=None)
    if v is None:
        v = slot_value(sigma, cls, "mean_theta", default=0.0)
    return float(v)


def orientation_theta_radians(sigma: Any) -> float:
    cls = getattr(sigma, "class_", "")
    specs = POPULATION_SLOTS.get(cls, ())
    q = getattr(sigma, "quantized", ()) or ()
    for i, s in enumerate(specs):
        if s.name in ("theta", "mean_theta") and i < len(q):
            return ((float(q[i]) + 0.5) / max(1, s.n_bins)) * (s.hi - s.lo) + s.lo
    if q:
        return ((float(q[0]) + 0.5) / 12.0) * math.pi
    return 0.0
