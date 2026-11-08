from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from core.population import PopAxis, soft_bump_from_value
from core.types import EdgeType


# Physical population-coding schemas for the 8 spatial EdgeType categories.
# Every bound derives from geometry or sensory scale:
#   * PROXIMITY.distance: log1p(dist / extraction_scale) in [0, 4] (0 at dist=0,
#     covers up to e^4 - 1 ~= 53.6 extraction scales without log(0) singularity).
#   * PROXIMITY.bearing: axial orientation in [0, pi) (circular=True; None when
#     dist == 0 so co-located pairs emit a truthful uniform distribution).
#   * CONTAINMENT.polar_angle: full-circle bearing in [0, 2*pi) (circular=True).
#   * OCCLUSION: sign=+1 is directed near -> far ordinal depth; sign=-1 is
#     repulsive figure-ground segregation (source_cue=3), not material evidence.
EDGE_SCHEMA: Dict[EdgeType, Tuple[PopAxis, ...]] = {
    EdgeType.PROXIMITY: (
        PopAxis("distance", 8, 0.0, 4.0, circular=False),
        PopAxis("bearing", 12, 0.0, math.pi, circular=True),
        PopAxis("scale_ratio", 8, -2.0, 2.0, circular=False),
    ),
    EdgeType.CONTINUITY: (
        PopAxis("gap", 8, 0.0, 4.0, circular=False),
        PopAxis("turn_angle", 12, -math.pi / 2.0, math.pi / 2.0, circular=False),
        PopAxis("perp_offset", 8, 0.0, 3.0, circular=False),
        PopAxis("curv_agree", 8, -1.0, 1.0, circular=False),
    ),
    EdgeType.CONTAINMENT: (
        PopAxis("rel_dist", 8, 0.0, 2.0, circular=False),
        PopAxis("polar_angle", 12, 0.0, 2.0 * math.pi, circular=True),
        PopAxis("scale_ratio", 8, -3.0, 1.0, circular=False),
    ),
    EdgeType.OCCLUSION: (
        PopAxis("depth_diff", 8, 0.0, 4.0, circular=False),
        PopAxis("source_cue", 4, 0.0, 3.0, circular=False),
        PopAxis("strength", 8, 0.0, 1.0, circular=False),
    ),
    EdgeType.JUNCTION: (
        PopAxis("angle_delta", 12, 0.0, math.pi / 2.0, circular=False),
        PopAxis("topo_class", 7, 0.0, 6.0, circular=False),
        PopAxis("arm_count", 6, 2.0, 7.0, circular=False),
    ),
    EdgeType.SURFACE: (
        PopAxis("normal_angle", 12, 0.0, math.pi, circular=False),
        PopAxis("material_diff", 8, 0.0, 1.0, circular=False),
        PopAxis("slant_diff", 8, 0.0, math.pi / 2.0, circular=False),
    ),
    EdgeType.SHAPE: (
        PopAxis("sym_score", 8, 0.0, 1.0, circular=False),
        PopAxis("radial_ratio", 8, 0.0, 3.0, circular=False),
        PopAxis("curvature", 8, -1.0, 1.0, circular=False),
    ),
    EdgeType.MATERIAL: (
        PopAxis("chrom_diff", 8, 0.0, 1.0, circular=False),
        PopAxis("lum_diff", 8, 0.0, 1.0, circular=False),
        PopAxis("tex_diff", 8, 0.0, 1.0, circular=False),
    ),
}

EDGE_POPULATION_AXES: Dict[Union[EdgeType, str], Tuple[PopAxis, ...]] = {}
EDGE_UNIT_TABLE: Dict[Union[EdgeType, str], Tuple[PopAxis, ...]] = {}
for _et, _axes in EDGE_SCHEMA.items():
    EDGE_POPULATION_AXES[_et] = _axes
    EDGE_UNIT_TABLE[_et] = _axes
    EDGE_POPULATION_AXES[_et.name] = _axes
    EDGE_POPULATION_AXES[f"EDGE_{_et.name}"] = _axes
    EDGE_UNIT_TABLE[_et.name] = _axes
    EDGE_UNIT_TABLE[f"EDGE_{_et.name}"] = _axes


@dataclass(frozen=True)
class EdgeParams:
    schema: str
    bins: Tuple[int, ...]
    population: Tuple[Tuple[float, ...], ...] = ()
    sign: int = 1
    confidence: float = 1.0
    # Original named measurements are retained only as provenance so the
    # binder can canonicalise legacy emitter payloads into the 8 physical
    # edge schemas without guessing from already-quantised bins.
    measurements: Tuple[Tuple[str, Optional[float]], ...] = field(
        default=(), compare=False, repr=False
    )


@dataclass
class EdgeParamsMixture:
    schema: str
    protos: List[EdgeParams] = field(default_factory=list)
    mass: List[float] = field(default_factory=list)
    last_t: int = 0

    @classmethod
    def from_point(cls, ep: EdgeParams, t: int) -> "EdgeParamsMixture":
        return cls(schema=ep.schema, protos=[ep], mass=[1.0], last_t=int(t))

    def observe(self, ep: EdgeParams, t: int, weight: float = 1.0) -> None:
        w = max(0.0, float(weight))
        self.last_t = int(t)
        for idx, p in enumerate(self.protos):
            if p.bins == ep.bins and p.sign == ep.sign:
                self.mass[idx] += w
                return
        self.protos.append(ep)
        self.mass.append(w)


def _resolve_schema_axes(
    schema: Union[EdgeType, str], values: Dict[str, Any]
) -> Tuple[str, Tuple[PopAxis, ...]]:
    if isinstance(schema, EdgeType):
        return schema.name, EDGE_SCHEMA[schema]
    name = str(schema)
    if name in EDGE_POPULATION_AXES:
        canon = name[5:] if name.startswith("EDGE_") and name[5:] in EDGE_POPULATION_AXES else name
        return canon, EDGE_POPULATION_AXES[name]
    axes = []
    for k in sorted(values.keys()):
        lo, hi, circ = 0.0, 1.0, False
        if "angle" in k or "theta" in k or "bearing" in k:
            lo, hi, circ = 0.0, math.pi, True
        axes.append(PopAxis(k, 8, lo, hi, circ))
    t_axes = tuple(axes)
    EDGE_POPULATION_AXES[name] = t_axes
    EDGE_UNIT_TABLE[name] = t_axes
    return name, t_axes


def edge_params(
    schema: Union[EdgeType, str],
    values: Dict[str, Any],
    sign: int = 1,
    confidence: float = 1.0,
) -> EdgeParams:
    canon_name, axes = _resolve_schema_axes(schema, values)
    bins: List[int] = []
    pop: List[Tuple[float, ...]] = []
    for ax in axes:
        v = values.get(ax.name, None)
        pop.append(soft_bump_from_value(v, ax))
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            bins.append(0)
        else:
            span = max(1e-9, float(ax.hi) - float(ax.lo))
            fv = float(v)
            if ax.circular:
                frac = ((fv - float(ax.lo)) / span) % 1.0
            else:
                frac = max(0.0, min(0.999999, (fv - float(ax.lo)) / span))
            bins.append(int(math.floor(frac * ax.n_bins)))
    conf = float(confidence)
    if not math.isfinite(conf):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return EdgeParams(
        schema=canon_name,
        bins=tuple(bins),
        population=tuple(pop),
        sign=int(sign),
        confidence=conf,
        measurements=tuple(
            (k, float(values[k]) if isinstance(values[k], (int, float))
             and math.isfinite(float(values[k])) else None)
            for k in sorted(values)
        ),
    )
