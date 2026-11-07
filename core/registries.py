from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from core.population import PopAxis, soft_bump_from_value, soft_bump_bins
from core.quantize import q_linear

ABSENT = -1.0
ACTION_NOOP = "NOOP"
ACTION_NULL = "NULL"
GABOR_NT = 12
GABOR_NSF = 5
CONTOUR_PEAK_STRENGTH_Z_SCALE = 1.0
SELF_BINDER_TYPES = frozenset()


@dataclass(frozen=True)
class SlotSpec:
    name: str
    n_bins: int
    lo: float = 0.0
    hi: float = 1.0
    circular: bool = False
    weight: float = 1.0


def _S(name, n_bins=8, lo=0.0, hi=1.0, circular=False):
    return SlotSpec(name=name, n_bins=n_bins, lo=lo, hi=hi, circular=circular)


def b1_slot_layout(nt: int = GABOR_NT, nsf: int = GABOR_NSF) -> List[Tuple[str, float, float, float, str]]:
    cells = nt * nsf
    twopi = 2.0 * math.pi
    out: List[Tuple[str, float, float, float, str]] = [
        ("E_peak", 0.0, 1.0, 0.0, "base"),
        ("energy_conc", 0.0, 1.0, 0.0, "base"),
        ("sf_peak_ratio", 0.0, 2.0, 0.0, "base"),
        ("sf_curvature", -1.0, 1.0, 0.0, "sf_curv"),
        ("is_2d_peak", 0.0, 1.0, 0.0, "base"),
        ("simple_flag", 0.0, 1.0, 0.0, "base"),
        ("on_simple_flag", 0.0, 1.0, 0.0, "base"),
        ("off_simple_flag", 0.0, 1.0, 0.0, "base"),
        ("dtheta", -0.5, 0.5, 0.0, "base"),
        ("dlogsf", -0.5, 0.5, 0.0, "base"),
        ("region_lum_mean", 0.0, 1.0, 0.0, "base"),
        ("region_lum_rms", 0.0, 1.0, 0.0, "base"),
        ("surround_lum_mean", 0.0, 1.0, 0.0, "base"),
        ("lum_mean_diff", -1.0, 1.0, 0.0, "base"),
        ("boundary_interior_ratio", 0.0, 1.0, 0.0, "base"),
        ("cross_scale_phase", -math.pi, math.pi, twopi, "base"),
        ("on_dtheta", -0.5, 0.5, 0.0, "onoff_subbin"),
        ("on_dlogsf", -0.5, 0.5, 0.0, "onoff_subbin"),
        ("off_dtheta", -0.5, 0.5, 0.0, "onoff_subbin"),
        ("off_dlogsf", -0.5, 0.5, 0.0, "onoff_subbin"),
    ]
    for k in range(cells):
        out.append((f"hist_{k}", 0.0, 1.0, 0.0, "hist"))
    for k in range(cells):
        out.append((f"on_hist_{k}", 0.0, 1.0, 0.0, "hist"))
    for k in range(cells):
        out.append((f"off_hist_{k}", 0.0, 1.0, 0.0, "hist"))
    phase_slots = [
        ("phase_R", 0.0, 1.0, 0.0), ("phase_skew", -math.pi, math.pi, twopi),
        ("phase_diff_theta_plus", -math.pi, math.pi, twopi), ("phase_R_theta_plus", 0.0, 1.0, 0.0),
        ("phase_diff_theta_minus", -math.pi, math.pi, twopi), ("phase_R_theta_minus", 0.0, 1.0, 0.0),
        ("phase_diff_theta_plus2", -math.pi, math.pi, twopi), ("phase_R_theta_plus2", 0.0, 1.0, 0.0),
        ("phase_diff_theta_minus2", -math.pi, math.pi, twopi), ("phase_R_theta_minus2", 0.0, 1.0, 0.0),
        ("phase_diff_sf_plus", -math.pi, math.pi, twopi), ("phase_R_sf_plus", 0.0, 1.0, 0.0),
        ("phase_diff_sf_minus", -math.pi, math.pi, twopi), ("phase_R_sf_minus", 0.0, 1.0, 0.0),
        ("phase_diff_sf_plus2", -math.pi, math.pi, twopi), ("phase_R_sf_plus2", 0.0, 1.0, 0.0),
        ("phase_diff_sf_minus2", -math.pi, math.pi, twopi), ("phase_R_sf_minus2", 0.0, 1.0, 0.0),
        ("phase_diff_tp_sp", -math.pi, math.pi, twopi), ("phase_diff_tp_sm", -math.pi, math.pi, twopi),
        ("phase_diff_tm_sp", -math.pi, math.pi, twopi), ("phase_diff_tm_sm", -math.pi, math.pi, twopi),
        ("phase_grad_theta", -math.pi, math.pi, twopi), ("phase_grad_sf", -math.pi, math.pi, twopi),
        ("on_phase_R", 0.0, 1.0, 0.0), ("on_phase_skew", -math.pi, math.pi, twopi),
        ("off_phase_R", 0.0, 1.0, 0.0), ("off_phase_skew", -math.pi, math.pi, twopi),
        ("on_phase_diff_theta_plus", -math.pi, math.pi, twopi),
        ("on_phase_diff_theta_minus", -math.pi, math.pi, twopi),
        ("off_phase_diff_theta_plus", -math.pi, math.pi, twopi),
        ("off_phase_diff_theta_minus", -math.pi, math.pi, twopi),
        ("on_off_phase_diff", -math.pi, math.pi, twopi),
    ]
    for nm, lo, hi, per in phase_slots:
        out.append((nm, lo, hi, per, "phase"))
    for q_idx in range(4):
        out.append((f"sp_nonempty_{q_idx}", 0.0, 1.0, 0.0, "sp_nonhist"))
        out.append((f"sp_theta_rel_{q_idx}", -math.pi / 2.0, math.pi / 2.0, math.pi, "sp_nonhist"))
        out.append((f"sp_R_{q_idx}", 0.0, 1.0, 0.0, "sp_nonhist"))
        out.append((f"sp_dtheta_{q_idx}", -0.5, 0.5, 0.0, "sp_nonhist"))
        out.append((f"sp_dlogsf_{q_idx}", -0.5, 0.5, 0.0, "sp_nonhist"))
        for k in range(nt * 3):
            out.append((f"sp_hist_{q_idx}_{k}", 0.0, 1.0, 0.0, "sp_hist"))
    for nm in ("01", "23", "02", "13"):
        out.append((f"sp_phase_diff_{nm}", -math.pi, math.pi, twopi, "q_phase_diff"))
    for nm in ("01", "23"):
        out.append((f"sp_theta_diff_{nm}", -math.pi / 2.0, math.pi / 2.0, math.pi, "q_theta_diff"))
    out.extend([
        ("surround_theta", 0.0, math.pi, math.pi, "surround"),
        ("surround_theta_conc", 0.0, 1.0, 0.0, "surround"),
        ("surround_logsf_mean", -4.0, 0.0, 0.0, "surround"),
        ("surround_logsf_std", 0.0, 2.0, 0.0, "surround"),
        ("surround_on_mean", 0.0, 1.0, 0.0, "surround"),
        ("surround_off_mean", 0.0, 1.0, 0.0, "surround"),
    ])
    for k in range(nt * 2):
        out.append((f"surround_hist_{k}", 0.0, 1.0, 0.0, "surround"))
    return out


def b1_class_name(nt: int = GABOR_NT, nsf: int = GABOR_NSF) -> str:
    return f"ORIENTATION_RICH_{nt}_{nsf}"


def b4_class_name(nt: int = GABOR_NT, nsf: int = GABOR_NSF) -> str:
    return f"TEXTURE_RICH_{nt}_{nsf}"


_B4_SLOTS_12_5 = tuple(_S(f"cell_{i}", 8, 0.0, 1.0) for i in range(4 * GABOR_NT * GABOR_NSF)) + (
    _S("orient_conc", 8, 0.0, 1.0),
    _S("sf_centroid", 8, 0.0, 1.0),
    _S("roughness", 8, 0.0, 1.0),
    _S("contrast_energy", 8, 0.0, 1.0),
)

_B1_SLOTS_12_5 = tuple(
    _S(nm, 8, lo, hi, per > 0.0)
    for (nm, lo, hi, per, _blk) in b1_slot_layout(GABOR_NT, GABOR_NSF)
)

POPULATION_SLOTS: Dict[str, Tuple[SlotSpec, ...]] = {
    "BOUNDARY": (_S("circ", 11, 0.0, 1.0), _S("comp", 11, 0.0, 1.0)),
    "BOUNDARY_ANCHORED": (_S("circ", 11, 0.0, 1.0), _S("comp", 11, 0.0, 1.0), _S("anchor", 16, 0.0, 15.0)),
    "COLOR": (
        _S("lum", 8, -1.0, 1.0),
        _S("rg", 16, -1.0, 1.0),
        _S("by", 12, -1.0, 1.0),
        _S("sat", 8, 0.0, 1.0),
        _S("hue", 16, 0.0, 2 * math.pi, True),
    ),
    "COLOR_LEGACY": (_S("r", 12, 0.0, 1.0), _S("g", 12, 0.0, 1.0), _S("b", 12, 0.0, 1.0)),
    "TEXTURE": (_S("roughness", 11, 0.0, 1.0), _S("isotropy", 10, 0.0, 1.0)),
    "POSITION": (_S("x", 16, -1.0, 1.0), _S("y", 16, -1.0, 1.0), _S("z", 16, -1.0, 1.0)),
    "ONSET": (_S("age", 8, 0.0, 1.0), _S("sig", 8, 0.0, 1.0)),
    "MOTION": (_S("dir", 8, 0.0, 2 * math.pi, True), _S("speed", 8, 0.0, 1.0)),
    "SIZE": (_S("radius", 10, 0.0, 1.0),),
    "POLARITY": (_S("polarity", 4, -1.0, 1.0),),
    "ORIENTATION": (
        _S("theta", 12, 0.0, math.pi, True),
        _S("elongation", 10, 0.0, 1.0),
        _S("anchor", 16, 0.0, 15.0),
    ),
    "ORIENTATION_POL": (
        _S("theta", 12, 0.0, math.pi, True),
        _S("sf", 5, 0.0, 4.0),
        _S("polarity", 3, -1.0, 1.0),
        _S("phase", 8, -math.pi, math.pi, True),
        _S("energy", 8, 0.0, 1.0),
    ),
    "ORIENTATION_SEGMENT": (
        _S("theta", 24, 0.0, math.pi, True),
        _S("arc", 8, 0.0, 1.0),
        _S("pos_x", 8, 0.0, 1.0),
        _S("pos_y", 8, 0.0, 1.0),
    ),
    "CONTOUR_FRAGMENT": (
        _S("kappa", 8, -1.0, 1.0),
        _S("theta", 12, 0.0, 2 * math.pi, True),
        _S("r", 8, 0.0, 1.0),
        _S("tvr", 8, 0.0, 1.0),
    ),
    "CONTOUR_PEAK": (
        _S("theta", 12, 0.0, math.pi, True),
        _S("sf", 5, 0.0, 4.0),
        _S("curvature", 8, -1.0, 1.0),
        _S("bend", 8, 0.0, math.pi),
        _S("phase", 4, -math.pi, math.pi, True),
        _S("strength", 8, 0.0, 1.0),
        _S("coherence", 8, 0.0, 1.0),
        _S("endstop", 4, 0.0, 1.0),
    ),
    "PEAK_CHAIN": (
        _S("count", 16, 0.0, 16.0),
        _S("theta_dist", 12, 0.0, math.pi, True),
        _S("spread", 8, 0.0, 1.0),
        _S("closure", 2, 0.0, 1.0),
        _S("curvature", 8, 0.0, 1.0),
        _S("length", 8, 0.0, 1.0),
    ),
    "CORNER": (
        _S("angle", 8, 0.0, math.pi),
        _S("bisector", 12, 0.0, 2 * math.pi, True),
        _S("prominence", 8, 0.0, 1.0),
        _S("arm_frac", 8, 0.0, 1.0),
    ),
    "ENDSTOPPED": (
        _S("theta", 12, 0.0, math.pi, True),
        _S("sf", 5, 0.0, 1.0),
        _S("rel_single", 8, 0.0, 1.0),
        _S("rel_double", 8, 0.0, 1.0),
        _S("side", 3, -1.0, 1.0),
    ),
    "BORDER_OWNERSHIP": (
        _S("arc", 8, 0.0, 1.0),
        _S("signed_dir", 8, -1.0, 1.0),
        _S("curvature", 8, -1.0, 1.0),
        _S("strength", 8, 0.0, 4.0),
    ),
    "CONTOUR": (
        _S("arc_len", 8, 0.0, 32.0),
        _S("mean_theta", 24, 0.0, math.pi, True),
        _S("circ_var", 8, 0.0, 1.0),
        _S("closure", 2, 0.0, 1.0),
        _S("total_turn", 8, 0.0, 4.0),
        _S("corner_count", 8, 0.0, 8.0),
        _S("chirality", 8, -4.0, 4.0),
    ) + tuple(_S(f"curv{k}", 9, -1.0, 1.0) for k in range(20)),
    "JUNCTION": (
        _S("degree", 6, 2.0, 7.0),
        _S("ref_theta", 24, 0.0, 2 * math.pi, True),
        _S("topo", 7, 0.0, 6.0),
        _S("spread", 8, 0.0, 1.0),
        _S("gap0", 12, 0.0, 2 * math.pi),
        _S("gap1", 12, 0.0, 2 * math.pi),
        _S("gap2", 12, 0.0, 2 * math.pi),
        _S("gap3", 12, 0.0, 2 * math.pi),
        _S("arm_support", 8, 0.0, 8.0),
        _S("arm_sharp_mean", 8, 0.0, 1.0),
        _S("arm_sharp_min", 8, 0.0, 1.0),
    ) + tuple(_S(f"arm_angle{i}", 8, 0.0, 6.0) for i in range(8)),
    "TEXTURE_BOUNDARY": (
        _S("arc_pos", 8, 0.0, 1.0),
        _S("dtheta", 12, 0.0, math.pi, True),
        _S("dlogsf", 8, 0.0, 4.0),
        _S("d_contrast", 8, 0.0, 1.0),
        _S("d_ori", 8, 0.0, 1.0),
        _S("lum_dist", 8, 0.0, 1.0),
        _S("conc_a", 8, 0.0, 1.0),
        _S("conc_b", 8, 0.0, 1.0),
    ),
    "SURFACE_PATCH": (
        _S("shape", 8, 0.0, 1.0),
        _S("lum", 8, 0.0, 1.0),
        _S("hue", 16, 0.0, 2 * math.pi, True),
        _S("sat", 8, 0.0, 1.0),
        _S("tex", 8, 0.0, 1.0),
        _S("grad_mag", 8, 0.0, 4.0),
        _S("grad_side", 8, -1.0, 1.0),
    ),
    "SURFACE_FRAGMENT": (
        _S("slant", 8, 0.0, math.pi / 2),
        _S("tilt", 12, 0.0, 2 * math.pi, True),
        _S("curv_sign", 3, -1.0, 1.0),
        _S("confidence", 8, 0.0, 1.0),
    ),
    "ILLUSORY_CONTOUR": (
        _S("theta", 12, 0.0, math.pi, True),
        _S("length", 8, 0.0, 1.0),
        _S("strength", 8, 0.0, 1.0),
    ),
    "CURVE_COMPLEX": (
        _S("n_contours", 8, 0.0, 8.0),
        _S("n_junctions", 8, 0.0, 8.0),
        _S("extent", 8, 0.0, 1.0),
        _S("closure", 4, 0.0, 1.0),
    ) + tuple(_S(f"template{i}", 8, -1.0, 1.0) for i in range(8)) + tuple(_S(f"shape{i}", 8, 0.0, 1.0) for i in range(12)),
    "REGION_SHAPE": tuple(_S(f"hist{i}", 8, 0.0, 1.0) for i in range(16)) + (
        _S("area", 8, 0.0, 1.0),
        _S("elong", 8, 0.0, 1.0),
        _S("sym", 8, 0.0, 1.0),
        _S("axis", 12, 0.0, math.pi, True),
        _S("fold", 4, 0.0, 3.0),
    ),
    "TEXTURE_RICH_12_5": _B4_SLOTS_12_5,
    "ORIENTATION_RICH_12_5": _B1_SLOTS_12_5,
}

POPULATION_AXES: Dict[str, Tuple[PopAxis, ...]] = {
    cls: tuple(PopAxis(s.name, s.n_bins, s.lo, s.hi, s.circular) for s in specs)
    for cls, specs in POPULATION_SLOTS.items()
}
POPULATION_AXES["PEAK_CHAIN"] = (
    PopAxis("theta_dist", 12, 0.0, math.pi, True),
    PopAxis("curvature", 8, 0.0, 1.0, False),
    PopAxis("length", 8, 0.0, 1.0, False),
)

CONSTITUTIONAL_RANGES: Dict[str, Any] = {
    cls: tuple(float(s.n_bins - 1) for s in specs)
    for cls, specs in POPULATION_SLOTS.items()
}
CONSTITUTIONAL_RANGES["COLOR_LEGACY"] = (12.0, 12.0, 12.0)
CONSTITUTIONAL_RANGES["BOUNDARY"] = (10.0, 10.0)
CONSTITUTIONAL_RANGES["BOUNDARY_ANCHORED"] = (10.0, 10.0, 0.0)

CIRCULAR_CYCLE_LENGTHS: Dict[str, Dict[int, float]] = {
    cls: {i: float(s.n_bins) for i, s in enumerate(specs) if s.circular}
    for cls, specs in POPULATION_SLOTS.items()
    if any(s.circular for s in specs)
}

SLOT_WEIGHT_PRIORS: Dict[str, Tuple[float, ...]] = {
    cls: tuple(1.0 / max(1, len(specs)) for _ in specs)
    for cls, specs in POPULATION_SLOTS.items()
}


def _verify_circular_ranges(ranges_dict=None, cycles_dict=None, radian_set=None):
    return True


def slot_specs(class_: str):
    return POPULATION_SLOTS.get(class_, ())


def slot_spec(class_: str, name: str):
    for s in POPULATION_SLOTS.get(class_, ()):
        if s.name == name:
            return s
    return None


def get_population_axes(class_: str):
    return POPULATION_AXES.get(class_, ())


def population_axis(class_: str, name: str):
    for ax in POPULATION_AXES.get(class_, ()):
        if ax.name == name:
            return ax
    return None


def slot_axes(class_: str):
    specs = POPULATION_SLOTS.get(class_, ())
    axes = POPULATION_AXES.get(class_, ())
    return tuple(zip(specs, axes)) if len(specs) == len(axes) else ()


def quantise(spec: SlotSpec, value: float) -> int:
    if spec.circular:
        span = max(1e-9, spec.hi - spec.lo)
        pos = ((float(value) - spec.lo) % span) / span
        return int(math.floor(pos * spec.n_bins)) % spec.n_bins
    return q_linear(value, spec.n_bins, spec.lo, spec.hi)


def population_for_slot_value(class_: str, slot_name: str, value: Optional[float]):
    ax = population_axis(class_, slot_name)
    return soft_bump_from_value(value, ax) if ax is not None else ()


def population_from_values(class_: str, values: Sequence[Optional[float]]):
    axes = POPULATION_AXES.get(class_, ())
    if not axes or len(values) != len(axes):
        return ()
    return tuple(soft_bump_from_value(v, ax) for v, ax in zip(values, axes))


def population_values_in_span(class_: str, values: Sequence[Optional[float]]):
    axes = POPULATION_AXES.get(class_, ())
    if not axes or len(values) != len(axes):
        return tuple(values)
    out = []
    for v, ax in zip(values, axes):
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            out.append(None)
        elif ax.circular:
            span = max(1e-9, ax.hi - ax.lo)
            out.append(ax.lo + ((float(v) - ax.lo) % span))
        else:
            out.append(min(ax.hi, max(ax.lo, float(v))))
    return tuple(out)


def population_for_bins(class_: str, bins: Sequence[float]):
    axes = POPULATION_AXES.get(class_, ())
    if not axes or len(bins) != len(axes):
        return ()
    return tuple(soft_bump_bins(float(b) + 0.5, ax) for b, ax in zip(bins, axes))


def population_from_named_vectors(class_: str, vecs: Dict[str, Tuple[float, ...]]):
    axes = POPULATION_AXES.get(class_, ())
    if not axes:
        return ()
    out = []
    for ax in axes:
        v = vecs.get(ax.name)
        if not v or len(v) != ax.n_bins:
            v = tuple(1.0 / ax.n_bins for _ in range(ax.n_bins))
        out.append(tuple(float(x) for x in v))
    return tuple(out)


def population_slot_vector(pop, class_: str, slot_name: str):
    axes = POPULATION_AXES.get(class_, ())
    for i, ax in enumerate(axes):
        if ax.name == slot_name and i < len(pop):
            return pop[i]
    return ()


def is_action_lens_eligible(k: str) -> bool:
    return False


def is_noop_action(a: Any) -> bool:
    return a in (None, ACTION_NOOP, ACTION_NULL)


def is_probe_action(a: Any) -> bool:
    return False


def is_teleological_action(a: Any) -> bool:
    return False
