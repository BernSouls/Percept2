from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple


class EdgeType(Enum):
    PROXIMITY = "PROXIMITY"
    CONTINUITY = "CONTINUITY"
    CONTAINMENT = "CONTAINMENT"
    OCCLUSION = "OCCLUSION"
    JUNCTION = "JUNCTION"
    SURFACE = "SURFACE"
    SHAPE = "SHAPE"
    MATERIAL = "MATERIAL"


UNDIRECTED_EDGE_TYPES = frozenset({
    EdgeType.PROXIMITY,
    EdgeType.CONTINUITY,
    EdgeType.JUNCTION,
    EdgeType.SURFACE,
    EdgeType.SHAPE,
    EdgeType.MATERIAL,
})
REPRODUCIBLE_EDGE_TYPES = frozenset(EdgeType)
HEBBIAN_BOOST_TYPES: frozenset = frozenset()


@dataclass(frozen=True)
class TypedSignature:
    kind: str
    class_: str
    relation: str
    quantized: Tuple[float, ...] = ()
    population: Tuple[Tuple[float, ...], ...] = ()

    @property
    def hash_value(self) -> int:
        from core.hashing import _node_signature_hash, hash_64
        return hash_64(_node_signature_hash(self))

    def similarity(self, other: "TypedSignature") -> float:
        import math
        from core.registries import (
            CIRCULAR_CYCLE_LENGTHS,
            CONSTITUTIONAL_RANGES,
            SLOT_WEIGHT_PRIORS,
        )
        if other is None or self.class_ != other.class_:
            return 0.0
        qa = self.quantized or ()
        qb = other.quantized or ()
        if not qa or len(qa) != len(qb):
            return 0.0
        ranges = CONSTITUTIONAL_RANGES.get(self.class_)
        cycles = CIRCULAR_CYCLE_LENGTHS.get(self.class_, {})
        weights = SLOT_WEIGHT_PRIORS.get(self.class_)
        num = 0.0
        den = 0.0
        for i, (xa, xb) in enumerate(zip(qa, qb)):
            w = float(weights[i]) if weights and i < len(weights) else 1.0
            r = float(ranges[i]) if ranges and i < len(ranges) else 7.0
            d = abs(float(xa) - float(xb))
            if i in cycles and cycles[i] > 0.0:
                p = float(cycles[i])
                d = d % p
                d = min(d, p - d)
            norm = d / max(r, 1.0)
            num += w * (norm * norm)
            den += w
        if den <= 1e-12:
            return 0.0
        return float(math.exp(-4.5 * (num / den)))


@dataclass(frozen=True)
class Detection:
    k: str
    source_layer: int
    scale: int
    sigma: TypedSignature
    alpha: Any
    z: float
    t: int
    rho: Tuple[Any, ...] = ()
    substrate_hash: int = 0
