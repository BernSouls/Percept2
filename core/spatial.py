from collections import defaultdict
import math
from typing import Any, Dict, List, Optional, Tuple


def _pos_dist(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    n = min(len(a), len(b))
    return math.sqrt(sum((float(a[k]) - float(b[k])) ** 2 for k in range(n)))


class SpatialHash:
    def __init__(
        self,
        positions: Optional[Dict[Any, Tuple[float, ...]]] = None,
        cell_size: float = 16.0,
    ) -> None:
        if isinstance(positions, (int, float)):
            cell_size = float(positions)
            positions = None
        self.cell_size = max(float(cell_size), 1e-6)
        self._cells: Dict[Tuple[int, ...], List[Tuple[Any, Tuple[float, ...]]]] = defaultdict(list)
        if isinstance(positions, dict):
            for anchor, pos in positions.items():
                if pos is not None and len(pos) >= 2:
                    self.insert(anchor, pos)

    def _key(self, pos: Tuple[float, ...]) -> Tuple[int, ...]:
        return tuple(int(math.floor(float(pos[k]) / self.cell_size)) for k in range(min(2, len(pos))))

    def insert(self, item: Any, pos: Tuple[float, ...]) -> None:
        self._cells[self._key(pos)].append((item, pos))

    def neighbors(self, pos: Tuple[float, ...], radius: float) -> List[Tuple[Any, Tuple[float, ...]]]:
        r = max(0.0, float(radius))
        span = int(math.ceil(r / self.cell_size))
        c = self._key(pos)
        out: List[Tuple[Any, Tuple[float, ...]]] = []
        if len(c) == 2:
            for dx in range(-span, span + 1):
                for dy in range(-span, span + 1):
                    for item, ipos in self._cells.get((c[0] + dx, c[1] + dy), ()):
                        if _pos_dist(pos, ipos) <= r:
                            out.append((item, ipos))
        return out

    def query_radius(self, pos: Tuple[float, ...], radius: float) -> List[Any]:
        return [item for item, _ in self.neighbors(pos, radius)]
