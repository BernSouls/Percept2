# Deferred Relational Modeling — No New Edge Types

This note supersedes the early audit's proposals for adding EdgeType categories. The active taxonomy is fixed at exactly eight spatial categories; deferred work must refine emitters, measurements, or gates inside those categories, not add temporal or spatial EdgeTypes.

| Deferred question | Current category home | What remains unmodeled |
|---|---|---|
| Smooth surface continuation versus a crease | `SURFACE` | The current schema measures normal angle, material difference, and slant difference. Explicit convex/concave dihedral sign and broader coplanarity evidence would need validated measurements in the existing `SURFACE` schema. |
| Figure/ground or phase opposition across time | Signed `MATERIAL` | Anti-phase routing is implemented, but `P.ANTIPHASE_ENABLED` remains off by default; broader temporal validation is still needed. Do not restore a separate temporal opposition edge. |
| Coherent motion | `MATERIAL` | Co-motion is folded into `MATERIAL`; test it against richer multi-object and longer-duration sequences before extending its measurements. |
| Illumination boundary versus intrinsic material change | `MATERIAL` | The current appearance axes do not explicitly separate reflectance from illumination. Any refinement should preserve the canonical `MATERIAL` category. |
| Physical support, contact, and part structure | Existing `CONTAINMENT`, `PROXIMITY`, `SURFACE`, and `OCCLUSION` | More scenes and calibrated measurements are needed to distinguish support/contact from mere adjacency. Do not create `SUPPORT`, `CONTACT`, or `PART` EdgeTypes without revising the eight-category requirement. |
| Rich-surface grouping selectivity | `SURFACE` | In the current static two-step trace the `SURFACE` graph grows by 1,015 edges on the second step. Audit duplicate/hysteretic grouping and candidate gates without reintroducing temporal edge categories. |

## Preserved invariants

- `CONTAINMENT` remains directed container → member.
- `OCCLUSION` remains directed near → far with positive sign only.
- Co-motion and signed anti-phase evidence use `MATERIAL`.
- `PROXIMITY`, `CONTINUITY`, `JUNCTION`, `SURFACE`, `SHAPE`, and `MATERIAL` remain undirected.
- Population axes stay physical and category-specific; confidence is endpoint-minimum, separate from geometric signal.

See [`edge_classification.md`](edge_classification.md) for the current axis schemas and [`left_for_user.md`](left_for_user.md) for the concrete follow-up and test-environment limitations.
