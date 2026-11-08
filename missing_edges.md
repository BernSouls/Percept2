# Deferred Relational Modeling — No New Edge Types

This note supersedes the early audit's proposals for adding EdgeType categories. The active taxonomy is fixed at exactly eight spatial categories; deferred work must refine emitters, measurements, or gates inside those categories, not add temporal or spatial EdgeTypes.

| Deferred question | Current category home | What remains unmodeled |
|---|---|---|
| Smooth surface continuation versus a crease | `SURFACE` | The current schema measures normal angle, material difference, and slant difference. Explicit convex/concave dihedral sign and broader coplanarity evidence would need validated measurements in the existing `SURFACE` schema. |
| Figure/ground or phase opposition across time | Negative-sign `OCCLUSION` (`source_cue=3`) | Anti-phase routing remains behind `P.ANTIPHASE_ENABLED` (off by default); preserve the negative sign and validate with controlled sequences. Do not route it to `MATERIAL` or add a temporal edge category. |
| Coherent motion | Gated off | No video-backed input or approved kinematic representation is available. Do not encode co-motion in `MATERIAL`; only enable it after those prerequisites and an explicit schema decision. |
| Illumination boundary versus intrinsic material change | `MATERIAL` | The current appearance axes do not explicitly separate reflectance from illumination. Any refinement should preserve the canonical `MATERIAL` category. |
| Physical support, contact, and part structure | Existing `CONTAINMENT`, `PROXIMITY`, `SURFACE`, and `OCCLUSION` | More scenes and calibrated measurements are needed to distinguish support/contact from mere adjacency. Do not create `SUPPORT`, `CONTACT`, or `PART` EdgeTypes without revising the eight-category requirement. |
| Rich-surface grouping selectivity | `SURFACE` | The current cue-gated static trace stays at 4 `SURFACE` edges across two steps; an earlier `54→1,069` result predates the normal/material gates and cross-stream `PROXIMITY` routing, so it is not directly comparable. Validate these gates on more materials, slants, folds, and occlusion scenes without reintroducing temporal edge categories. |

## Preserved invariants

- `CONTAINMENT` remains directed container → member.
- Positive `OCCLUSION` remains directed near → far with sign `+1`; anti-phase segregation remains negative-sign `OCCLUSION`.
- `MATERIAL` is appearance-only; co-motion is gated off until approved video-backed representation exists.
- `PROXIMITY`, `CONTINUITY`, `JUNCTION`, `SURFACE`, `SHAPE`, and `MATERIAL` remain undirected.
- Population axes stay physical and category-specific; confidence is endpoint-minimum, separate from geometric signal.

See [`edge_classification.md`](edge_classification.md) for the current axis schemas and [`left_for_user.md`](left_for_user.md) for the concrete follow-up and test-environment limitations.
