# Consolidated Spatial Edge Classification

**Canonical taxonomy: exactly eight active `EdgeType` categories.** The temporal categories from the earlier draft have been removed; temporal change is not represented as a separate edge class.

| Category | Direction and sign contract | Physical population axes (bins; range) |
|---|---|---|
| `PROXIMITY` | Undirected; geometric relation between nearby anchors. | `distance` (8; `[0,4]`, `log1p(dist / extraction_scale)`), `bearing` (12; axial `[0,π)`), `scale_ratio` (8; `[-2,2]`). |
| `CONTINUITY` | Undirected; locally compatible contour/chain continuation. | `gap` (8; `[0,4]`), `turn_angle` (12; `[-π/2,π/2]`), `perp_offset` (8; `[0,3]`), `curv_agree` (8; `[-1,1]`). |
| `CONTAINMENT` | **Directed container → member.** | `rel_dist` (8; `[0,2]`), `polar_angle` (12; circular `[0,2π)`), `scale_ratio` (8; `[-3,1]`). |
| `OCCLUSION` | **Directed.** `sign=+1` identifies near → far depth; `sign=-1` preserves repulsive figure-ground evidence (`source_cue=3`). | `depth_diff` (8; `[0,4]`), `source_cue` (4; `[0,3]`), `strength` (8; `[0,1]`). |
| `JUNCTION` | Undirected; local topology around a shared junction. | `angle_delta` (12; `[0,π/2]`), `topo_class` (7; `[0,6]`), `arm_count` (6; `[2,7]`). |
| `SURFACE` | Undirected; nearby surface continuity requires normal agreement and similar appearance; a fold uses measured dihedral plus material similarity. Cross-stream contacts without those cues use `PROXIMITY`. | `normal_angle` (12; `[0,π]`), `material_diff` (8; `[0,1]`), `slant_diff` (8; `[0,π/2]`). |
| `SHAPE` | Undirected; radiality, concentricity, symmetry, and spiral evidence. | `sym_score` (8; `[0,1]`), `radial_ratio` (8; `[0,3]`), `curvature` (8; `[-1,1]`). |
| `MATERIAL` | Undirected; appearance differences only. Co-motion is gated off, and anti-phase evidence remains signed `OCCLUSION`. | `chrom_diff` (8; `[0,1]`), `lum_diff` (8; `[0,1]`), `tex_diff` (8; `[0,1]`). |

`core/schema.py::EDGE_SCHEMA` is the schema source of truth. `EDGE_POPULATION_AXES` and `EDGE_UNIT_TABLE` accept the `EdgeType`, canonical category name, and legacy writer key `EDGE_<NAME>`, but all stored edge parameters are canonicalized to one of the eight category schemas. The parameter-level `EDGE_TYPE_NAMES` contains the same eight names. `FAMILY_FORMING_EDGE_TYPES` aliases that set; `NON_FAMILY_FORMING_EDGE_TYPES` is empty.

## Binder contracts

- V1/V2/V4 and other intra-step readers emit edges as soon as their sub-pass completes. Active/emitted counts and `M_prev` accumulate across sub-passes; channel thresholds and homeostasis advance once in `finalize_step()`.
- Stored edge confidence is `min(z_a, z_b)`. Candidate geometric signal remains available to the relation competition and is retained separately from confidence.
- Every `SURFACE` emitter requires cues measured for its relation: grouping/shared-boundary edges are spatially local and gate on compatible normals and appearance; fold edges require measured nearby normals plus compatible material. Cross-stream contact lacking normal/material evidence is `PROXIMITY`, not `SURFACE`.
- Quantized `COLOR`, `COLOR_LEGACY`, `TEXTURE`, and `SURFACE_PATCH` appearance slots used by material gates are converted from bin indices to bin centers normalized to `[0,1]` via `POPULATION_SLOTS`. Circular surface hue uses shortest-arc distance. Surface grouping uses a local spatial hash, a ≤30° normal-agreement gate, and a ≤0.30 normalized material-difference gate.
- `CONTAINMENT` preserves container-to-member endpoint order. Positive `OCCLUSION` is near-to-far; negative-sign segregation remains `OCCLUSION` and uses `source_cue=3`.
- `MATERIAL` carries appearance evidence only. Co-motion is gated off without video input and an approved representation; no temporal `EdgeType` is emitted.
- The contour spine is `PeakLens` → `ChainLens` (`PEAK_CHAIN`). `ContourLens` is behind a disabled-by-default gate. Junction, continuity, and occlusion geometry use nearest member peaks and local chain-vertex tangents.

See [`integration_trace.md`](integration_trace.md) for the two-step box-over-box run, physical SurfaceFragment measurements, shape positive/negative controls, and determinism results. Earlier proposals for additional edge classes have been superseded; see [`missing_edges.md`](missing_edges.md) for deferred refinements that must stay within this taxonomy.
