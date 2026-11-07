# Two-Step Box-Over-Box Integration Trace

**Status: runtime-verified for the requested consolidation, SurfaceFragment path, homeostasis lifecycle, and determinism checks.** The trace uses the same static image at `t=0` and `t=1`; it does not use renderer truth as a perception input.

## Scene and execution

`synthetic3d.occlusion_scene(front="box", back="box", width=256, height=256)` with `supersample=2`:

- Foreground box: centre `(-0.08, -0.04, 0.55)`, pose `(0.18, -0.22, 0.08)`, object-space stripes (`albedo=0.92`, frequency `10`, contrast `0.45`).
- Background box: centre `(0.12, 0.08, 0.0)`, pose `(-0.15, 0.28, -0.12)`, object-space isotropic noise (`albedo=0.58`, frequency `14`, contrast `0.35`).
- Renderer-only visible-pixel counts: canvas `51,822`, foreground layer `0`: `10,152`, background layer `1`: `3,562`.

Two independently rendered images were byte-identical. Each run called `Observation2D.from_image(..., rich=True)`, `SensoryFloor`, `PeakLens`, `ChainLens`, `JunctionLens`, the disabled `ContourLens` gate, `SurfaceFragmentLens`, and Binder V1/V2/V4/LO-PIT passes over two steps.

## Step counts and consolidated graph

| Step | Raw detections | `CONTOUR_PEAK` | `PEAK_CHAIN` | `Junction` | `Contour` | SurfaceFragment detections | Total graph edges |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,871 | 138 | 13 | 23 | 0 | 9 | 1,356 |
| 1 | 2,183 | 138 | 13 | 23 | 0 | 9 | 2,371 |

`ContourLens` returned no detections with its default-off gate. The eight-category enum and parameter name set are exactly `PROXIMITY`, `CONTINUITY`, `CONTAINMENT`, `OCCLUSION`, `JUNCTION`, `SURFACE`, `SHAPE`, and `MATERIAL`; no temporal edge categories were present.

| Consolidated category | Edges after `t=0` | Edges after `t=1` |
|---|---:|---:|
| `PROXIMITY` | 787 | 787 |
| `CONTINUITY` | 126 | 126 |
| `CONTAINMENT` | 65 | 65 |
| `OCCLUSION` | 13 | 13 |
| `JUNCTION` | 286 | 286 |
| `SURFACE` | 54 | 1,069 |
| `SHAPE` | 0 | 0 |
| `MATERIAL` | 25 | 25 |

The increase of 1,015 `SURFACE` edges at `t=1` occurs during the existing V1 surface-grouping pass on the repeated static observation. This trace does not isolate which richer raw detections or candidate pairs account for the increase, so it remains a selectivity item—not a temporal-edge flood. No `SHAPE` candidate survived the box-specific radiality, spirality, and concentricity gates.

## `PEAK_CHAIN` → ownership → light → fragments → Binder

- The image adapter built `269` observation atoms and `123` fragment ownership records from the `PeakLens`/`ChainLens` spine.
- It fitted and retained an image-based `LightPrior` with direction approximately `(0.070513, 0.069235, 0.995105)` and ambient `0.0`; no renderer light or material truth was passed to the estimator.
- Fragment construction used the calibrated widths `sigma_contour=(3.0, 10.9)` and `shading_sigma=(3.0, 10.9)`. The 123 fragments included ordinal layers `{0, 1}` (five layer-labelled fragments per layer in this render).
- `SurfaceFragmentLens` emitted 9 detections: 7 with a measurable normal signature and 2 depth-layer-only detections. The latter carry an explicit `SURFACE_FRAGMENT_DEPTH_LAYER` relation and do not fabricate slant, tilt, curvature, or confidence bins.
- `SurfaceFragmentLens.positions` remains `alpha -> list[(x, y)]`; its separate `centroid_positions` map exposed 9 centroid anchors to Binder.
- Binder emitted 4 measured SurfaceFragment normal/dihedral `SURFACE` edges, 2 SurfaceFragment material-difference `MATERIAL` edges, and a direct depth-layer `OCCLUSION` edge from layer `0` to layer `1`. That depth edge had `sign=+1`, `depth_diff=1`, `source_cue=2`, geometric signal `0.381517`, and `EdgeParams.confidence=0.856851` (the minimum endpoint activation).

`Observation2D.get_t_junctions()` returned zero on this frame; the fragment ordering path still assigned layers from the ownership/T-junction ordering evidence available to `surface_fragments`.

## Contract and shape checks

- `core.schema.EDGE_SCHEMA` has exactly the eight consolidated categories and physical axes; lookups work by `EdgeType`, canonical name, and legacy `EDGE_<NAME>` writer key.
- Contract checks passed for co-motion folding into `MATERIAL`, negative anti-phase OCCLUSION candidates rerouting to signed `MATERIAL`, positive-only OCCLUSION, and directed `CONTAINMENT` (`container -> member`).
- Before `finalize_step()`, category thresholds and `M_prev` remained unchanged across V1/V2/V4 sub-passes. After finalization, the observed channel EMA values matched one aggregate update per category.
- Box-chain shape-emitter candidate counts were `concentricity=0`, `radiality=0`, `spirality=0`. Positive controls still fired: a three-spoke radial star produced 3 radiality edges, nested closed circles produced 1 concentricity edge, and a paired Archimedean-spiral test produced 1 spirality edge.
- `python -m py_compile` succeeded for all repository Python files.

## Determinism and test-environment note

Two independent full runs produced identical graph digests: `1,467` pattern nodes and `2,371` edges, including endpoints, category, pending geometric signal, and canonical parameter protos. The two synthetic renders were also byte-identical.

This checkout has no standard test suite or installed package shim for its extracted `perception.*` modules (and omits the original `sensory_floor`, `learning`, and `memory` import packages). Runtime verification used temporary `/tmp` package-path symlinks and import-only compatibility stubs; those shims are not part of the repository patch. The exercised perception, Binder, schema, and observation implementations were the repository files. A packaged CI/test entry point remains a separate integration task.
