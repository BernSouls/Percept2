# Two-Step Box-Over-Box Integration Trace

**Current result:** the repeated static-scene trace is stable for `SURFACE`, `SHAPE`, and `MATERIAL` after the measured-cue gates and cross-stream reclassification. This report reflects the current source, not the earlier historical trace.

## Scene and execution

The temporary trace harness rendered one 256×256 synthetic observation with `synthetic3d.occlusion_scene(front="box", back="box", ...)` and `supersample=2`. The foreground box used centre `(-0.08, -0.04, 0.55)`, pose `(0.18, -0.22, 0.08)`, and object-space stripes (`albedo=0.92`, frequency `10`, contrast `0.45`); the background used centre `(0.12, 0.08, 0.0)`, pose `(-0.15, 0.28, -0.12)`, and object-space noise (`albedo=0.58`, frequency `14`, contrast `0.35`). The same observation was passed through `Observation2D.from_image(..., rich=True)`, `SensoryFloor`, the Peak/Chain and V2/V4 lens passes, `SurfaceFragmentLens`, and Binder for `t=0` and `t=1`. The disabled `ContourLens` emitted no detections. The test did not feed renderer layer/material truth into perception.

| Step | Raw detections | Peaks | Chains | V2 detections | V4 detections | Surface fragments | Total graph edges |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,871 | 138 | 13 | 261 | 41 | 9 | 4,693 |
| 1 | 2,183 | 138 | 13 | 283 | 41 | 9 | 4,888 |

### Consolidated graph counts

| Category | After `t=0` | After `t=1` |
|---|---:|---:|
| `PROXIMITY` | 817 | 817 |
| `CONTINUITY` | 915 | 915 |
| `CONTAINMENT` | 29 | 29 |
| `OCCLUSION` | 3 | 3 |
| `JUNCTION` | 2,719 | 2,914 |
| `SURFACE` | 4 | 4 |
| `SHAPE` | 10 | 10 |
| `MATERIAL` | 196 | 196 |

The enum remains exactly these eight canonical categories. `SURFACE`, `SHAPE`, and `MATERIAL` each had 0% growth between the two steps. The graph's total grew by 195 edges, all in `JUNCTION`; the stable target-category counts do not imply that the whole graph is unchanged.

An earlier trace recorded `SURFACE=54→1,069`. That result predates the stricter spatial/normal/material gates and the change that routes cross-stream contacts without those cues to `PROXIMITY`; it is historical and not directly comparable to the current counts.

## Targeted contract checks

Temporary direct checks against the current Binder implementation passed:

- The enum contains exactly eight `EdgeType`s; no temporal edge category was added. Temporal `ConceptFamily` and action/schema values remain outside the emitted edge taxonomy.
- Quantized `COLOR`, `COLOR_LEGACY`, `TEXTURE`, and `SURFACE_PATCH` appearance slots are converted from bin indices to bin centers normalized to `[0,1]` using `POPULATION_SLOTS`. Circular `SURFACE_PATCH.hue` differences use the shortest arc.
- Positive controls emitted surface candidates when nearby normal/material cues agreed. Spatially distant pairs, disagreeing normals, dissimilar material, and missing required measurements abstained in the targeted grouping/shared-boundary checks. The measured SurfaceFragment fold path remains in place.
- Cross-stream contour/surface contacts without normal/material measurements emitted `PROXIMITY`, not `SURFACE`.
- Positive depth evidence remained directed near→far with sign `+1`. Negative figure-ground evidence remained signed `OCCLUSION` with `source_cue=3`; both signs survived on a shared test pair. `P.ANTIPHASE_ENABLED` remains false by default, so the static scene did not exercise the enabled anti-phase path.
- The kinematic co-motion emitter is absent. A direct same-direction-velocity check emitted no `MATERIAL`, and the canonicalizer rejects legacy co-motion/motion-opposition payloads when presented as `MATERIAL`.

## Validation and limits

- `python -m py_compile` passed for all tracked Python files after the edits.
- The two-step runtime trace and focused gate/sign checks passed using temporary import-path symlinks and compatibility shims under `/tmp`; those shims are not tracked.
- This checkout has no standard test suite. The previously referenced `diagnostics/_u20_gate.py` is absent, so no `WRONG=0` result is claimed.
- The broader dynamic-video, multi-object, and additional occlusion-scene cases remain outside this static trace. Co-motion should stay gated until video input and an approved kinematic representation are available.
