# Follow-Up Notes for the User

## Completed and verified

The requested edge consolidation and pipeline wiring are implemented. The canonical enum and parameter-name set contain exactly `PROXIMITY`, `CONTINUITY`, `CONTAINMENT`, `OCCLUSION`, `JUNCTION`, `SURFACE`, `SHAPE`, and `MATERIAL`; physical population axes match across `core.schema` and the parameter registry. The contour spine uses `PEAK_CHAIN`, with `ContourLens` default-disabled. Binder now emits intra-step edges immediately, accumulates per-channel state across sub-passes, finalizes homeostasis once, retains geometric signal separately from endpoint-minimum confidence, preserves directed containment/occlusion, and routes co-motion and signed anti-phase evidence through `MATERIAL`.

Verification completed after the final code edits:

- All repository Python files passed `py_compile`.
- Temporary contract checks passed for the eight-category taxonomy and physical axes, `MATERIAL` co-motion with `confidence=min(z_a,z_b)`, negative anti-phase rerouting, and directed positive `CONTAINMENT`.
- Positive shape controls passed for radiality, concentricity, and spirality; a box-over-box chain emitted no false candidates for those three degeneracy-sensitive emitters.
- The full 256×256 two-step box-over-box trace passed. Its independently rendered images were identical, and two fresh pipeline runs matched exactly at `1,467` pattern nodes and `2,371` edges. See [`integration_trace.md`](integration_trace.md) for counts and measurements.

## Remaining follow-up

1. **Surface-grouping selectivity:** on the same static image, graph `SURFACE` edges rise from `54` after step 0 to `1,069` after step 1 (`1,015` new edges), with the increase occurring in the repeated V1 surface-grouping pass. The current trace does not isolate which raw detections or candidate pairs drive it. Inspect duplicate grouping and candidate gates while retaining the eight-category taxonomy.
2. **Dynamic multi-object validation:** co-motion and anti-phase routing passed targeted contract checks, but the image trace is intentionally static and does not validate longer dynamic sequences. `P.ANTIPHASE_ENABLED` remains disabled by default; the negative-to-`MATERIAL` route was tested directly, not exercised by the box image.
3. **Depth-layer corpus:** the test image produced 7 oriented and 2 depth-layer-only SurfaceFragment detections. The depth-only signatures correctly omit fabricated surface measurements. `Observation2D.get_t_junctions()` returned zero for this frame; ownership/fragment ordering still assigned ten fragments to two ordinal layers. Validate this path on additional occlusion geometries, especially frames with weak ownership probabilities.
4. **Contour gate:** `ContourLens` remains disabled by default as requested. Any future enabling should be validated against the `PEAK_CHAIN` spine and should not revive the former contour-structure path implicitly.
5. **Packaged test/runtime entry point:** this checkout contains no standard test suite and lacks the original top-level `perception.*`, `sensory_floor`, `learning`, and `memory` package wrappers. Runtime checks used temporary import-path symlinks and import-only compatibility stubs under `/tmp`; those are not committed. Add a reproducible package/test entry point before relying on these checks as CI coverage.

Earlier additional-edge proposals are superseded. Any deferred relation work should refine measurements, emitters, or gates within the eight canonical categories; see [`missing_edges.md`](missing_edges.md).
