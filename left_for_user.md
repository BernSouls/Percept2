# Follow-Up Notes for the User

## Completed in this pass

- Preserved exactly eight canonical `EdgeType`s: `PROXIMITY`, `CONTINUITY`, `CONTAINMENT`, `OCCLUSION`, `JUNCTION`, `SURFACE`, `SHAPE`, and `MATERIAL`. Temporal `ConceptFamily` and action/schema values remain in the model; Binder does not claim to emit temporal edge types.
- Kept spatially gated surface grouping and the measured SurfaceFragment/legacy surface-fold paths. `SURFACE` grouping and shared-boundary emitters now require spatial proximity, compatible measured normal cues, and appearance similarity. Quantized color, texture, and surface-patch slots are compared as normalized bin centers; circular surface hue uses shortest-arc distance.
- Reclassified cross-stream contour/surface contacts that lack normal/material measurements as `PROXIMITY`, not `SURFACE`.
- Kept `MATERIAL` appearance-only and co-motion disabled pending video input plus an approved kinematic representation. Anti-phase figure-ground evidence is negative-sign `OCCLUSION` with `source_cue=3`; positive depth occlusion remains directed near→far with sign `+1`.
- Removed the four tracked root inline verification/diagnostic blocks and restored their original line endings. Updated the stale Binder, schema, parameter, and documentation claims.

## Verification

- `py_compile` passed for all tracked Python files; `git -c core.whitespace=cr-at-eol diff --check` passed (the override recognizes restored CRLF files).
- The two-step static box-over-box trace completed with the current source. `SURFACE=4→4`, `SHAPE=10→10`, and `MATERIAL=196→196`; the full category table is in [`integration_trace.md`](integration_trace.md).
- Temporary targeted checks passed for the eight-category enum, positive and negative `OCCLUSION` signs, co-motion refusal as `MATERIAL`, positive/negative `SURFACE` gate cases (spatial, normal, appearance, missing cues), and cross-stream `PROXIMITY` routing.
- Runtime checks used the repository code with temporary `/tmp` package-path symlinks and compatibility shims. No standard test suite is present, and the referenced `diagnostics/_u20_gate.py` is absent; no `WRONG=0` result is reported.

## Remaining validation

1. Run the pipeline through the intended packaged runtime/CI entry point once the repository's omitted top-level import packages and test entry point are available.
2. Exercise anti-phase with controlled dynamic/occlusion scenes before enabling `P.ANTIPHASE_ENABLED` (it remains off by default). Keep co-motion gated until both video input and an approved kinematic representation exist.
3. Test the surface gates on a broader variety of materials, slants, folds, and weak ownership cases. The current static image validates stability on one scene, not general perception quality.
4. Keep the historical `SURFACE=54→1,069` trace labeled as pre-gate/pre-routing behavior; it is not directly comparable to the current `4→4` trace.

Deferred refinements must remain within the eight canonical categories; see [`missing_edges.md`](missing_edges.md).
