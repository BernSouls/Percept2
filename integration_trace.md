# End-to-End Integration Trace: Box-Over-Box Occlusion Scene (`integration_trace.md`)

**Scene Specification**: `perception.synthetic3d.occlusion_scene(front="box", back="box", width=256, height=256)`  
- **Foreground Object (`layer = 0`)**: Rotated 3D `Box` at `front_centre = (-0.08, -0.04, 0.55)`, `front_pose = (0.18, -0.22, 0.08)`, continuous object-space striped Lambertian material (`albedo=0.92, texture="stripes", texture_freq=10.0, texture_contrast=0.45, texture_space="object"`).
- **Background Object (`layer = 1`)**: Rotated 3D `Box` at `back_centre = (0.12, 0.08, 0.0)`, `back_pose = (-0.15, 0.28, -0.12)`, continuous object-space isotropic noise material (`albedo=0.58, texture="noise", texture_freq=14.0, texture_contrast=0.35, texture_space="object"`).
- **Rendered Ground Truth (`supersample=2`)**:
  - Image tensor: `(256, 256, 3)` `float32` (`65,536` pixels total)
  - Visible foreground pixels (`layer == 0`): **`10,152` px** (`15.49%`)
  - Visible partially-occluded background pixels (`layer == 1`): **`3,562` px** (`5.44%`)
  - Background canvas pixels (`layer == -1`): **`51,822` px** (`79.07%`)

---

## 1. Stage-by-Stage Execution Walkthrough (`t = 0` and `t = 1`)

### Stage 1: Pixel-to-Region Front-End (`Observation2D.from_image(rt.image, rich=True)`)
1. **Opponent Retinex & Luminance Adaptation (`filters.py`)**:
   - Computes opponent planes `(L, RG, BY)`, white-patch color constancy, local chromatic contrast, multi-scale `compute_opponent_retinex` (now protected against negative float `log10` `NaN`s), and Naka–Rushton adapted luminance `_lum_adapted`.
2. **Gabor Filterbank & B1/B4 Rich Signatures (`filters.py`, `rich.py`)**:
   - Runs `12 orientations x 5 spatial frequencies = 60` quadrature Gabor pairs on `_lum_adapted`, ON-split, and OFF-split planes; caches raw V1 responses via `cache_v1_responses` so `PeakLens` reuses them without reconvolving.
   - Registers `ORIENTATION_RICH_12_5` (`433` slots + 3-axis `(theta, sf, joint)` graded population) and `TEXTURE_RICH_12_5` (`244` slots + 3-axis graded population). Zero-energy background windows now cleanly emit `population = ()` (`rich.py` fix).
   - Executes per-region rich extraction **once** across `all_regions` (eliminating the duplicate overwritten `hypercolumn_regions` pass in `observation_2d.py`).
3. **Stream 1 + Stream 2 Entity Construction**:
   - **Total `Observation2D` atoms**: **`269`**
     - Surface stream (`V1RetinotopicLattice` `16x16` hypercolumns): **`256` anchors** (`0..255`)
     - Structural stream (association-field contours): **`13` anchors** (`256..268`)
   - **Cross-stream contour–surface contacts**: **`106` pairs**
   - **Median extraction scale (`extraction_scale`)**: **`12.800` px**

---

### Stage 2: SensoryFloor, Raw V1 Lenses & V1.5 Field Lenses
1. **`SensoryFloor.step(obs)` (`floor.py`, `base.py`)**:
   - Constructs validated `StatePattern` `S_t` (`269 x 16`, with `STATE_CHANNEL_NAMES[9] == "clutter_count"` aligned in `observation.py`).
   - `LensExecutor.run_all` emits raw V1 detections:
     - `Boundary`: **`269`**, `Color`: **`269`**, `Position`: **`269`**, `Texture`: **`269`** (`361` at `t=1` as rich B4 contrast gates cross hysteresis), `Onset`: **`64`**
     - `Orientation`: **`536`** (`756` at `t=1` with rich B1 hysteresis), `OrientationPol`: **`146`**, `ContourFragment`: **`49`**
2. **`PeakLens.step` & `EndStoppedLens.step` (`peak_lens.py`, `contour_field.py`, `derived_base.py`)**:
   - `PeakLens` builds the continuous V1/V2 association field (`field_from_luminance`) and allocates anchors `PEAK_ANCHOR_BASE + 0 .. PEAK_ANCHOR_BASE + 137`:
     - `CONTOUR_PEAK`: **`138` peaks**
   - `EndStoppedLens` runs `compute_endstopped_map` (now stepping along the true `[Q1]` bar tangent `(cos th, sin th)` from our `derived_common.py` fix):
     - `EndStopped`: **`43` line/edge terminations** (at box corners and T-junction stems!)

---

### Stage 3: V1 Order-1 Relational Binding (`Binder.step`)
`Binder.step(t, D_v1, static_frame=True, ...)` mints pattern nodes, stages `_v1_dets_this_step` and `_v1_node_ids` for V2 cross-tier channels (`binder.py` fix), and emits V1 spatial/feature edges:
- `CO_LOCATED`: **`767`** (binds `Color`, `Texture`, `Position`, `Orientation` sharing each hypercolumn/contour anchor)
- `ORIENTATION_CONTINUITY`: **`915`** (port winner-take-all association links over V1 segments)
- `ORIENTATION_BEND`: **`915`** (local turn angles, now wrap-safe across `0 / pi`)
- `ORIENTATION_JUNCTION`: **`1,776`** (crossing/meeting V1 segment pairs)
- `TOPOLOGY`: **`24`** & `DEGREE`: **`2`** (over the `13` structural contour anchors; 2D `(x, y)` anchors now accepted)
- `CONTACT`: **`13`** (spatial + cross-stream contour-surface contacts)

---

### Stage 4: V2 Mid-Level Grouping Lenses & `emit_pending_edges("v2")`
1. **V2 Lenses (`chain_lens.py`, `corner_lens.py`, `v2.py`)**:
   - `ChainLens`: Forms **`13` `PEAK_CHAIN`s** across the `138` contour peaks (previously crashed with `(138, 32)` vs `(1, 138)` broadcast error in `chain_formation.py` because `138 > 32`!).
   - `CornerLens`: Emits **`5` `Corner`** detections at sharp chain bends.
   - `BorderOwnershipLens`: Emits **`123` `BorderOwnership`** fragment claims across the `13` chains' member peaks.
   - `ContourLens` & `JunctionLens`: Group the `ORIENTATION_CONTINUITY` spine into **`998` `Contour`** and **`32` `Junction`** detections.
   - `TextureBoundaryLens`: Measures flank second-order/opponent discontinuity across the `13` chains (`chain_flanks.py` open-endpoint clamp fix), emitting **`123` `TextureBoundary`** detections separating the striped foreground box from the noisy background box and smooth background!
2. **V2 Relational Binding (`binder.emit_pending_edges("v2")`)**:
   - `JUNCTION_CONTOUR`: **`430`**, `JUNCTION_ORIENTATION`: **`195`**
   - `CONTOUR_ORIENTATION`: **`206`**, `CONTOUR_CONTINUITY`: **`121`**
   - `CURVATURE_SIGN_CONTINUITY`: **`342`** (now properly centred at `(q[6] - 4.0) / 4.0`)
   - `COCIRCULAR`: **`89`**, `PARALLEL`: **`19`**, `AMODAL_LINK`: **`44`**
   - `COLOR_OPPONENT_BORDER`: **`23`** *(0 pre-fix; restored by V1->V2 node staging!)*
   - `LINE_TERMINATION`: **`8`** (`2` inhibitory `sign = -1` termination claims)
   - `BORDER_OWNERSHIP_EDGE`: **`1`**

---

### Stage 5: V4 Surface / Shape Lenses & `emit_pending_edges("v4" / "lo_pit")`
1. **V4 Lenses (`v4.py`)**:
   - `SurfaceLens`: Emits **`13` `Surface` (`SURFACE_PATCH`)** detections — one per `PEAK_CHAIN`, now matching all **`123` `BorderOwnership`** member claims via `"member"` role anchors (`v4.py` fix!).
   - `IllusoryContourLens`: Emits **`7` `IllusoryContour`** bridges between aligned `EndStopped` terminations, registering midpoint coordinates in `anchor_positions` (`v4.py` fix!).
   - `CurveComplexLens`: Bundles connected V2 `Contour` + `Junction` components into **`69` `CurveComplex`** arity-24 descriptors.
   - `RegionShapeLens`: Emits **`10` `RegionShape`** boundary conformation descriptors in deterministic anchor order.
2. **V4 & LO/PIT Binding (`emit_pending_edges("v4")`, `emit_pending_edges("lo_pit")`)**:
   - `CURVE_CONTAINMENT`: **`52`**, `PART_CONTAINMENT`: **`11`**, `REGION_CONTENT`: **`57`**
   - `SYMMETRY`: **`214`**, `TEXTURE_BOUNDARY_EDGE`: **`171`**
   - `CONCENTRICITY`: **`36`** *(crashed with `ValueError` pre-fix; now runs cleanly)*
   - `RADIALITY`: **`345`**, `SPIRALITY`: **`101`**

---

## 2. Quantitative Comparison Across Modes (`t=0`, `t=1`, and Wired 2.5D + Anti-Phase)

| Metric | `t = 0` (Default Flags) | `t = 1` (2nd Step, Default) | `t = 0` (`SurfaceFragments` Wired + `ANTIPHASE_ENABLED=True`) |
|---|---:|---:|---:|
| **Pattern Nodes in `ConceptGraph`** | `1,464` | `1,776` | `1,473` (`+9` `SurfaceFragment` nodes) |
| **Total Edges in `ConceptGraph`** | `6,877` | `13,893` | `6,879` |
| **Distinct Active `EdgeType`s** | `26` | `30` | `27` (`+OCCLUSION`) |
| **`SurfaceFragment` Detections** | `0` *(unwired)* | `0` *(unwired)* | **`9`** *(collapsed from `123` fragments)* |
| **Ordinal Depth Layers (`assign_depth_layers`)** | — | — | **`{0: 3, 1: 3}`** *(exact 2-layer separation!)* |
| **T-Junction Sites (`t_junction_sites`)** | — | — | **`2` occlusion T-sites** |
| **Anti-Phase `OCCLUSION` (`sign=-1`)** | `0` *(flag off)* | `0` *(flag off)* | **`3` offered, `3` emitted, `0` unresolved** |
| **Non-Perceptual Temporal Flood (`PERSISTENCE + SEQUENCE + TRANSITION`)** | `0` | **`6,000`** (`2,000` each!) | `0` |

---

## 3. Five Key Architectural Insights Proven by the Live Trace

1. **V1->V2 Staged Node Visibility Fix Validated**:
   - With `P.ANTIPHASE_ENABLED = True`, `anti_phase_pairs_offered` produced `3` figure-over-ground containment pairs and **all `3` resolved cleanly into directed `(near -> far)` `OCCLUSION` edges (`anti_phase_unresolved = {}`)**, while `COLOR_OPPONENT_BORDER` emitted `23` edges. Prior to staging `_v1_dets_this_step` / `_v1_node_ids`, V1 interior endpoints were invisible in `emit_pending_edges("v2")`.
2. **Temporal/Statistical Edge Flood Proven Empirically (`+6,000` Noise Edges at `t=1`)**:
   - Stepping from `t=0` to `t=1` on the **exact same static image** injected `2,000 PERSISTENCE + 2,000 SEQUENCE + 2,000 TRANSITION = 6,000` cap-saturated non-perceptual edges (`43.2%` of the entire graph!), confirming that the 7 `DELETE` channels in `edge_classification.md` severely corrupt perceptual subgraphs.
3. **`SurfaceFragmentLens` Requires Upstream Pipeline Wiring**:
   - `Observation2D.from_image` does not populate `obs.surface_fragments`. When `fragment_ownership_records` -> `build_surface_fragments` -> `assign_depth_layers` is wired explicitly, the 123 chain-member ownership records produce 123 2.5D `SurfaceFragment`s (collapsing by content hash to `9` canonical `(slant, tilt, curv_sign, confidence)` detections), `2` T-junction sites, and exact `{layer 0: 3, layer 1: 3}` foreground/background ordinal depth separation!
4. **Dual Contour Spine Multiplicity (`13 PEAK_CHAIN`s vs `998 Contour`s)**:
   - Stream 2's continuous contour field extracts **`13` clean `PEAK_CHAIN`s** (which drive `BorderOwnership: 123`, `TextureBoundary: 123`, `Surface: 13`, `RegionShape: 10`, and `SurfaceFragment: 9`), whereas `ContourLens` walking V1 `ORIENTATION_CONTINUITY` emits **`998` overlapping `Contour` sub-chains** (driving `CurveComplex: 69` and `JUNCTION_CONTOUR: 430`). Unifying `ContourLens` onto `PEAK_CHAIN` will cut graph bloat by an order of magnitude.
5. **Polar V4 Channels (`RADIALITY: 345`, `SPIRALITY: 101`) Over-Fire on Rectilinear Boxes**:
   - Two polyhedral boxes with zero curved or spiral geometry triggered `345 RADIALITY`, `101 SPIRALITY`, and `36 CONCENTRICITY` edges because pairs/triples of straight box contours at corner vertices satisfy loose angular tolerances. Tightening spoke-multiplicity (`>= 4` spokes) and nonzero monotonic curvature gates will make V4 polar channels selective for genuine radial/spiral forms.
