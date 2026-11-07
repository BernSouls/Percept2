# Fixes Applied (`FIX NOW` Mechanical Bug Repairs)

**Branch**: `arena/0fccf1ef-percept2`  
**Diff Summary**: `19 files changed, 437 insertions(+), 475 deletions(-)`  
**Verification**: `43/43` modules import cleanly; `17/17` unit test suites pass; full 2-step `occlusion_scene(front="box", back="box")` end-to-end trace passes (`exit_code=0`).

---

## Summary Table of Modified Files

| # | File | Defect Category | Key Functions Repaired | Verification Suite |
|---:|---|---|---|---|
| 1 | `observation.py` | Schema index mismatch | `STATE_CHANNEL_NAMES` (`"clutter_count"` index `15 -> 9`) | Suite 3 + E2E Trace |
| 2 | `observation_2d.py` | Geometry + dead duplicate pass | `_polyline_shape_invariants`, `from_image` | Suite 4 + E2E Trace |
| 3 | `rich.py` | Zero-energy population invariant | `b1_region_signature` | Suite 5 + E2E Trace |
| 4 | `contour_structure.py` | `(x, y)` `atan2` swap + unit bug | `_tangent_at`, `_detect_t_junctions`, `extract_region_features` | Suite 6 + E2E Trace |
| 5 | `filters.py` | Negative log `NaN` + small pad | `compute_opponent_retinex`, `_conv2d` | Suite 1 + E2E Trace |
| 6 | `contour_field.py` | Small-image reflect pad crash | `_convolve_same` | Suite 2 + E2E Trace |
| 7 | `chain_formation.py` | `(N, 32)` vs `(1, N)` broadcast crash | `_candidate_pairs` | Suite 7 (`50` peaks) + E2E (`138` peaks) |
| 8 | `chain_flanks.py` | Open-chain modular wrap bug | `perpendicular_offsets`, `chain_flank_fields` | Suite 8 + E2E Trace |
| 9 | `chain_lens.py` | Missing ring closing chord | `ChainLens.step` | E2E Trace |
| 10 | `derived_common.py` | `[Q1]` normal vs tangent step | `compute_endstopped_map` | Suite 9 + E2E Trace |
| 11 | `anti_phase.py` | Duplicate shadowed function/guard | `_claim_endpoints`, `opposite_ownership_pairs` | E2E Trace (`ANTIPHASE_ENABLED=True`) |
| 12 | `base.py` | `(x, y)` `atan2` + `ABSENT` sentinel + sort | `_segment_polyline`, `_emit`, `_run_orientation_pol`, `PolarityLens.step` | Suite 10 + E2E Trace |
| 13 | `v4.py` | Member BO lookup + `IllusoryContour` + sort | `SurfaceLens.step`, `IllusoryContourLens.step`, `RegionShapeLens.step` | Suite 14 + E2E Trace |
| 14 | `relatable_completion.py` | Missing `CF` + split off-by-one + curvature | `_split_at_junctions`, `_make_arc`, `relatable_completion` | Suite 13 + E2E Trace |
| 15 | `ring_hypothesis.py` | `+pi/2` `[Q1]` error + `cos/sin` + DP closure | `seed_ring_centers`, `trace_hypotheses`, `_viterbi_ring` | Suite 11 (`24/24` circle peaks) |
| 16 | `surface_fragments.py` | Endpoint curvature window + self-loops | `contour_curvature`, `t_junction_arms` | Suite 12 + E2E Trace |
| 17 | `concept_graph.py` | Empty-to-populated posterior count | `_accumulate_pattern_population` | Suite 16 + E2E Trace |
| 18 | `synthetic3d.py` | Object-space UV + `width/height` drop | `Scene._albedo_at`, `single_object_scene`, `occlusion_scene` | Suite 17 + E2E Trace |
| 19 | `binder.py` | V1->V2 staging + directedness + 3-tuple + axial mean | `step`, `emit_pending_edges`, `_surface_normal`, `_run_anti_phase_containment`, `_run_curvature_sign_continuity`, `_run_concentricity`, `_run_topology`, `_run_degree`, `_run_orientation_angle_relation` | Suite 15, 15b + E2E Trace |

---

## Detailed File-by-File Change Log

### 1. `observation.py`
- **Root Cause**: `STATE_CHANNEL_NAMES` placed `"clutter_count"` at index `15`, shifting `"mean_r"` (`10`), `"mean_g"` (`11`), `"mean_b"` (`12`), `"Size_norm"` (`13`), `"Motion_mag"` (`14`), and `"boundary_Persistence"` (`15`) off by 1 relative to `SensoryFloor._construct_3d_state` / `_construct_symbolic_state` in `floor.py`.
- **Fix**: Moved `"clutter_count"` to index `9` so all 16 channel names match `floor.py` byte-for-byte.

### 2. `observation_2d.py`
- **Root Cause**:
  1. `_polyline_shape_invariants` computed shoelace polygon area on open polylines (producing arbitrary chord-dependent area/circularity), did not subtract `(cx, cy)` before cross-products, and skipped the two wrap-around vertices when measuring closed-polyline turn roughness.
  2. `from_image` had a duplicate `@classmethod` decorator.
  3. `from_image` ran an ~130-line rich-feature extraction loop over `hypercolumn_regions` and then immediately ran an identical capped loop over `all_regions = hypercolumn_regions + contour_regions`, overwriting every dict entry computed in the first loop.
- **Fix**: Restricted shoelace area/circularity to `is_closed == True` with centroid-shifted coordinates, computed all `n` modular vertex turns on closed contours, removed the duplicate `@classmethod`, and eliminated the redundant overwritten `hypercolumn_regions` pass.

### 3. `rich.py`
- **Root Cause**: In `b1_region_signature`, zero-energy windows (`win_energy.sum() <= _TINY`) fell through to division by `total_energy = 1e-9`, emitting three all-zero tuples `((0.0,...), (0.0,...), (0.0,...))` instead of `population = ()`.
- **Fix**: Guarded `raw_energy = float(win_energy.sum()) <= _TINY` to return `population = ()`.

### 4. `contour_structure.py`
- **Root Cause**:
  1. Polylines in `contour_structure.py` store `(x, y)` pixel coordinates, but `_tangent_at` and `_detect_t_junctions` unpacked them as `(r, c)` and called `math.atan2(dx, dy)`, reflecting every tangent across `y = x`.
  2. `extract_region_features` computed `r90_sq = float(dists_sorted[i])` (missing `** 2`) when `(n - 1) * 0.90` was an exact integer, mixing `px` and `px^2` in `area / (pi * r90_sq)`.
- **Fix**: Unpacked `(x0, y0)` and computed `math.atan2(dy, dx)`; squared `float(dists_sorted[i]) ** 2` in the integer quantile branch. Verified on a synthetic disk (`circularity = 0.850` vs `~0.08` pre-fix).

### 5. `filters.py`
- **Root Cause**: `compute_opponent_retinex` fed unclamped `rgb_float` and `lms` into `np.log10(lms + 1e-4)`, producing `NaN` on negative undershoot; `_conv2d` used `k.shape[0] // 2` for both axes and failed when `pad >= h` or `pad >= w`.
- **Fix**: Clamped `rgb_float` and `lms` at `0.0` prior to `log10`; used rectangular `(ph, pw)` padding with `mode="edge"` fallback when kernel half-width meets or exceeds image dimensions.

### 6. `contour_field.py`
- **Root Cause**: `_convolve_same` called `np.pad(..., mode="reflect")` unconditionally, raising `ValueError` when `ph >= H` or `pw >= W` on small crops.
- **Fix**: Used `pad_mode = "reflect" if (ph < H and pw < W) else "edge"`, keeping normal `>= 32x32` frames bit-identical under `_FIELD_FORMULA_VERSION = 7`.

### 7. `chain_formation.py`
- **Root Cause**: `_candidate_pairs` truncated the spatial neighbor table to `K_MAX = 32` columns (`cols` shape `(N, 32)`), then computed `dx = xs[None, :] - xs[:, None]` / `th[None, :]` with full width `N`, raising `ValueError: operands could not be broadcast together with shapes (N,32) (1,N)` whenever `len(peaks) > 32`.
- **Fix**: Replaced full-row `[None, :]` indexing with `xs[cols]`, `ys[cols]`, and `th[cols]` of shape `(N, K)`. Verified on `50` circle peaks and `138` box-over-box peaks.

### 8. `chain_flanks.py`
- **Root Cause**: `perpendicular_offsets` indexed `(i - 1) % n` and `(i + 1) % n` unconditionally, wrapping tangents across the open endpoints of open chains.
- **Fix**: Added `closed: bool = False` to `perpendicular_offsets` (clamping `0` and `n - 1` when `not closed`) and passed `closed=is_closed` from `chain_flank_fields`.

### 9. `chain_lens.py`
- **Root Cause**: `ChainLens.step` omitted the closing chord `peaks[members[-1]] -> peaks[members[0]]` when accumulating `length` for `is_ring == True`.
- **Fix**: Added closing chord Euclidean distance when `is_ring and len(members) >= 3`.

### 10. `derived_common.py`
- **Root Cause**: `compute_endstopped_map` stepped along `ay = -math.cos(th), ax = math.sin(th)` (`th + pi/2`), perpendicular to `[Q1]` bar-tangent `th`, measuring cross-bar contrast instead of along-bar termination.
- **Fix**: Changed step vector to `ay = math.sin(th), ax = math.cos(th)`.

### 11. `anti_phase.py`
- **Root Cause**: Contained a shadowed duplicate `_claim_endpoints` definition and a duplicate `if min_strength is None:` block in `opposite_ownership_pairs`.
- **Fix**: Removed both duplicate blocks cleanly.

### 12. `base.py`
- **Root Cause**:
  1. `_segment_polyline` unpacked `(x, y)` contour polylines as `(r, c)` and called `atan2(dx, dy)`.
  2. `_emit` stored hysteresis state in `M_prev` without guarding tuple key canonicalization.
  3. `_run_orientation_pol` added `on_dtheta` / `off_dtheta` without checking `!= ABSENT (-1.0)`.
  4. `PolarityLens.step` iterated anchors in non-canonical order.
- **Fix**: Repaired `(x, y)` `atan2(dy, dx)`, canonicalized `M_prev` keys, guarded `ABSENT` slots in `_run_orientation_pol`, and sorted `PolarityLens` anchors deterministically.

### 13. `v4.py`
- **Root Cause**:
  1. `SurfaceLens.step` indexed `bo_by[chain_det.alpha]`, never matching `BorderOwnership` detections emitted on chain member anchors (`PEAK_ANCHOR_BASE + i`).
  2. `IllusoryContourLens.step` called `slot_value(d.sigma, "ORIENTATION", ...)` on `"ENDSTOPPED"` signatures, sorted detections with raw `<` on `Detection`, lacked deterministic tie-breaking, and omitted writing `(mx, my)` into `anchor_positions[alpha]`.
  3. `RegionShapeLens.step` iterated `polys.items()` without deterministic key sorting.
- **Fix**: Collected `BorderOwnership` detections across chain `"member"` role anchors in `SurfaceLens`; fixed `"ENDSTOPPED"` slot lookups, `_detection_sort_key`, tie-break sorting, and `anchor_positions[alpha] = (mx, my)` in `IllusoryContourLens`; sorted `polys.items()` by `serialize_value(kv[0])` in `RegionShapeLens`.

### 14. `relatable_completion.py`
- **Root Cause**: Missing `CF` (`chain_formation`) import; off-by-one `jstart` and `part_closed` bugs in `_split_at_junctions`; constant `curvature = 0.0` on curved Hermite bridges.
- **Fix**: Imported `chain_formation as CF`; fixed sub-chain slicing (`part_closed = False`) and endpoint port tangents; computed exact cubic-Hermite midpoint curvature `|x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)` and turn-angle bound.

### 15. `ring_hypothesis.py`
- **Root Cause**: Added `+ pi/2` to `[Q1]` bar-tangent `peak["theta"]` at three sites, used `|cos(th - phi)|` instead of `|sin(th - phi)|` for tangential alignment around a ring center, failed to pin bin `0` to `start_choice` in circular Viterbi DP, and sorted hypotheses ascending instead of descending by `llr_ev`.
- **Fix**: Removed all three spurious `+ pi/2` offsets, changed tangential alignment to `abs(math.sin(th - phi))`, ran per-`start_choice` pinned Viterbi DP so bin `B - 1 -> bin 0` wraps validly to `start_choice`, and sorted descending by `-h["llr_ev"]`.

### 16. `surface_fragments.py`
- **Root Cause**: `contour_curvature` used `[max(0, index - w) : min(n, index + w + 1)]`, truncating endpoint windows to `w + 1` points on open chains; `t_junction_arms` allowed self-loops `(a, a)` and returned raw index pairs instead of deduplicated `resolved` alpha pairs.
- **Fix**: Shifted open-chain windows to preserve `min(n, 2*w + 1)` support at endpoints; filtered `a == b` self-loops and returned `resolved` pairs in `t_junction_arms`.

### 17. `concept_graph.py`
- **Root Cause**: `_accumulate_pattern_population` used `n_prev = float(self._pattern_node_population_count.get(nid, 1))` without checking `if prev_pop else 0.0`, causing nodes created initially with `population = ()` to weight an empty prior if count defaulted nonzero.
- **Fix**: Computed `n_prev = float(self._pattern_node_population_count.get(nid, 1)) if prev_pop else 0.0`, ensuring `0 -> 1 -> 2` exact running-mean posteriors.

### 18. `synthetic3d.py`
- **Root Cause**: `Scene._albedo_at` ignored `Material.uv_for(p_local)` when `mat.texture_space == "object"`; `single_object_scene` and `occlusion_scene` dropped `width` and `height` when instantiating `Scene(...)`.
- **Fix**: Evaluated `mat.uv_for(p_local)` in `Scene._albedo_at` for `"object"`-space materials (`diff_l1 = 0.1434` vs `"face"` on rotated box) and forwarded `width=int(width), height=int(height)` in both scene factories.

### 19. `binder.py`
- **Root Cause**:
  1. Duplicate top-level import blocks.
  2. `_surface_normal` thresholded 8-bin `grad_side` (`0..7`) at `< 3.5` instead of `< 4.0`.
  3. `emit_pending_edges("v2")` passed only `_higher_dets_this_step` / `_higher_node_ids` to `_run_anti_phase_border_ownership`, `_run_anti_phase_containment`, and `_run_color_opponent_border`, starving them of V1 `Color` / `Position` / `Texture` detections and node IDs.
  4. `_run_anti_phase_containment` sorted `(a_id, b_id) = min(u, v), max(u, v)` on **directed** `EdgeType.OCCLUSION`.
  5. `_run_curvature_sign_continuity` used raw `q[6] in [0..7]` instead of signed `(q[6] - 4.0) / 4.0`.
  6. `_run_concentricity` unpacked 3-tuple `_contour_point_geometry` as 4 values (`ValueError: not enough values to unpack`).
  7. `_run_topology` and `_run_degree` rejected 2-tuple `(x, y)` positions (`len(pa) < 3`).
  8. `_run_orientation_angle_relation` took the linear mean of axial angles across the `0 / pi` seam.
- **Fix**: Deduplicated imports; fixed `q[6] < 4.0`; staged `_v1_dets_this_step` & `_v1_node_ids` in `Binder.step` and merged into `all_dets` / `all_node_ids` for cross-tier `"v2"` channels; preserved directed `(u, v)` in `_run_anti_phase_containment`; centred chirality `(q[6] - 4.0) / 4.0`; unpacked `r, ecc, ang_ext = geom`; accepted `len(pa) >= 2` in `_run_topology` and `_run_degree`; replaced linear angle average with `0.5 * math.atan2(...) % math.pi`.
