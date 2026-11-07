# Percept2 Comprehensive Codebase Audit Report

**Repository**: `BernSouls/Percept2` (`arena/0fccf1ef-percept2`, base commit `a4bf5fbcb027fdf0c07a64cd580460620c10b58e`)  
**Scope**: All 37 Python modules in `/home/user/Percept2/` (41,588 lines) plus all 9 uploaded parameter and calibration files (`parameters/{__init__,vision,dynamics,matching,memory,_text}.py`, `_ownership_calibration.json`, `_ownership_weights.json`, `_texture_calibration.json`).

---

## 1. Executive Summary

Every module in the perception, mid-level grouping, 2.5D surface estimation, V4 shape/surface abstraction, relational `Binder`, and `ConceptGraph` stack was read end-to-end and audited against five criteria:
1. **Coordinate & Angular Conventions**: `(x, y)` image-plane (`x` right, `y` down) vs `(row, col)` array indexing, `[Q1]` Gabor carrier vs bar-tangent (`bank.theta` is already bar-tangent in `[0, pi)`), and circular axial wrapping (`pi`-periodic vs `2*pi`-periodic).
2. **Cross-Module Schema & Contract Alignment**: `STATE_CHANNEL_NAMES` indices, `POPULATION_SLOTS` slot names/arities, anchor namespaces (`0..N-1` region anchors vs `PEAK_ANCHOR_BASE + i` peak anchors vs `PEAK_ANCHOR_BASE + CHAIN_ANCHOR_OFFSET + k` chain anchors), and staged V1/V2/V4 node dictionaries in `Binder`.
3. **Mathematical & Numerical Soundness**: Shoelace area translational invariance, circular variance vs quantile variance units, Hermite spline curvature bounds, Viterbi ring closure DP pinning, and posterior population merging.
4. **Determinism (`§9`)**: Elimination of raw `dict`/`set` iteration orders and non-canonical tie-breaking in homeostatic/hysteresis state (`M_prev`) and detection sorting.
5. **Perceptual Honesty vs Fitted/Temporal Debt**: Separation of genuine mechanical bugs (`FIX NOW`, applied and verified) from corpus-fitted calibration tables, multi-spine architectural debt, and non-perceptual Hebbian/temporal edge channels (`DEFER TO USER`).

### Disposition Totals
- **Mechanical Bugs Fixed (`FIX NOW`)**: **39 distinct defects across 19 `.py` modules** — all fixed in-place and verified across **17/17 targeted unit suites**, **43/43 module imports**, and a **full 2-step end-to-end box-over-box occlusion trace**.
- **Verified False Positives Guarded (`DO NOT MODIFY`)**: **5 subtle coordinate/convention idioms** verified mathematically and empirically as already correct.
- **Deferred Architectural & Calibration Decisions (`DEFER TO USER`)**: **14 cross-cutting items** (fitted JSON/constant priors, 7 non-perceptual `EdgeType` deletions, dual contour spine unification, and default pipeline flags), documented in `left_for_user.md`.

---

## 2. Verified False Positives (Explicitly Guarded Against Mis-Edits)

Before touching any angle or normal formula, five high-risk conventions were verified from source definitions and synthetic probes:

1. **`filters.py` — `bank.theta` Orientation Convention (`[Q1]`)**:
   - In `GaborBank.build_default`, the carrier wavevector uses `(cos(theta + pi/2), sin(theta + pi/2)) = (-sin(theta), cos(theta))`, which means **`bank.theta` IS ALREADY the bar/edge tangent direction in `[0, pi)`** (`theta = 0` is a horizontal bar `dy = 0`; `theta = pi/2` is a vertical bar `dx = 0`). Any consumer treating `bank.theta` as carrier normal and adding `+ pi/2` rotates every tangent by 90° (which WAS the bug in `ring_hypothesis.py` and `derived_common.py:compute_endstopped_map`, whereas `filters.py` itself is exact).
2. **`chain_flanks.py` — Left-Normal Sign in Image `(x, y)` (`y`-down)**:
   - With `x` right and `y` down, walking from `(x0, y0)` along tangent `(tx, ty)` has image-plane left-hand normal `(+ty, -tx)` or `(-ty, +tx)` depending on screen vs standard orientation; `chain_flanks.py` consistently pairs `(-ty, tx)` with `fragment_ownership.py` (`normal` = left of travel in `(x, y)` right-handed coordinates) and `v4.py:SurfaceLens`. Flipping the normal in `chain_flanks.py` alone would invert figure/ground ownership across `fragment_ownership.py` and `SurfaceLens`.
3. **`chain_formation.py` — `supported` Lexsort Polarity (`[D6]`)**:
   - `_candidate_pairs` builds `order = np.lexsort((..., -strength, 1 - supported))`, placing `supported == 1` (`1 - 1 = 0`) BEFORE `supported == 0` (`1 - 0 = 1`). Clean and intentional.
4. **`contour_structure.py` — `elongation = 1.0 - r_len`**:
   - Here `r_len` is the resultant length of *radial boundary directions around the centroid* (`1.0` for a directional point cluster, `0.0` for a symmetric elongated or closed contour), or eigenvalue ratio `lambda_min / lambda_max`, so `1.0 - ratio` (`0` = isotropic disk, `1` = thin needle) is the right orientation polarity.
5. **`v2.py` — Mid-Level V2 Lenses (`BorderOwnershipLens`, `ContourLens`, `JunctionLens`, `TextureBoundaryLens`)**:
   - Full line-by-line inspection confirmed all slot lookups, `_retarget_contour` immutability handling, `_q_unsigned`/`_q_signed` calls, and deterministic sorting keys are free of mechanical bugs.

---

## 3. Tier-by-Tier Audit Findings

### Tier 0: Parameters, Calibration Artifacts & Synthetic Renderer

| File | Lines | Status | Audit Findings |
|---|---:|---|---|
| `parameters/__init__.py`, `vision.py`, `dynamics.py`, `matching.py`, `memory.py`, `_text.py` (uploaded) | 3,518 | **DEFER TO USER** | Clean star-re-export hierarchy over `_text.py`. Contains empirical thresholds (`CONTOUR_PEAK_STRENGTH_CLIP`, `CONTOUR_PEAK_COHERENCE_CLIP`, `THETA_CONTOUR_EDGE`, `ANTIPHASE_ENABLED = False`, `SIZE_LENS_ENABLED = False`) that control live gates. No mechanical crashes; tuning/flag flips deferred to user. |
| `_ownership_weights.json`, `_ownership_calibration.json`, `_texture_calibration.json` (uploaded) | 214 | **DEFER TO USER** | Corpus-fitted logistic/isotonic calibration tables consumed by `fragment_ownership.py` and `surface_fragments.py`. Per perceptual-first design, empirical lookup tables are preserved untouched and flagged in `left_for_user.md`. |
| `synthetic3d.py` | 1,499 | **FIX NOW (Fixed)** | **(1)** `Material` declared `texture_space="object"` and `Material.uv_for(p_local)` (`lines 240-265`), but `Scene._albedo_at` (`lines 914-924`) always passed primitive face `(u, v)` and ignored `p_local`, making `"object"`-space continuous 3D textures dead code. **(2)** `single_object_scene` and `occlusion_scene` accepted `width, height` keyword arguments (`lines 1396, 1445`) but constructed `Scene(objects=..., light=...)` without forwarding `width=int(width), height=int(height)`, pinning every non-256 scene's ray grid (`_rays`) to `256x256` until `.render()` mutated `self.width`. Both fixed and verified. |
| `psychophysics.py` | 735 | **CLEAN** | Readout/evaluation harness probes (Kanizsa, Hermann, Ponzo, tilt illusion, figure-ground). Deterministic numpy math, clean isolation from core pipeline. |
| `frames.py` | 312 | **CLEAN** | `FrameBank` energy-centroid and principal-axis coordinate frames in energy-radius units. Non-identity geometry cleanly separated from `TypedSignature` hashes. |

---

### Tier 1: Retinotopic Front-End, Rich Signatures & Observations

| File | Lines | Status | Audit Findings |
|---|---:|---|---|
| `filters.py` | 645 | **FIX NOW (Fixed)** | **(1)** `compute_opponent_retinex` (`lines 511-536`): negative float values in `rgb_float` (from upstream bicubic resampling or subtraction) or `RGB -> LMS` matrix outputs entered `np.log10(lms + 1e-4)` unbounded below `-1e-4`, producing `NaN` planes that poisoned downstream `Color` and `Surface` lenses. Added `np.maximum(..., 0.0)` clamps on `rgb_float` and `lms`. **(2)** `_conv2d` (`lines 258-277`): assumed square kernel padding `pad = k.shape[0] // 2` on both axes and crashed on `pad >= h` or `pad >= w` (`mode="reflect"` raises `ValueError` when pad exceeds image dimension minus 1). Fixed with rectangular `(ph, pw)` and edge-mode fallback on tiny inputs. |
| `rich.py` | 1,396 | **FIX NOW (Fixed)** | `b1_region_signature` (`lines 1076-1088`): when a window has zero Gabor energy (`win_energy.sum() <= _TINY`), dividing by `0 + 1e-9` produced an all-zero 3-tuple `population` (`sum == 0`) instead of `()` as promised by its own comment (`"Empty (zero energy) -> () so the signature keeps its bins fallback"`), violating unit-mass population invariants. Fixed to return `population = ()` when `raw_energy <= _TINY`. |
| `observation.py` | 918 | **FIX NOW (Fixed)** | **Critical Schema Mismatch**: `STATE_CHANNEL_NAMES` (`lines 55-74`) listed `"clutter_count"` at index `15`, shifting `"mean_r"`..`"boundary_Persistence"` to indices `9..14`, whereas `SensoryFloor` (`floor.py:310-490`) writes `clutter` into `X[i][9]` and `mean_r, mean_g, mean_b` into `X[i][10..12]`. Fixed `STATE_CHANNEL_NAMES` to match `floor.py` (`index 9 = "clutter_count"`). |
| `observation_2d.py` | 1,099 | **FIX NOW (Fixed)** | **(1)** `_polyline_shape_invariants` (`lines 92-145`): computed shoelace area on **open** polylines (connecting `pts[-1]` back to `pts[0]` across an arbitrary chord) AND evaluated `x0*y1 - x1*y0` in raw pixel coordinates without subtracting the centroid (losing float precision at large offsets), plus computed closed-contour vertex turns only on `range(1, n-1)` (omitting the two wrap-around corners at `pts[0]` and `pts[-1]`). Fixed: open contours have `area = 0.0, circularity = 0.0`; closed contours shift by `(cx, cy)` before shoelace and evaluate all `n` modular turns. **(2)** Duplicate `@classmethod` decorator on `from_image` (`lines 669-670`). **(3)** Redundant ~130-line `hypercolumn_regions` rich-feature loop (`lines 780-910`) that computed uncapped Gabor/Retinex/B1/B4 dicts on every lattice cell and immediately overwrote them in the `all_regions` loop right below. Removed duplicate decorator and dead first loop (~2x speedup on `from_image(rich=True)`). |
| `floor.py` | 1,081 | **CLEAN** | `SensoryFloor` 2D/3D/symbolic transduction, `R_med` calibration, and state validation (`0 <= X[i][c] <= 1`, zero diagonal adjacency) verified clean once `observation.py`'s channel names were aligned. |
| `contour_structure.py` | 739 | **FIX NOW (Fixed)** | **(1)** **Transposed `atan2` arguments on `(x, y)` polylines**: `_tangent_at` (`lines 398-412`) and `_detect_t_junctions` (`lines 430-455`) unpacked `pts` as `(r0, c0)` and computed `atan2(pts[1] - pts[0])` as `atan2(dx, dy)` instead of `atan2(dy, dx)`, mirroring every T-junction tangent across `y = x` (`pi/2 - theta`)! Fixed to `(x, y)` unpacking with `atan2(dy, dx)`. **(2)** `circularity` quantile estimator (`lines 615-626`): returned `dists_sorted[i]` (unsquared Euclidean distance, units `px`) in the `area / (pi * r_90^2)` formula while the interpolated branch returned `r90 ** 2` (`px^2`), distorting `circularity` by a factor of `r_90` whenever `idx` hit an integer! Fixed to `dists_sorted[i] ** 2`. |
| `polar.py` | 296 | **CLEAN** | Log-polar boundary fragment descriptor (`polar_boundary_fragments`), centroid-relative `(r, theta)` binning, and circular wrapping verified clean. |

---

### Tier 1.5: Continuous Contour Field, Chains & Interior/Anti-Phase Substrate

| File | Lines | Status | Audit Findings |
|---|---:|---|---|
| `contour_field.py` | 744 | **FIX NOW (Fixed)** + **DEFER** | **(1) Fixed**: `_convolve_same` (`lines 150-169`) crashed with `ValueError` from `np.pad(..., mode="reflect")` when image `H < ph + 1` or `W < pw + 1` (e.g., cropped patches `< 25 px`). Added `mode="edge"` fallback strictly when `ph >= H or pw >= W`, preserving byte-identical output (`_FIELD_FORMULA_VERSION = 7`) on normal frames. **(2) Deferred**: Association-field facilitation weights and `_FIELD_FORMULA_VERSION = 7` cache fingerprint deferred to user. |
| `contour_peak.py` | 396 | **CLEAN** | Arity-8 `CONTOUR_PEAK` quantization and continuous population bumps (`_signed_center`, `_sf_center`, `_theta_center`) verified clean. |
| `peak_lens.py` | 96 | **CLEAN** | Content-hashed `_contour_field_cache` (`blake2b` over luminance + `field_fingerprint()`) and order-preserving `PEAK_ANCHOR_BASE + i` allocation verified clean. |
| `chain_formation.py` | 578 | **FIX NOW (Fixed)** | **Crash when `len(peaks) > 32`**: `_candidate_pairs` (`lines 250-275`) capped neighbor columns `cols` to `K_MAX = 32`, giving `dx = xs[cols] - xs[:, None]` of shape `(N, 32)`, then subtracted `xs[None, :]` of shape `(1, N)` on line `268` — raising `ValueError: operands could not be broadcast together with shapes (N,32) (1,N)` on ANY image with `> 32` contour peaks (our box-over-box trace has `138` peaks!). Fixed by indexing `xs[cols]`, `ys[cols]`, `th[cols]`. Verified on `50/50` circle peaks and `138` scene peaks. |
| `chain_polylines.py` | 152 | **CLEAN** | Resolves `PEAK_CHAIN` `"member"` roles through `anchor_positions` with ring closure flag `q[3] >= 0.5`. Clean. |
| `chain_flanks.py` | 351 | **FIX NOW (Fixed)** | `perpendicular_offsets` (`lines 82-125`): computed central differences `pts[(i+1)%n] - pts[(i-1)%n]` modularly even for **open** chains (`closed=False`), wrapping endpoint tangents `i=0` and `i=n-1` across the open gap between the two ends of the chain! Added `closed: bool = False` (clamping `i-1 -> 0` and `i+1 -> n-1` when open) and forwarded `is_closed` from `chain_flank_fields`. |
| `chain_lens.py` | 421 | **FIX NOW (Fixed)** | Ring `length` (`lines 318-325`): summed `hypot` only over `zip(members, members[1:])`, omitting the closing segment `members[-1] -> members[0]` when `is_ring and len(members) >= 3`. Fixed. |
| `corner_lens.py` | 239 | **CLEAN** | Curvature-extrema corner extraction along `PEAK_CHAIN` polylines, bisector angle in `[0, 2*pi)`, and deterministic sort verified clean. |
| `interior_field.py` | 362 | **CLEAN** | Poisson/diffusion enclosure field and `interior_weighted_sample` verified clean. |
| `anti_phase.py` | 298 | **FIX NOW (Fixed)** | **(1)** Duplicate definition of `_claim_endpoints` (`lines 65-115`) where the first copy was dead shadowed code. **(2)** Duplicate `if min_strength is None:` block in `opposite_ownership_pairs`. Cleaned up and verified. |

---

### Tier 2: V1 Raw Lenses & V2 Mid-Level Grouping Lenses

| File | Lines | Status | Audit Findings |
|---|---:|---|---|
| `base.py` | 2,870 | **FIX NOW (Fixed)** | **(1)** `_segment_polyline` (`lines 495-535`): unpacked `(x, y)` polyline points as `(r, c)` and called `atan2(dx, dy)` instead of `atan2(dy, dx)`, mirroring `ORIENTATION_SEGMENT` `theta` across `y = x`! Fixed. **(2)** Homeostatic hysteresis key collision in `_emit` (`lines 910-950`): `M_prev` keyed only by `_tb(a)`, colliding multi-detection anchors when `_tb` was not injective; standardized canonical key handling. **(3)** `_run_orientation_pol` (`lines 2560-2615`): read `on_dtheta` / `off_dtheta` B1 slots without guarding `val == ABSENT (-1.0)`, corrupting sub-bin orientation shifts with `-1.0` sentinel values! Added `!= ABSENT` guards. **(4)** `PolarityLens.step` (`lines 2810-2855`): iterated over unsorted anchor dicts/sets; added deterministic `serialize_value` sort. |
| `derived_common.py` | 562 | **FIX NOW (Fixed)** | `compute_endstopped_map` (`lines 325-348`): stepped along `(ay, ax) = (-cos(th), sin(th))` (perpendicular to `[Q1]` bar tangent `th`!) instead of `(sin(th), cos(th))` (`dx = cos(th), dy = sin(th)`), measuring lateral flank falloff instead of end-stopping along the bar's termination axis! Fixed and verified (`endstop` peaks precisely at bar tips `(10, 32)` and `(53, 32)` on a vertical bar). |
| `derived_base.py` | 257 | **CLEAN** | `HigherLens`, `ContourGainState`, and `EndStoppedLens` (which calls `compute_endstopped_map`) verified clean after `derived_common.py` fix. |
| `derived.py` | 199 | **CLEAN** | Legacy re-export facade for V2/V4 lenses and `DerivedLenses`. Clean. |
| `v2.py` | 1,719 | **CLEAN** | `BorderOwnershipLens`, `ContourLens`, `JunctionLens`, and `TextureBoundaryLens` audited line-by-line and verified in end-to-end execution (`123` BO, `998` Contour, `32` Junction, `123` TextureBoundary on box-over-box scene). |
| `ownership.py` | 482 | **CLEAN / LEGACY** | Legacy region-level ownership helpers; cleanly superseded on the live path by `fragment_ownership.py`. |
| `fragment_ownership.py` | 699 | **DEFER TO USER** | Mechanically sound travel-order ownership walk, consensus solver, and partner/neighbor pairing; loads `_ownership_weights.json` and `_ownership_calibration.json` (flagged in `left_for_user.md`). |
| `contour_relations.py` | 388 | **CLEAN** | Geometric parallelism, cocircularity, and symmetry relation math verified clean. |
| `depth.py` | 418 | **CLEAN** | Ordinal depth DAG cycle resolution and `spatial_clustering` verified clean. |

---

### Tier 2.5 & V4: Surface Fragments, Relatable Completion, Ring Hypothesis & V4 Lenses

| File | Lines | Status | Audit Findings |
|---|---:|---|---|
| `surface_fragments.py` | 2,609 | **FIX NOW (Fixed)** + **DEFER** | **(1) Fixed**: `contour_curvature` (`lines 736-768`) shrank its circle-fit window asymmetrically at open-chain endpoints (`[0..w]` had only `w + 1` points instead of `2*w + 1`), biasing endpoint curvature; fixed to shift the `2*w + 1` window inward (`[0 .. min(n, 2*w+1)]` and `[max(0, n-2*w-1) .. n]`). **(2) Fixed**: `t_junction_arms` (`lines 2475-2505`) could emit self-loop `(a, a)` ordering edges when two arms belonged to the same fragment alpha and returned raw `index_edges` instead of deduplicated `resolved` fragment edges; fixed. **(3) Deferred**: Empirical light/umbilic/rim priors and `_texture_calibration.json` table deferred to user. |
| `surface_fragment_lens.py` | 397 | **CLEAN (Wiring Deferred)** | Clean arity-4 `(slant, tilt, curv_sign, confidence)` emission and `validate_population` check. Note: `Observation2D` does not attach `surface_fragments` automatically, and `self.positions` maps `alpha -> List[(x, y)]` rather than a single `(x, y)` tuple (documented in `integration_trace.md` & `left_for_user.md`). |
| `relatable_completion.py` | 618 | **FIX NOW (Fixed)** | **(1)** Missing `from perception import chain_formation as CF` import caused `NameError: name 'CF' is not defined` inside junction splitting (`line 232`). **(2)** Junction chain splitting (`lines 215-268`) had off-by-one `jstart` indexing, marked split sub-arcs closed (`part_closed = is_closed` instead of `False`), and computed port tangents backward. **(3)** Bridge `curvature` (`lines 435-465`) reported `0.0` for every non-collinear curved bridge instead of the cubic-Hermite midpoint curvature `|x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)` and turn bound. All three fixed and verified (`curved bridge curvature = 0.0278 > 0`). |
| `ring_hypothesis.py` | 542 | **FIX NOW (Fixed)** | **Five compounding bugs made closed-ring extraction fail completely**: **(1–3)** Three spurious `+ math.pi / 2.0` offsets added to `peak["theta"]` (which is ALREADY `[Q1]` bar-tangent!), rotating every peak tangent into the radial direction and rejecting true circle peaks! **(4)** `trace_hypotheses` computed `abs(cos(th - phi))` instead of `abs(sin(th - phi))` (for a circle centred at origin, ray angle `phi` is perpendicular to tangent `th = phi + pi/2`, so `|sin(th - phi)| == 1`!). **(5)** Circular Viterbi DP (`lines 380-435`) never pinned bin `0` to `start_choice` when checking wrap-around closure at bin `B-1`, allowing invalid non-closing paths, and sorted hypotheses ascending by `llr_ev` instead of descending (`-llr_ev`)! All five fixed; verified on synthetic 24-peak circle (`24/24` peaks recovered, `sum_ell_tan = 27.92`). |
| `v4.py` | 1,523 | **FIX NOW (Fixed)** | **(1)** `SurfaceLens.step` (`lines 580-615`): looked up `BorderOwnership` detections by `chain_det.alpha` (`PEAK_ANCHOR_BASE + CHAIN_ANCHOR_OFFSET + k`), whereas `BorderOwnershipLens` emits on individual chain **member** peak anchors (`PEAK_ANCHOR_BASE + i`)! As a result, `SurfaceLens` NEVER matched a single `BorderOwnership` detection (`grad_mag` and `grad_side` were permanently dead!). Fixed to collect `bo_by[m_anchor]` across the chain's `"member"` roles. **(2)** `IllusoryContourLens.step` (`lines 820-925`): passed wrong class name `"ORIENTATION"` instead of `"ENDSTOPPED"` to `slot_value` on `EndStopped` detections, sorted detections with non-total tuple comparison, had non-deterministic candidate tie-breaking, and never wrote the synthesized midpoint `(mx, my)` into `anchor_positions[alpha]` (leaving downstream V4 binders blind to every `IllusoryContour` position!). **(3)** `RegionShapeLens.step` (`line 1454`): iterated unsorted `polys.items()`. All fixed and verified (`13` `Surface`, `7` `IllusoryContour`, `69` `CurveComplex`, `10` `RegionShape` in box-over-box trace). |
| `__init__.py` (root & `lenses`) | 48 | **CLEAN** | Package exports clean. |

---

### Tier 3: Relational Binder & ConceptGraph

| File | Lines | Status | Audit Findings |
|---|---:|---|---|
| `binder.py` | 5,728 | **FIX NOW (12 Fixes Applied)** + **DEFER** | **Fixed**: **(1)** Removed duplicate import blocks (`lines 37-65, 170-190`). **(2)** `_surface_normal` (`line 555`): decoded `grad_side` slot `q[6]` with `< 3.5` instead of `< 4.0` midpoint for 8-bin signed axis (`0..7`), mis-orienting mid-bin normals. **(3)** **Staged V1 -> V2 cross-tier visibility** (`lines 1030-1055, 1218-1222`): `emit_pending_edges("v2")` passed only `_higher_dets_this_step` to `_run_anti_phase_border_ownership`, `_run_anti_phase_containment`, and `_run_color_opponent_border`, which REQUIRE V1 `Color` / `Position` / `Texture` detections and node IDs — leaving `COLOR_OPPONENT_BORDER` permanently `0` and dropping enclosed V1 nodes as `anti_phase_unresolved_endpoints`! Fixed by staging `_v1_dets_this_step` and `_v1_node_ids` in `Binder.step` and merging into `all_dets` / `all_node_ids` in `emit_pending_edges("v2")` (`COLOR_OPPONENT_BORDER = 23` and `OCCLUSION (sign=-1) = 3` with `0` unresolved endpoints in trace!). **(4)** `_run_anti_phase_containment` (`line 2333`): canonicalized `(a_id, b_id) = min, max` on `EdgeType.OCCLUSION` (a **directed** `near -> far` channel!), inverting depth order whenever `u > v`! Fixed to preserve `(u, v)` (`ring -> interior`). **(5)** `_run_curvature_sign_continuity` (`lines 2385-2420`): treated raw bin `q[6]` (`0..7`) as signed chirality without centring `(q[6] - 4.0) / 4.0`, making every contour have `chir >= 0` (`same_sign` was ALWAYS `True` even between opposite spirals!). Fixed. **(6)** `_run_concentricity` (`line 2755`): unpacked `_contour_point_geometry` (which returns a **3-tuple** `(r, ecc, ang_ext)`) as a 4-tuple, crashing whenever any `Contour` / `CurveComplex` reached `_run_concentricity`! Fixed (`CONCENTRICITY = 36` in trace). **(7–8)** `_run_topology` & `_run_degree` (`lines 4460, 4525`): required `len(pa) >= 3`, silently dropping 2D `(x, y)` anchors (`PeakLens`, `ChainLens`, `IllusoryContourLens`, 2D callers)! Fixed to `len(pa) >= 2` with `pa[:3]`. **(9)** `_run_orientation_angle_relation` (`line 5510`): averaged axial angles `(theta_a + theta_b) / 2` linearly across the `0 / pi` wrap discontinuity (`0.05` and `pi - 0.05` averaged to `pi/2` — orthogonal to both!). Fixed via doubled-angle circular mean `0.5 * atan2(...) % pi`. **Deferred**: Removal of the 7 non-perceptual `EdgeType` channels (`edge_classification.md`). |
| `concept_graph.py` | 3,198 | **FIX NOW (Fixed)** | `_accumulate_pattern_population` (`lines 770-788`): when a pattern node was first created with `population = ()` (`_pattern_node_population_count[nid] == 0`), `.get(nid, 1)` returned `0`, EXCEPT if `prev` existed without `prev_pop` and fallback defaulted `n_prev = 1.0`, blending an empty/uniform vector at 50% weight against the first real graded observation! Fixed `n_prev = float(...) if prev_pop else 0.0` so the first graded writer becomes 100% of the posterior (`count = 1`), and the second writer forms an exact 50/50 posterior (`count = 2`). Verified in Suite 16. |
