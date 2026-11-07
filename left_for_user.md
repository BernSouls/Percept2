# Deferred Architectural & Calibration Decisions (`left_for_user.md`)

All unambiguous mechanical, coordinate, unit, contract, and determinism bugs (`39` defects across `19` modules) have been repaired and verified (`fixes_applied.md`). The items below involve **corpus-fitted parameters, cross-module architectural contracts, or public `EdgeType` enum removals** that intentionally require explicit user sign-off before changing behavior or hashes.

---

## Priority 1: Perceptual Graph Hygiene (Recommended Next Commit)

### 1. Delete the 7 Non-Perceptual Temporal / Statistical `EdgeType`s
- **Channels**: `CO_ACTIVATION`, `TRANSITION`, `CONTRAST`, `PERIODICITY`, `PERSISTENCE`, `SEQUENCE`, `SYNCHRONY` (detailed in `edge_classification.md`).
- **Why Deferred**: Removing members from `core.types.EdgeType` touches external modules outside `/home/user/Percept2/` (`core/types.py`, `core/schema.py`, Tier 4 `Consolidator` / `Builder` / `Prediction` / `DerivedLenses`).
- **Measured Impact**: Eliminates **6,000 cap-saturated noise edges per step** (`PERSISTENCE: 2000`, `SEQUENCE: 2000`, `TRANSITION: 2000` measured at `t=1` on a static 256x256 scene).

### 2. Split Anti-Phase Repulsion (`sign = -1`) Out of `EdgeType.OCCLUSION` into `EdgeType.FIGURAL_SEPARATION`
- **Where**: `binder.py:2215-2358` (`_run_anti_phase_border_ownership`, `_run_anti_phase_containment`), `parameters/_text.py:903` (`ANTIPHASE_ENABLED: bool = False`).
- **Why Deferred**:
  1. `P.ANTIPHASE_ENABLED` is currently `False` by default in `_text.py`.
  2. When enabled, anti-phase separation writes `sign = -1` edges under `EdgeType.OCCLUSION`, colliding on `(src, dst, EdgeType.OCCLUSION)` with positive ordinal-depth `OCCLUSION` edges (`sign = +1`).
- **Decision Needed**: Add `EdgeType.FIGURAL_SEPARATION` (`missing_edges.md` #4) and flip `ANTIPHASE_ENABLED = True` once downstream phase-oscillator / consolidator readers are updated to read `FIGURAL_SEPARATION`.

---

## Priority 2: Pipeline Wiring & Dual-Spine Unification

### 3. Wire `fragment_ownership_records -> build_surface_fragments -> SurfaceFragmentLens` in the Live Pipeline
- **Where**: `surface_fragment_lens.py:283-305`, `observation_2d.py`, and the outer Pipeline driver; plus `SurfaceFragmentLens.positions` (`Dict[int, List[Tuple[float, float]]]`).
- **Finding**:
  1. `Observation2D.from_image` never computes or attaches `surface_fragments`, so `SurfaceFragmentLens.step` returns `[]` (`0` detections) unless the caller manually invokes `fragment_ownership_records` and `build_surface_fragments`.
  2. `SurfaceFragmentLens.positions` stores a **list** of `(x, y)` coordinates per content-hashed `alpha` (because content-identical fragments collapse to one detection), whereas `Binder` expects `anchor_positions[alpha]` to be a single `(x, y)` / `(x, y, z)` tuple!
  3. Zero binder channels in `binder.py` currently consume `d.k == "SurfaceFragment"`, leaving 3D `(slant, tilt, curv_sign)` unused during relational binding (for example, `SURFACE_FOLD` fired `0` times on two 3D boxes!).
- **Decision Needed**:
  - Should `SurfaceFragmentLens` compute fragments lazily from `(PEAK_CHAIN, observation)` when `fragments=None` (mirroring `BorderOwnershipLens` and `SurfaceLens`), and how should multi-instance content alphas expose positions to `SURFACE_CONTINUITY` / `DIHEDRAL_CREASE` (`missing_edges.md` #1–#3)?

### 4. Unify the Dual Contour Spine (`PEAK_CHAIN` [13] vs `ContourLens` [998])
- **Where**: `chain_lens.py` (`PEAK_CHAIN`), `observation_2d.py` (`_extract_association_field_contours`), `binder.py` (`_run_orientation_continuity`), and `v2.py` (`ContourLens`, `JunctionLens`).
- **Finding**: On the 256x256 box-over-box scene, the continuous field + `ChainLens` extracts **13 clean `PEAK_CHAIN`s**, while `ContourLens` over V1 `ORIENTATION_CONTINUITY` emits **998 overlapping `Contour` detections**! Half of V2/V4 (`BorderOwnershipLens`, `TextureBoundaryLens`, `CornerLens`, `SurfaceLens`, `RegionShapeLens`) already migrated to `PEAK_CHAIN`, while the other half (`ContourLens`, `JunctionLens`, `CurveComplexLens`, `CONTOUR_CONTINUITY`, `COCIRCULAR`, `PARALLEL`, `SYMMETRY`) still reads `Contour`.
- **Decision Needed**: Confirm whether `ContourLens` should project directly from `PEAK_CHAIN` (retiring the 998-node V1 segment walk) or remain separate.

### 5. Tighten V4 Polar Detectors (`RADIALITY`, `SPIRALITY`, `CONCENTRICITY`) on Polygonal Scenes
- **Where**: `binder.py:2729-3055` (`_run_concentricity`, `_run_radiality`, `_run_spirality`).
- **Finding**: Two rectilinear boxes produced `345 RADIALITY`, `101 SPIRALITY`, and `36 CONCENTRICITY` edges because 2–3 straight edges meeting at box corners pass pairwise radial/spiral angle gates.
- **Decision Needed**: Approve minimum spoke multiplicity (`>= 4` spokes per hub for `RADIALITY`), nonzero intrinsic contour curvature (`|chirality| > tau` for `SPIRALITY`), and closed-contour circularity gates (`circularity > 0.35` for `CONCENTRICITY`).

### 6. Gate `SURFACE_GROUPING` by Intervening Contour / Ownership Energy
- **Where**: `binder.py:4970-5060` (`_run_surface_grouping`).
- **Finding**: `_run_surface_grouping` emitted `0` edges at `t=0` and `1,015` edges at `t=1` (once rich `Texture` hysteresis fired), linking hypercolumn `Position` nodes by `(Color, Texture)` similarity without checking whether a strong `PEAK_CHAIN` / `BorderOwnership` boundary separates the two patches.
- **Decision Needed**: Gate candidate patch pairs `(a, b)` by line-of-sight contour-field / `PEAK_CHAIN` crossing penalty so two distinct objects of similar material do not merge across an occluding contour.

---

## Priority 3: Corpus-Fitted Calibration Tables & Constants (Untouched)

Per perceptual-first audit discipline, no corpus-fitted JSON tables or empirical constants were altered or deleted without user approval:

1. **`_ownership_weights.json` & `_ownership_calibration.json`** (consumed by `fragment_ownership.py:84-145`):
   - Contains fitted cue weights and isotonic/logistic probability calibration bins for fragment border ownership. Replaceable by closed-form log-likelihood ratios over enclosure, convexity, and T-junction stem geometry if zero-fitted-weights purity is desired.
2. **`_texture_calibration.json`** (consumed by `surface_fragments.py:1533-1640`):
   - Empirical slant/sigma lookup table for `texture_cue_calibrated` foreshortening. Preserve for benchmark reproducibility or replace with analytic isotropic foreshortening $\sigma = \arccos(\lambda_{\min}/\lambda_{\max})$ uncertainty derived from patch Fisher information.
3. **Empirical Clipping & Prior Constants in `parameters/_text.py` & `surface_fragments.py`**:
   - `CONTOUR_PEAK_STRENGTH_CLIP`, `CONTOUR_PEAK_COHERENCE_CLIP` (measured on 16 ETH-80 frames at 256px), `RIM_DEFICIT_*` quantiles, and `_LIGHT_FLOOR_QUANTILE = 0.05` in `surface_fragments.py`.
4. **`contour_field.py` — `_FIELD_FORMULA_VERSION = 7`**:
   - Preserved intact so precomputed field caches (`field_fingerprint()`) remain valid; only small-image (`ph >= H or pw >= W`) padding fallback was added, which is unreachable at standard `>= 32x32` resolutions.
