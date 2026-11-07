# Missing Perceptual Relational Edges (`missing_edges.md`)

While `Percept2` already defines 35 perceptual `EdgeType` channels (once the 7 temporal/statistical channels in `edge_classification.md` are removed), our Phase 1–4 audit and live box-over-box occlusion trace exposed **7 critical perceptual relations that cannot be expressed cleanly by the existing edge vocabulary**.

Every proposed edge below is strictly **perceptual and geometric/photometric** — computed closed-form from observables `Percept2` ALREADY measures (`SurfaceFragment` `(slant, tilt, curv_sign)`, `PEAK_CHAIN` polylines, `chain_flanks` opponent/luminance ratios, `BorderOwnership`, and `relatable_completion`), with zero learned weights and explicit `PopAxis` unit-table specifications.

---

## Summary of Proposed Perceptual Edge Types

| # | Proposed `EdgeType` | Dir | Stage | Endpoints (`src -> dst`) | Missing Perceptual Capability Closed |
|---:|---|---|---|---|---|
| 1 | `SURFACE_CONTINUITY` | Undir | `v2` / `v4` | `SurfaceFragment <-> SurfaceFragment` | Smooth 2.5D `(slant, tilt)` surface continuation across adjacent contour/interior fragments |
| 2 | `DIHEDRAL_CREASE` | Undir | `v4` | `SurfaceFragment <-> SurfaceFragment` | Intrinsic 3D roof (convex `+1`) vs valley (concave `-1`) crease between faces of ONE object (e.g. box edges!) |
| 3 | `COPLANARITY` | Undir | `v4` | `SurfaceFragment <-> SurfaceFragment` | Non-adjacent surface fragments lying on the **same 3D plane** across an occluder (amodal surface completion) |
| 4 | `FIGURAL_SEPARATION` | Dir | `v2` | `Occluder/Figure -> Ground` | Explicit anti-phase figure/ground separation, decoupling `sign=-1` repulsion from `EdgeType.OCCLUSION` |
| 5 | `ILLUMINATION_BOUNDARY` | Dir | `v2` | `Lit Flank -> Shadowed Flank` | Separates cast-shadow / shading edges (chromaticity-invariant luminance ratio) from reflectance/object boundaries |
| 6 | `CONCAVE_MINIMA_JOINT` | Dir | `lo_pit` | `Body Part -> Attached Limb/Part` | Hoffman & Richards minima-rule part boundary & attachment joint at negative-curvature cusps / concave creases |
| 7 | `PHYSICAL_SUPPORT` | Dir | `lo_pit` | `Supporting Surface -> Supported Object` | Gravity-aligned contact + upward surface-normal support relation (`n . g < 0`), distinct from generic `CONTACT` |

---

## Detailed Architectural Specifications

### 1. `SURFACE_CONTINUITY` (Smooth 2.5D Surface Integration)
- **Why It Is Missing**: `SURFACE_GROUPING` (`binder.py:4970`) groups V1 `Position` anchors solely by 2D `Color` and `Texture` bin similarity, completely ignoring 3D surface orientation! Meanwhile `SurfaceFragmentLens` emits `SurfaceFragment` detections with graded `(slant, tilt, curv_sign, confidence)` populations, **but zero binder channels in `binder.py` read `SurfaceFragment`!** On a curved cylinder or shaded sphere, color luminance changes continuously across Lambertian shading, causing 2D `Color`-based grouping to split a single smooth 3D surface while missing its smooth normal field.
- **Endpoints**: `SurfaceFragment <-> SurfaceFragment` (undirected; canonical `u < v`).
- **Binder Stage**: `emit_pending_edges("v2")` (immediately after `SurfaceFragmentLens`).
- **Observables & Closed-Form Signal**:
  - Convert `(slant_a, tilt_a)` and `(slant_b, tilt_b)` to unit 3D normals:
    $$\mathbf{n} = (\sin\sigma \cos\tau,\; \sin\sigma \sin\tau,\; \cos\sigma)$$
  - Geodesic normal angle $\Delta\psi = \arccos(\operatorname{clamp}(\mathbf{n}_a \cdot \mathbf{n}_b, -1, 1)) \in [0, \pi]$.
  - First-order spherical/umbilic compatibility along chord $\mathbf{d} = \mathbf{p}_b - \mathbf{p}_a$: check that $(\mathbf{n}_b - \mathbf{n}_a) \parallel \mathbf{d}$ up to `curv_sign` tolerance.
  - Signal: $s = \exp(-\|\mathbf{d}\| / (3 R_{\text{scale}})) \cdot \exp(-(\Delta\psi / \sigma_\psi)^2) \cdot \min(z_a, z_b)$, gated by absence of an intervening `OCCLUSION` edge.
- **`EdgeParams` Schema (`EDGE_SURFACE_CONTINUITY`)**:
  - `normal_angle`: `PopAxis("normal_angle", n_bins=8, lo=0.0, hi=math.pi/2, circular=False)` (radians)
  - `curv_compat`: `PopAxis("curv_compat", n_bins=8, lo=-1.0, hi=1.0, circular=False)`
  - `dist`: `PopAxis("dist", n_bins=8, lo=0.0, hi=6.0, circular=False)` (in `extraction_scale` units)

---

### 2. `DIHEDRAL_CREASE` (Signed Convex Roof vs Concave Valley Fold)
- **Why It Is Missing**: On our box-over-box scene, each box has 3 visible planar faces meeting along **internal convex creases** (`Y`-junctions). Currently `SURFACE_FOLD` (`binder.py:3060`)fired **0 times** in our trace because it reads 2D `Surface` patches' `grad_side` rather than 3D `SurfaceFragment` normals, and it has no slot for **convex (`+1`, roof/ridge) vs concave (`-1`, valley/reflex) dihedral sign**! Distinguishing convex self-folds (one rigid polyhedron) from concave creases (two touching objects or articulated part joints, per Hoffman & Richards) is foundational to 3D shape perception.
- **Endpoints**: `SurfaceFragment <-> SurfaceFragment` (undirected) sharing a `PEAK_CHAIN` / `Contour` crease boundary.
- **Binder Stage**: `emit_pending_edges("v4")`.
- **Observables & Closed-Form Signal**:
  - Let $\mathbf{p}_a, \mathbf{p}_b$ be the centroids of two `SurfaceFragment`s on opposite flanks of a shared internal chain, with unit normals $\mathbf{n}_a, \mathbf{n}_b$ and unit vector $\hat{\mathbf{u}}_{ab} = (\mathbf{p}_b - \mathbf{p}_a) / \|\mathbf{p}_b - \mathbf{p}_a\|$.
  - Dihedral angle: $\theta_{\text{dih}} = \arccos(\operatorname{clamp}(\mathbf{n}_a \cdot \mathbf{n}_b, -1, 1)) \in [\theta_{\min}, \pi)$.
  - **Convexity index**:
    $$\chi = \frac{(\mathbf{n}_b - \mathbf{n}_a) \cdot \hat{\mathbf{u}}_{ab}}{2 \sin(\theta_{\text{dih}} / 2)} \in [-1, +1]$$
    - $\chi > 0$ (`sign = +1`): normals splay outward away from the crease $\implies$ **convex roof edge** (single rigid box!).
    - $\chi < 0$ (`sign = -1`): normals point inward toward each other $\implies$ **concave valley / part-attachment crease**.
  - Additionally verify `BorderOwnership` neutrality or `texture_space="object"` flank continuity (`chain_flanks.py`) to separate an internal crease from an occluding rim.
- **`EdgeParams` Schema (`EDGE_DIHEDRAL_CREASE`)**:
  - `dihedral_angle`: `PopAxis("dihedral_angle", n_bins=8, lo=0.0, hi=math.pi, circular=False)`
  - `convexity`: `PopAxis("convexity", n_bins=8, lo=-1.0, hi=1.0, circular=False)`
  - `shared_arc`: `PopAxis("shared_arc", n_bins=8, lo=0.0, hi=4.0, circular=False)`

---

### 3. `COPLANARITY` (Amodal 3D Surface Plane Completion)
- **Why It Is Missing**: `AMODAL_LINK` links 1D occluded *contours* (`relatable_completion.py`), but when a foreground box partially splits a background box face into two disconnected regions (or indents a face), 1D contour continuation alone cannot bind the two visible surface remnants when their outer corners are occluded. Human vision completes **coplanar 3D surfaces** behind occluders (`Nakayama, Shimojo & Silverman`).
- **Endpoints**: `SurfaceFragment <-> SurfaceFragment` (undirected).
- **Binder Stage**: `emit_pending_edges("v4")`.
- **Observables & Closed-Form Signal**:
  - Normal parallelism: $\Delta\psi = \arccos(\operatorname{clamp}(\mathbf{n}_a \cdot \mathbf{n}_b, -1, 1)) < \tau_\psi$ (e.g. $15^\circ$).
  - **3D Plane-Offset Residual**: Using orthographic depth slope $(\partial z/\partial x, \partial z/\partial y) = (-n_x / n_z, -n_y / n_z)$ and ordinal depth layer agreement (`depth_layer_a == depth_layer_b` from `assign_depth_layers`), verify that an intervening foreground region (`depth_layer < depth_layer_a`) lies along segment $\overline{\mathbf{p}_a \mathbf{p}_b}$ and material reflectance (`Color` chromaticity `(rg, by)` + `Texture` `(roughness, isotropy)`) matches.
- **`EdgeParams` Schema (`EDGE_COPLANARITY`)**:
  - `normal_err`: `PopAxis("normal_err", n_bins=8, lo=0.0, hi=math.pi/4, circular=False)`
  - `material_sim`: `PopAxis("material_sim", n_bins=8, lo=0.0, hi=1.0, circular=False)`
  - `gap_w`: `PopAxis("gap_w", n_bins=8, lo=0.0, hi=8.0, circular=False)`

---

### 4. `FIGURAL_SEPARATION` (Decoupled Anti-Phase Figure/Ground Repulsion)
- **Why It Is Missing**: Currently `_run_anti_phase_border_ownership` and `_run_anti_phase_containment` (`binder.py:2215-2358`) write `sign=-1` repulsive coupling edges into **`EdgeType.OCCLUSION`**, which is ALSO written with `sign=+1` by `_run_occlusion` (`binder.py:5130`). Because `ConceptGraph.edges` is keyed by `(src, dst, EdgeType)`, a positive ordinal-depth `OCCLUSION` edge and a negative oscillator-separation `OCCLUSION` edge between the same node pair **collide on the exact same key**, merging opposite-sign `EdgeParams` into one `EdgeParamsMixture`!
- **Endpoints**: `Figure Node -> Ground Node` (directed: figure/owner -> ground/background).
- **Binder Stage**: `emit_pending_edges("v2")`.
- **Observables & Closed-Form Signal**:
  - Directly consumes `anti_phase.opposite_ownership_pairs` (source `0`: across-chain opposite flank ownership) and `anti_phase.containment_pairs` (source `1`: closed silhouette ring vs interior background).
  - Always carries `sign = -1` for downstream phase-oscillator desynchronization while leaving `EdgeType.OCCLUSION` purely `sign = +1` ordinal depth (`near -> far`).
- **`EdgeParams` Schema (`EDGE_FIGURAL_SEPARATION`)**:
  - `strength`: `PopAxis("strength", n_bins=8, lo=0.0, hi=1.0, circular=False)`
  - `source`: `PopAxis("source", n_bins=3, lo=0.0, hi=2.0, circular=False)` (`0` = BO flank, `1` = ring containment, `2` = motion shear)
  - `sep_w`: `PopAxis("sep_w", n_bins=8, lo=0.0, hi=4.0, circular=False)`

---

### 5. `ILLUMINATION_BOUNDARY` (Cast Shadow / Shading vs Reflectance Edge)
- **Why It Is Missing**: Every luminance step in an image is currently treated by `PeakLens` -> `ChainLens` -> `BorderOwnershipLens` as a candidate object/material boundary. A sharp cast shadow across a floor or box face produces strong `CONTOUR_PEAK` chains and `BorderOwnership` claims, causing false figure segmentation. Yet `filters.py` (`compute_opponent_retinex`) and `chain_flanks.py` (`planes_from_rgb`) ALREADY compute opponent chromaticity `(rg, by)` and fine texture across both flanks of every `PEAK_CHAIN`!
- **Endpoints**: `Color/Surface` (lit flank) `->` `Color/Surface` (shadowed flank) across a `PEAK_CHAIN`.
- **Binder Stage**: `emit_pending_edges("v2")`.
- **Observables & Closed-Form Signal**:
  - Across a chain's left/right flanks (`chain_flanks.chain_flank_fields`):
    - High luminance step: $|\Delta L| = |\bar{L}_+ - \bar{L}_-| > \tau_L$.
    - **Chromaticity invariance** (Rubin & Richards / Retinex invariant): $\|\Delta(rg, by)\| < \tau_{\text{chrom}}$ AND **micro-texture continuity**: $|\Delta \text{roughness}| < \tau_{\text{tex}}$ (the surface's grain continues undisturbed through the shadow penumbra!).
  - Emits `ILLUMINATION_BOUNDARY` (`lit -> shadow`), allowing `BorderOwnership` and `SURFACE_CONTINUITY` to bridge the underlying surface right across the shadow edge.
- **`EdgeParams` Schema (`EDGE_ILLUMINATION_BOUNDARY`)**:
  - `lum_ratio`: `PopAxis("lum_ratio", n_bins=8, lo=0.0, hi=1.0, circular=False)`
  - `chrom_drift`: `PopAxis("chrom_drift", n_bins=8, lo=0.0, hi=0.5, circular=False)`
  - `penumbra_width`: `PopAxis("penumbra_width", n_bins=8, lo=0.0, hi=4.0, circular=False)`

---

### 6. `CONCAVE_MINIMA_JOINT` (Hoffman–Richards Part Segmentation & Attachment)
- **Why It Is Missing**: `PART_CONTAINMENT` (`binder.py:3415`) records generic container-to-part inclusion, and the comment at `binder.py:1073` notes that `PART_SEQUENCE / PART_AT_ANGLE / OBJECT_PART` were removed. However, without an explicit **part-attachment joint** at matched pairs of negative-curvature contour extrema (`Corner` / `CONTOUR` negative curvature minima facing each other across an interior neck), V4/LO cannot segment a non-convex silhouette (e.g., an L-polyomino, animal limb, or overlapping silhouette) into articulated parts or record *where* and *at what angle* two parts join.
- **Endpoints**: `RegionShape / CurveComplex` (larger trunk part) `->` `RegionShape / CurveComplex` (smaller limb/protrusion part).
- **Binder Stage**: `emit_pending_edges("lo_pit")`.
- **Observables & Closed-Form Signal**:
  - Identify owner-relative **concave** `Corner` detections (`turn * sign(signed_bo) < 0`) along a silhouette `PEAK_CHAIN`.
  - Pair two concave corners $(c_1, c_2)$ whose inward bisectors oppose across the interior field (`interior_field_for > 0.5` along $\overline{c_1 c_2}$) and whose chord length $\|\mathbf{p}_{c_1} - \mathbf{p}_{c_2}\|$ is a local minimum of interior width (necking).
- **`EdgeParams` Schema (`EDGE_CONCAVE_MINIMA_JOINT`)**:
  - `neck_ratio`: `PopAxis("neck_ratio", n_bins=8, lo=0.0, hi=1.0, circular=False)` (chord width / part diameter)
  - `joint_angle`: `PopAxis("joint_angle", n_bins=12, lo=0.0, hi=2*math.pi, circular=True)` (attachment bearing in trunk frame)
  - `cusp_sharpness`: `PopAxis("cusp_sharpness", n_bins=8, lo=0.0, hi=math.pi, circular=False)`

---

### 7. `PHYSICAL_SUPPORT` (Gravity-Aligned Surface Support)
- **Why It Is Missing**: `CONTACT` (`binder.py:4560`) is undirected proximity: two objects brushing side-by-side, an object hanging from a ceiling, and a box resting on a table all produce the exact same undirected `CONTACT` edge. Yet `v2.py` (`BO_GRAVITY_WEIGHT` image-down prior `+y`) and `SurfaceFragment` (`normal_world` with upward image component $n_y < 0$ in `y`-down image coordinates) already distinguish horizontal supporting tops from vertical walls!
- **Endpoints**: `Surface / SurfaceFragment` (supporting lower surface) `->` `Surface / RegionShape` (supported upper object).
- **Binder Stage**: `emit_pending_edges("lo_pit")`.
- **Observables & Closed-Form Signal**:
  - Shared horizontal `SHARED_BOUNDARY` / `CONTACT` interface where:
    1. Supporting surface normal points against image gravity ($\mathbf{n}_{\text{sup}} \cdot (0, -1, 0) > \cos 35^\circ$, i.e., upward-facing top face of the lower box), and
    2. Supported object's centroid lies above the contact chord ($y_{\text{obj}} < y_{\text{contact}}$ in `y`-down coordinates) within horizontal support polygon bounds $[x_{\min}, x_{\max}]$.
- **`EdgeParams` Schema (`EDGE_PHYSICAL_SUPPORT`)**:
  - `upward_align`: `PopAxis("upward_align", n_bins=8, lo=0.0, hi=1.0, circular=False)`
  - `com_margin`: `PopAxis("com_margin", n_bins=8, lo=-1.0, hi=1.0, circular=False)` (horizontal stability margin inside support chord)
  - `contact_span`: `PopAxis("contact_span", n_bins=8, lo=0.0, hi=4.0, circular=False)`
