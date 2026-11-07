"""parameters/_text.py -- the canonical parameter text (Move 7b).

Byte-for-byte the former parameters.py. Executed exactly once, in the
original statement order, so every intra-file dependency
(e.g. THETA_FERTILE_PROTECT = 0.1 * RF_MAX) resolves as it did before
the split. Nothing else in this package re-defines a constant; the
view modules only re-export names from here.
"""
from __future__ import annotations

"""
parameters.py — Canonical Architecture Configuration
Single source of truth for constitutional constants and homeostatic baselines.

Organization (v31):
  1.  Numerical safety
  2.  Hash widths
  3.  Time & history windows
  4.  Crystallization & witness thresholds
  5.  Capacity limits
  6.  Activation & adaptation
  7.  Edge weights (shared EMA parameters)
  8.  Edge homeostasis & per-type thresholds
  9.  Fertility & growth potential
  10. Credit & step-blame
  11. Contextual fertility
  12. Extraction: lenses
  13. Extraction: binders (order-1)
  14. Extraction: binders (order-2)
  15. Extraction: self-binders
  16. Extraction: quantization
  17. Binder threshold mechanism
  18. Builder (§4.22)
  19. Consolidator (§5.13)
  20. Mapper (§6.8)
  21. Executor (§7.8)
  22. Prediction (§8.8)
  23. Utility engine
  24. Satiety & lockouts
  25. Edge type classification
  26. Legacy (v30 constants retained during migration)

Naming convention: constants use SPEC NAMES where the spec provides one.
Greek letters map to English names (τ→TAU, θ→THETA, κ→KAPPA, ε→EPS, λ→LAMBDA,
η→ETA, α→ALPHA, β→BETA, γ→GAMMA). Module prefixes are NOT added; if a
spec name is unique across all sections, it is used bare. Collisions are
handled by section context or explicit disambiguating suffixes.
"""

import math

# [GAS LINE] Toggle for discriminative-mass weighting in query scoring.
# When False, evidence_score sees uniform mass and behaves exactly as before
# this feature existed. Kill switch, not a tuning parameter.
EVIDENCE_DISCRIMINATIVE_MASS: bool = True
import os
from typing import Dict, Optional, Tuple

# =============================================================================
# 1. NUMERICAL SAFETY
# =============================================================================

EPSILON = 1e-9

# =============================================================================
# 2. HASH WIDTHS
# =============================================================================

HASH_BITS = 64  # Legacy OPAQUE signature hash width
HASH_BITS_CANONICAL = 128  # v31 canonical subgraph hash width (§4.12)

# =============================================================================
# 3. TIME & HISTORY WINDOWS
# =============================================================================

W_HISTORY = 20  # General history window (legacy)
W_T = 10  # Temporal coherence window for transduction Ch 3
W_R = 20  # Rhythm window (self-binder + Periodicity binder)
W_CM = 10  # Co-motion window (velocity coherence)
W_SEQ = 20  # Sequence window (§2.3, §5.7)
W_SYNC = 20  # Synchrony window (P2-2)
W_CONFIRM = 5  # Effect-detection window (causal attribution, §8.8)
W_CHUNK = 20  # CHUNK co-activation window
W_PLAN_CHUNK = 100  # Legacy: plan-reuse window for CHUNK

# =============================================================================
# 4. CRYSTALLIZATION & WITNESS THRESHOLDS
# =============================================================================

# v30 concept-path thresholds (retained for ConceptGraph's witness path)
THETA_OPP = 3
THETA_CONF_DEFAULT = 3
THETA_CONF_MIN = 2
THETA_CONF_MAX = 10
THETA_ANCHOR_DEFAULT = 2
THETA_ANCHOR_CAUSAL = 1
THETA_SPAN = 2
THETA_WITNESS = 5
THETA_DECISIVE = 0.5
ETA_ARITY = 0.05
RHO_SYM_MAX = 0.5
N_MAX_AMBIGUITY = 50
TAU_AMBIGUITY = 500
TAU_FORGET = 1000
TAU_TOMBSTONE = 1000
N_LN = 8
THETA_LN = 0.6

# v31 subgraph-path thresholds (§4.17)
THETA_OPP_SUBGRAPH = 3
THETA_CONF_SUBGRAPH = 3
THETA_SPAN_SUBGRAPH = 2
THETA_ANCHOR_SUBGRAPH_SINGLE = 1  # SPATIAL_OBJECT, EVENT_CAUSAL, TEMPORAL_CHAIN
THETA_ANCHOR_SUBGRAPH_DEFAULT = 2  # All other families

# =============================================================================
# 5. CAPACITY LIMITS
# =============================================================================

N_CAND_MAX = 1000
N_LAYER_MAX = 500
N_NODE_MAX = 5000
N_TOMB_MAX = 1000
# v30 §2.1.16: N_protoMax — max sensory-grounding prototypes. Previously
# misnamed N_ANCHOR_MAX ("InnateReflex working-memory cap"), which collided
# semantically with the constitutional anchor bound N_ANCHOR_MAX_3D = 16
# (13_Update §3.4 working-memory bottleneck).
N_PROTO_MAX = 1000
N_ANCHOR_MAX = N_PROTO_MAX  # DEPRECATED alias; prefer N_PROTO_MAX / N_ANCHOR_MAX_3D
N_ANCHOR_MAX_3D = 16  # 13_Update §3.4: constitutional working-memory bound
N_CTX_MAX = 50  # Per-concept context cap (FertilityEngine)
N_EDGE_MAX = 100000  # v30 §2.1.6: max tracked (state, action) pairs (AEM)

# v31 §2.8.1: max valid operator-instances considered per promoted operator
# (before the family-key/anchor filter prunes to valid_instances).
OPERATOR_MAX_INSTANCES: int = 100

# v31 Builder/Consolidator/Mapper capacity bounds
BUILDER_CANDIDATE_MAX = 50  # §4.12 max nodes in a CandidateSubgraph
CORE_SIZE_CAP_NODES = 20  # §4.12, §5.13
CORE_SIZE_CAP_EDGES = 40  # §4.12, §5.13
FRONTIER_MAX = 10_000  # §4.11.1
TRANSPOSITION_TABLE_MAX = 1_000_000  # §4.11.1
MAX_ANCHORS_PER_CANDIDATE = 32  # §4.3.1
MAX_SITUATION_SIZE = 50  # §6.2
MAX_MCS_CANDIDATES = 100  # §5.7
MAX_PENDING = 1000  # §8.5.1
MAX_SURPRISE_QUEUE = 100  # §4.10.5, §8.6.4
MAX_SURPRISE_PER_STEP = 10  # §8.6.4
N_REDUCTION_TEMPLATES = 50  # §5.4
PATTERN_NODE_TTL = 100  # §1.11
TARJAN_CADENCE = 1  # [P2-3] §5: recomputes of the articulation-point cache
# in _prune_pattern_nodes happen at most once every TARJAN_CADENCE steps.
# 1 = recompute every pruning pass (exact, no staleness); raise only if
# Tarjan cost dominates step time on large structure graphs.
N_PROTO_MAX_EBAF = 200  # §8.4.1
# NOTE (§4.17): candidate `contexts` is unbounded in v31. No cap constant
# exists; if memory pressure becomes real, add a deterministic hash-order
# eviction policy and introduce MAX_CONTEXTS_PER_CANDIDATE at that time.

# =============================================================================
# 6. ACTIVATION & ADAPTATION
# =============================================================================

THETA_ACT_MIN = 0.1
THETA_ACT_MAX = 0.9
THETA_ACT_FLOOR = 0.05
KAPPA_ACT = 0.04
RHO_ACT_TARGET = 0.2
KAPPA_SUPPORT = 1.0
ETA_SUPPORT = 0.05
BETA_SUPPORT = 0.8
S_MAX = 100.0
SIGMA_ACT_NOISE = 0.05
DELTA_ACT = 0.02
ADAPT_RATE = 0.02
# v30 §2.18.2 architectural guarantee: adapt_cap MUST satisfy
#     adapt_cap > 1.0 − θ_act_floor
# so sustained activation at z = 1.0 eventually deactivates and re-onsets.
# The v30 table default (0.7) violates its own MUST at θ_act_floor = 0.05;
# the normative constraint governs over the stale default. Derived from
# constitutional constants (one hysteresis margin δ_act above the bound):
ADAPT_CAP = 1.0 - THETA_ACT_FLOOR + DELTA_ACT  # = 0.97 > 0.95 ✓
ADAPT_DECAY = 0.8

# =============================================================================
# 7. EDGE WEIGHTS (shared EMA parameters)
# =============================================================================
# §1.12: all edge types share the same EMA update formula:
#   weight = BETA_W * weight + (1 - BETA_W) * signal
# β_edge in the spec corresponds to BETA_W here.
# W_INIT and W_MAX apply to all edge types uniformly in v31.

W_INIT = 0.5
W_MAX = 10.0
BETA_W = 0.99  # Spec: β_edge
ETA_W = 0.1    # Edge-weight delta EMA decay (v30 value; consumed by
               # concept_graph's per-concept params and edge homeostasis)
DELTA_W = 0.001
KAPPA_COACT = 0.1  # Legacy: edge.CoAct_EMA update rate

# --- Hebbian edge reinforcement (structural_evidence boost) ---
# β_e: per-step decay of the per-edge crystallization-evidence EMA.
#   0.998^500 ≈ 0.367, so an edge idle for TAU_EDGE_INACTIVE = 500 steps
#   has its evidence decayed to ~0.37 — clearly on its way to zero
#   without having fully forgotten recent usefulness. Half-life ≈ 346.
# η_e: boost magnitude at evidence saturation. An edge with e → 1.0 has
#   its firing signal multiplied by (1 + η_e). 5.0 → 6× boost at
#   saturation. Raised from 0.5 after Part 1b canary showed zero pin
#   movement — the observed evidence distribution (mean 1e-5 to 2e-3)
#   is three orders of magnitude below the range that η_e = 0.5 was
#   designed for. At η_e = 5.0, tool edges near the mean get a ~1%
#   boost and high-evidence edges get ~25%, which is a magnitude that
#   can plausibly affect activation spreading and situation retrieval.
#   If pins move by more than ~50, this is the first knob to lower.
BETA_E = 0.998
ETA_E = 5.0

# =============================================================================
# 8. EDGE HOMEOSTASIS & PER-TYPE THRESHOLDS
# =============================================================================
# §1.12 per-type thresholds and homeostatic targets. v31 initializes all
# types to the same values; type-specific tuning is v32.

THETA_EDGE_ACTIVE = 0.1  # Initial active-edge threshold, all types
RHO_EDGE_TARGET = 0.3  # §1.12 initial homeostatic target
KAPPA_EDGE_HOMEOSTASIS = 0.05  # κ in θ_edge ← θ_edge + κ(r̄ - ρ_target)
KAPPA_ENTROPY_FEEDBACK = 0.01  # [P2-4] §8: gain on type-dominance term in
# θ_edge ← θ_edge + κ(r̄ - ρ_target) + κ_entropy·max(0, share - 1/|types|).
# Pushes down over-dominant edge types to keep the type-share distribution
# from collapsing (entropy homeostasis).
TAU_EDGE_INACTIVE = 500  # Pruning threshold per edge type

# =============================================================================
# 9. FERTILITY & GROWTH POTENTIAL
# =============================================================================

RF_MAX = 10**9
ALPHA_GP = 0.3
BETA_GP = 0.6
GAMMA_GP = 0.1
DELTA_GP = 0.2
GP_MAX = 1.0 + ALPHA_GP + BETA_GP + GAMMA_GP + DELTA_GP
TAU_RECENT = 200

# =============================================================================
# 10. CREDIT & STEP-BLAME
# =============================================================================

EPS_INITIAL = 0.25
C_ENABLEMENT = 1.0
C_PLAN = 0.1
C_SUBGOAL = 0.05
C_MAX = 10**6
KAPPA_SPEC = 4.0
RHO_REVERSAL = 0.5
N_STEP_MIN = 5

# =============================================================================
# 11. CONTEXTUAL FERTILITY
# =============================================================================

ETA_CTX = 0.3  # §8.8 (v31 value; v30 was 0.5)
BETA_CTX = 0.995  # §8.12 per-step CF decay
K_CONTEXT = 10
CF_MAX = RF_MAX
C_CTX = C_ENABLEMENT
THETA_PROTO_EF = 0.2  # §8.4.2 prototype registration threshold
TAU_PROTO_DECAY = 1000  # §8.4.3 prototype decay timescale
ETA_EBAF = 0.2  # §8.8 η_ebaf (v30 had 0.3)

# =============================================================================
# 12. EXTRACTION: LENSES
# =============================================================================
# §2.2. Constants for lens emission, normalization, and signal construction.

BOUNDARY_LENS_RADIUS_MULT = 1  # radius = R_BOUNDARY * 2^scale_idx
COLOR_QUANT_BINS = 12  # 12 bins per RGB channel
MOTION_SPEED_MAX = 1.0  # velocity normalization ceiling (V_MAX)
MOTION_DIR_BINS = 8  # direction quantization bins per axis
POSITION_NORM_RANGE = 1.0  # egocentric position normalization range.
                                     # Set to the expected maximum displacement of
                                     # entities from the agent. Tabletop (m): 1.0
                                     # is correct. Larger environments must raise
                                     # this to avoid clamping-induced information loss.
ONSET_AGE_NORM = 100.0  # age normalization window
# SIZE_RADIUS_MIN / SIZE_RADIUS_MAX are derived from the adapter's physical
# radii in Section 28 (single definition; r_min / R_max_obj per 13_Update).
TEXTURE_DEPTH_VAR_MAX = 1.0  # local depth-variance ceiling
TEXTURE_DELTA_EMA_BETA = 0.9         # planarity-delta EMA decay (Texture lens)
PRESENCE_EMA_BETA = 0.9              # presence EMA decay (all lenses)

# --- Orientation lens (v8) ---
# Angle bins over [0, π). N_theta = 24 → 7.5° resolution. Slot 0 of the
# ORIENTATION signature is circular (see CIRCULAR_CYCLE_LENGTHS in
# primitives). NOTE: primitives.CIRCULAR_CYCLE_LENGTHS["ORIENTATION"][0]
# and the ORIENTATION range in CONSTITUTIONAL_RANGES must stay in sync
# with this value; both are hard-coded there for layering reasons.
ORIENTATION_N_THETA = 24
# Elongation bins over [0, 1].
ORIENTATION_N_ELONG = 5
# Above this elongation (near-circular region), the major axis is
# ill-defined and the lens skips the entity.
ORIENTATION_ELONG_SKIP = 0.9
# Minimum patch gradient energy (region "roughness") for the Orientation
# lens to emit at all. On a uniform patch grid, isotropic/flat patches
# pass the ORIENTATION_ELONG_SKIP gate (elongation ~= 0 < 0.9) and emit an
# essentially arbitrary major axis, seeding a noise lattice that
# ORIENTATION_BEND/JUNCTION weld into one giant component. Measured on the
# apple/car/cow 8x8 patch grid: 80%+ of emitters sit below roughness 0.05
# (flat background) sit below 0.05; smooth-object edges live in the
# 0.035-0.06 band (apple edge max is only 0.064 on 32px patches). 0.05 severed
# that band and killed CONTINUITY on apple/cow (Part=0); 0.035 keeps the
# soft-edge band while still pruning the flat floor (median roughness ~= 0.030).
ORIENTATION_GRADIENT_MIN = 0.035

# --- Orientation segment lens (v9) ---
ORIENTATION_SEGMENT_TARGET_N = 6
ORIENTATION_SEGMENT_MIN_ARC = 4.0
ORIENTATION_SEGMENT_MAX_PER_REGION = 8
ORIENTATION_SEGMENT_N_ARC = 8
ORIENTATION_SEGMENT_ARC_MAX = 32.0
ORIENTATION_SEGMENT_N_POS = 8
ORIENTATION_SEGMENT_MIN_POLYLINE = 12

# --- Contour chaining (v9, removed in v10; dormant) ---
CHAIN_MAX_GAP = 8.0
CHAIN_MAX_ANGLE_DEV = 0.4
CHAIN_MIN_POLYLINE = 8

# --- T-junction detection (v9) ---
T_JUNCTION_MAX_GAP = 6.0
T_JUNCTION_MAX_PERP_DEV = 0.5
T_JUNCTION_MIN_FOREIGN_ARC = 10.0

# --- CURVATURE binder (v9) ---
CURVATURE_SIGMA_THETA = 0.3
CURVATURE_MIN_DELTA_THETA = 0.05

# --- OCCLUSION binder (v9) ---
OCCLUSION_MIN_CONFIDENCE = 0.4
OCCLUSION_3D_MIN_DEPTH_DIFF = 0.08
OCCLUSION_3D_SIGMA_DEPTH = 0.20
# --- Stage 3: 2.5D Ordinal Depth DAG + V4 Polar Fragments ---
# All unmeasured priors, not tuned (futurework6.md §Stage 3).
# Per-source trust applied to depth-constraint effective weights before cycle
# invalidation; T-junctions are trusted above border-ownership votes.
DEPTH_SOURCE_TRUST = {"tjunction": 1.0, "border_ownership": 0.5}
# Minimum |mean signed border-ownership| before a figure/ground ordinal
# constraint is emitted (below this the vote is noise).
# [ROUND 10 / C5] Audited against the v2 field (diagnostics/_field_calibration.py):
# the compared quantity is the mean signed border-ownership VOTE m in [-1, 1],
# not an integration/driving value, so the v2 field formula does not move its
# units. Rule: emit an ordinal constraint only when |m| >= this, i.e. when the
# mean vote is a clear side rather than noise around zero.
DEPTH_BO_MIN_STRENGTH = 0.15
# Pasupathy & Connor object-centred polar fragments: target arc length as a
# fraction of the object envelope radius, and the Menger-curvature clamp.
CONTOUR_FRAGMENT_LEN_FRAC = 0.35
CONTOUR_FRAGMENT_KAPPA_CAP = 8.0

# =============================================================================
# STAGE 4: ARTMAP VIGILANCE & SHOCK SKELETON PARAMETERS
# =============================================================================
# Set to None to keep behavior byte-identical to pre-Stage 4.
# When set to a float in (0, 1], enables ARTMAP prototype resonance merging.
ART_VIGILANCE: Optional[float] = None
ART_MAX_CANDIDATES: int = 64

# Shock graph extraction parameters
SHOCK_STEP_FRAC: float = 0.03
SHOCK_R_MIN_FRAC: float = 0.03
SHOCK_R_MAX_FRAC: float = 1.05
SHOCK_TOL_SYM: float = math.pi / 12.0
SHOCK_TOL_TAPER: float = math.pi / 6.0
SHOCK_MIN_RUN: int = 4
SHOCK_PRUNE_ASPECT: float = 0.5
SHOCK_JOIN_FRAC: float = 1.2

# =============================================================================
# STAGE 5: ASPECT GRAPH & MULTI-VIEW TRANSFORMATION CONSTANTS
# =============================================================================
# [Stage 5 activation] ON. The descriptor source was rewired from dead shock
# ribbons to the live field's PEAK_CHAINs (memory/aspects.py
# build_view_descriptor_from_chains), so a view is now always constructible
# on the region path. Prototypes are viewpoint-windowed by
# ASPECT_ASSIGN_MAX_DEG and links connect DIFFERENT prototypes, so the graph
# is a manifold of per-viewpoint templates joined by experienced
# transformations -- not one averaged blob.
ASPECT_ENABLED: bool = True
ASPECT_PAIR_MODE: str = "metadata"       # "metadata" | "temporal"
ASPECT_VIGILANCE: float = 0.80
# [O7] THE MEMORY CAPS.
#   ASPECT_MAX_VIEWS       The view store's horizon.  It was UNBOUNDED:
#                          every view ever observed was retained in full
#                          (parts, surface, gist, peak map), so the
#                          system's largest memory sink grew without
#                          limit.  The horizon evicts the OLDEST views and
#                          protects a RARE prototype (one with a single
#                          member), because losing its member loses the
#                          prototype.
#   CORE_LIBRARY_MAX_ENTRIES The core library's capacity.  It had no cap:
#                          a core that was never queried accumulated
#                          forever and kept occupying the
#                          MAX_MCS_CANDIDATES ranking pool.  The valve
#                          evicts by recency and fertility and protects
#                          the rare classes.
#   CANDIDATE_CONTEXTS_MAX  A candidate's context set.  It was
#                          UNBOUNDED; the cap keeps the deterministically
#                          smallest context hashes (a hash order, so the
#                          policy does not depend on arrival time).
ASPECT_MAX_VIEWS: int = 4096
CORE_LIBRARY_MAX_ENTRIES: int = 512
CANDIDATE_CONTEXTS_MAX: int = 64
# [O5d] The prototype's RELIABILITY rate: how fast a prototype's
# reliability tracks the evidence of its latest recall.  The reliability
# is an EMA of (score - variability), so a prototype whose members
# disagree is less reliable than one whose members agree at the same
# mean.  0.2 = the reliability moves a fifth of the way to the evidence
# per recall.
ASPECT_PROTO_RELIABILITY_KAPPA: float = 0.2
ASPECT_ASSIGN_MAX_DEG: float = 25.0
ASPECT_LINK_MAX_DEG: float = 50.0
ASPECT_LINK_SIGMA_DEG: float = 30.0
ASPECT_TRACE_ETA: float = 0.2
ASPECT_W_MAX: float = 1.0
ASPECT_TRACE_DECAY: float = 0.998
ASPECT_LINK_MIN_W: float = 0.05
ASPECT_SIGMA_POS: float = 0.35             # in units of R_env
ASPECT_MATCH_MIN: float = 0.35
ASPECT_ORDER_MISMATCH: float = 0.5         # part similarity multiplier when shock order differs
ASPECT_PERSIST_CAP: int = 64
# [Q2] Graded part matching, SHIPPED DARK (off) per the operating policy
# (P1: a scoring-facing change is not enabled until second instances per
# class confirm it). When on, part similarity uses ASPECT_PART_SLOTS and the
# graded slot distance replaces the exact slot-0 equality gate.
# Slot choice is not a free parameter: it is the measured noise screen on
# TRAINING views (diagnostics/_q2_slots.py). A slot is dropped when a single
# value covers >= 75% of all parts (noise-dominated, no spread to match on):
#   slot 0 n_peaks_bin      top-1 share 0.313  F=0.34  -> kept
#   slot 1 mean_theta_bin   top-1 share 0.151  F=2.35  -> kept (strongest)
#   slot 2 theta_var_bin    top-1 share 0.764            -> dropped
#   slot 3 closure_bin      top-1 share 0.986            -> dropped
#   slot 4 curv_summary_bin top-1 share 0.973            -> dropped
#   slot 5 length_bin       top-1 share 0.322  F=0.67  -> kept
ASPECT_GRADED_PARTS: bool = False
ASPECT_PART_SLOTS: Tuple[int, ...] = (0, 1, 5)
# [SIZE SLOT] The view's absolute size (object RMS spread, pixels) is stored as
# its OWN slot (ViewRecord.size) apart from the normalized shape. The matcher
# reads it through a separate log-ratio kernel with this sigma (in log units:
# 0.7 ~= a factor-e mismatch). Enabled by default -- the slot exists precisely
# so size is available to matching; set False to score on normalized shape only.
ASPECT_SIZE_SLOT_ENABLED: bool = True
ASPECT_SIZE_SIGMA: float = 0.7

# =============================================================================
# [W2 / O7] POPULATION-CODED VIEW DESCRIPTOR V2
# =============================================================================
# PROVENANCE (written BEFORE the look-#2 graded number; never tuned after it):
# these are mechanism-derived, not fit to accuracy.
#   * The fragment slots ARE the PEAK_CHAIN population already carried on the
#     detection (POPULATION_AXES["PEAK_CHAIN"]: theta_dist circular 12,
#     curvature 8, length 8). Matching reads the graded overlap
#     (core.population.population_similarity); bins stay the lookup index and
#     enter only as a small coarse term.
#   * The arrangement is object-centred and UPRIGHT: r = |pos| in RMS-spread
#     units, theta = atan2(x, -y) measured from image-up. The position kernel
#     is split into a radial Gaussian (sigma 0.45 RMS) and a circular angular
#     Gaussian (sigma 45 deg), so a part contributes a graded soft address
#     rather than a hard bin.
#   * Surface slots are sampled with the soft interior field as the WEIGHT
#     (perception/interior_field.interior_weighted_sample), never a mask:
#     "interior" weights = interior, "flank" weights = 1 - interior, so the
#     two sides of each contour fragment are represented separately.
#   * Group weights sum to 1 and are declared here. They are the same for
#     every class and every arm; the ablations drop a whole group and
#     renormalise, they do not retune a weight.
# The switch ships DARK: the legacy matcher remains the scorer of record until
# the O9 parity gate is decided.
# [REPRESENTATION MISSION] The graded (population-coded) descriptor is the
# LIVE one: ON. The flag was dark, so the live descriptor carried no surface
# group and no global layout, and the live readout scored a descriptor the
# measured system never produced. It was RENAMED from ASPECT_V2_ENABLED
# because it now selects the V3 descriptor (gist + surface + arrangement +
# shape + populations), not the V2 one. Provenance: rule = "the system
# recognises through what it measures"; date 2026-10-20; mechanism version
# descriptor-v4.
ASPECT_V3_ENABLED: bool = True
ASPECT_V2_W_SHAPE: float = 0.45
ASPECT_V2_W_POS: float = 0.30
ASPECT_V2_W_SURFACE: float = 0.25
# [V3] The CALIBRATED fusion weights (label-free).
#
# Each group is weighted by the product of two MEASURED, LABEL-FREE
# quantities, both computed by diagnostics/_calibrate.py on the dev split:
#   reliability    -- the group's mean similarity between an image and its
#                     OWN perturbed version (shift, scale, gain, blur, noise);
#   distinctiveness-- 1 - (the mean off-diagonal similarity) / (the mean
#                     diagonal similarity) over the dev corpus: how far the
#                     group separates different images.
# The weight is reliability x distinctiveness, normalised over the groups
# that are present. NO LABEL is used in the derivation, so no label can leak
# into the representation. A label-based calibration is reported separately
# as a DIAGNOSTIC UPPER BOUND only.
# Provenance: rule = "a noisy group must not dilute an informative one";
# data source = the ETH-80 dev split (6 instances per class, 3 views);
# date 2026-10-20; mechanism version descriptor-v3.
# The VALUES below are the calibration's output (diagnostics/_calibrate.py,
# manifest diagnostics/_calibration_manifest.json). Reliability is the mean
# self-similarity under shift / scale / gain / blur / noise; distinctiveness
# is the LABEL-FREE nearest-neighbour margin (how far the best other item
# beats the mean of the rest), normalised by the group's similarity range;
# the weight is their product, normalised:
#   shape       reliability 0.7538  distinctiveness 0.1703  -> 0.1939
#   arrangement reliability 0.7565  distinctiveness 0.1792  -> 0.2048
#   surface     reliability 0.9039  distinctiveness 0.1574  -> 0.2149
#   layout      reliability 0.7856  distinctiveness 0.3256  -> 0.3863
# The layout group has the largest margin (it separates the most), so it
# leads; the surface group is the most reliable; the shape group is stable
# but uninformative, so it is quiet.
# (These are the values measured WITHOUT the orientation-anisotropy term in
# the layout's similarity: with that term the layout alone rose but the
# fused arm fell, so the term was dropped and the calibration re-measured
# for the configuration that ships.)
# [RETIRED] The chain-based shape group is retired from the fusion (weight
# 0): a chain's members change with the view, so the group diluted the
# informative ones. Its measurement is still stored on the descriptor (the
# chain's graded population), and the PEAK-FRAGMENT group -- the same
# measurement rebuilt on the peaks, which recur across views -- replaced it.
ASPECT_V3_W_SHAPE: float = 0.0
ASPECT_V3_W_ARRANGEMENT: float = 0.1544
ASPECT_V3_W_SURFACE: float = 0.2295
ASPECT_V3_W_LAYOUT: float = 0.3489
# The SIZE group: the object's absolute extent (its own kernel, the
# RMS-normalised spread is NOT reused -- the normalised shape is the other
# groups' business). Its weight is calibrated with the others; the size is
# the most SCALE-sensitive group, so the calibration gives it the smallest
# share if it deserves one.
#
# The calibration's output (diagnostics/_calibrate.py, manifest
# diagnostics/_calibration_manifest.json): reliability is the mean
# self-similarity under shift / scale / gain / blur / noise, distinctiveness
# is the LABEL-FREE nearest-neighbour margin, and the weight is their
# product, normalised:
#   arrangement reliability 0.7159  distinctiveness 0.1965  -> 0.1805
#   surface     reliability 0.8172  distinctiveness 0.2352  -> 0.2467
#   layout      reliability 0.7856  distinctiveness 0.3314  -> 0.3341
#   size        reliability 0.9966  distinctiveness 0.0  -> 0.0212
#   fragments   reliability 0.9323  distinctiveness 0.1818  -> 0.2175
# The PEAK-FRAGMENT group (the soft 3-D population over the peak fragments)
# is the MOST RELIABLE of the shape-like groups (0.9323 against the chain
# shape group's 0.7461): the peaks recur across views where the chains'
# members do not, which is exactly why the fragment map is built from them.
# The size is the MOST reliable group (an extent barely moves under a shift)
# and the LEAST distinctive (it separates almost nothing on its own), so it
# gets the smallest share -- which is what the product rule says.
ASPECT_V3_W_SIZE: float = 0.0203
# The PEAK-FRAGMENT group: the soft 3-D population over the peak fragments
# (the angular position around the energy centre x the signed curvature x
# the orientation relative to the radial direction). Its weight is
# calibrated with the others; it starts at zero and the calibration decides.
ASPECT_V3_W_FRAGMENTS: float = 0.2469

# [SCORER OF RECORD] The parity gate was MET on 2026-10-20 and the single
# permitted switch was TAKEN: the fused descriptor (ASPECT_V3_ENABLED) is
# the scorer of record. Recorded here so the switch state is a fact of the
# tree rather than an inference from a document.
ASPECT_SCORER_OF_RECORD: str = "fused"

# [W2/D + W2/F] The middle-tier groups and the per-group corpus nulls.
#
# WEIGHTS: the label-free calibration of diagnostics/_calibrate.py on the
# DEV split (each group's weight is its reliability under the five
# pre-registered perturbations times its distinctiveness -- the
# nearest-neighbour margin; no class label is consulted), renormalised
# over the RETAINED groups.
#
# RETIRED (measured on dev, 2026-10-21, with the calibrated kernel):
#   * the MIDDLE TIER (pairs, texture2) is retired to weight 0. It was
#     built and measured: the fragment-PAIR histogram scores 0.1823 on
#     its own (chance 0.125) and DILUTES the fusion (-0.0052); the
#     second-order texture autocorrelation scores 0.2552 on its own
#     and is neutral (+0.0001). The implementation stays in the tree,
#     measured and available.
#   * the per-group CORPUS NULL correction is OFF (the table is empty).
#     It was tested: chance-correcting each group against its own null
#     costs 0.099 overall (0.7604) and makes the fusion worse than its
#     own best group. The nulls are kept in the manifest as the
#     evidence for that decision.
#   * the CHAIN-based shape group stays retired (weight 0): a chain's
#     members change with the view, so the group was not informative.
#
# The ARRANGEMENT group is RETAINED: with its kernel calibrated to the
# measured radius and bearing spread it ADDS (+0.0110 overall, +0.0277
# on the quadruped group) where before it diluted.
#
# Manifest: diagnostics/_calibration_manifest_raw.json (scale: raw)
ASPECT_V3_W_PAIRS: float = 0.0000
ASPECT_V3_W_TEXTURE2: float = 0.0000
ASPECT_V3_GROUP_NULL: dict = {}
# =============================================================================
# [TIER + DEPTH] The two NEW groups: the MIDDLE TIER (the assemblies over
# fragment populations) and the STATIC DEPTH CUES.
#
# THEIR WEIGHTS come from the SAME label-free calibration as every other
# group (diagnostics/_calibrate.py: reliability x distinctiveness, both
# measured on the dev split, no label consulted). A group that does not
# earn its weight is retired to 0, exactly as the pair histogram was.
#
# The per-group CORPUS NULL correction stays OFF for these groups too: the
# null correction was MEASURED to cost 0.099 overall (0.7604 against 0.8599
# fused), so the shipped decision is that every group is scored raw. The new
# groups' own corpus nulls are RECORDED below (for the report and for the
# calibration manifest), not applied.
#
# The MIDDLE TIER's INTERNAL structure (the corpus's fragment-type
# co-occurrence and the formations the MDL admitted) is a corpus-derived
# constant: it is measured on the dev corpus by diagnostics/_tier_corpus.py,
# label-free, and installed here.
# =============================================================================
# [TIER+DEPTH CONSTANTS BEGIN] -- written by diagnostics/_tier_install.py
# from the measured manifests (diagnostics/_tier_corpus.json,
# diagnostics/_tier_calibration.json, diagnostics/_depth_gate.json and
# diagnostics/_calibration_manifest_raw.json). Do not hand-edit.
# [TIER] The corpus: 84 fragment types, 1759 null-calibrated links above the corpus's
# independence null (0.070112), and the 54 formations the EXISTING MDL shadow
# admitted (diagnostics/_tier_corpus.py, the dev corpus, label-free).
ASPECT_TIER_CORPUS: dict = {
    "null": 0.070112,
    "n_types": 84,
    "n_links": 1759,
    "n_formations": 54,
    "links": (
        ((2, 0.060428), (3, 0.020937), (4, 0.013166), (5, 0.028707), (6, 0.017099), (7, 0.010277), (9, 0.000519), (14, 0.03109), (15, 0.005422), (17, 0.013166), (18, 0.009808), (19, 0.018587), (21, 0.008), (22, 0.0268), (23, 0.032343), (26, 0.012415), (28, 0.035493), (29, 0.039558), (30, 0.012508), (31, 0.011793), (34, 0.024), (36, 0.00964), (37, 0.00578), (53, 0.003331), (65, 0.002538), (66, 0.00653), (67, 0.006158), (68, 0.02965), (70, 0.04), (71, 0.043072), (73, 0.007961), (74, 0.000612), (75, 0.003717), (76, 0.012621), (77, 0.028707), (78, 0.030097), (79, 0.028308), (81, 0.011475), (82, 0.018044), (83, 0.047796)),
        ((1, 0.060428), (3, 0.045716), (4, 0.03826), (5, 0.03826), (6, 0.052518), (7, 0.015399), (9, 0.018144), (10, 0.010417), (13, 0.024007), (14, 0.010497), (17, 0.008439), (18, 0.038126), (19, 0.042611), (20, 0.029289), (21, 0.005693), (22, 0.031303), (23, 0.01586), (26, 0.01136), (27, 0.015789), (28, 0.026035), (29, 0.015716), (31, 0.008878), (34, 0.005693), (36, 0.010417), (37, 0.006027), (41, 0.002983), (42, 0.004975), (53, 0.001417), (58, 0.018353), (60, 0.011159), (63, 0.005222), (66, 0.001275), (68, 0.003888), (70, 0.013283), (71, 0.007453), (74, 0.00645), (77, 0.008439), (78, 0.05558), (81, 0.011227), (82, 0.01109), (83, 0.010497)),
        ((1, 0.020937), (2, 0.045716), (4, 0.001156), (5, 0.030108), (7, 0.017693), (9, 0.00695), (12, 0.001071), (13, 0.029595), (14, 0.012885), (15, 0.016173), (18, 0.027124), (19, 0.002671), (20, 0.050193), (21, 0.012797), (22, 0.011408), (23, 0.018578), (25, 0.008373), (27, 0.009881), (28, 0.024681), (31, 0.015445), (38, 0.001239), (39, 0.009775), (45, 0.005525), (47, 0.001198), (52, 0.005515), (58, 0.015696), (63, 0.01737), (65, 0.001198), (67, 0.006946), (71, 0.000894), (73, 0.006945), (75, 0.002671), (78, 0.005442), (82, 0.00408), (83, 0.005463)),
        ((1, 0.013166), (2, 0.03826), (3, 0.001156), (5, 0.02287), (6, 0.043033), (7, 0.002345), (9, 0.022346), (10, 0.015894), (11, 0.005474), (12, 0.008416), (13, 0.021508), (14, 0.005463), (15, 0.008487), (17, 0.008394), (18, 0.012711), (19, 0.009795), (20, 0.027027), (22, 0.018841), (25, 0.015507), (27, 0.031883), (28, 0.009903), (31, 0.001279), (34, 0.02011), (35, 0.006947), (37, 0.00406), (38, 0.001239), (45, 0.012628), (46, 0.014039), (47, 0.015569), (50, 0.016173), (51, 0.008428), (60, 0.015507), (65, 0.008383), (68, 0.001239), (70, 0.005484), (71, 0.008463), (74, 0.006945), (77, 0.001156)),
        ((1, 0.028707), (2, 0.03826), (3, 0.030108), (4, 0.02287), (6, 0.011657), (7, 0.002345), (9, 0.030044), (10, 0.030805), (11, 0.005474), (13, 0.029595), (17, 0.001156), (18, 0.012711), (19, 0.03829), (20, 0.011583), (21, 0.02011), (23, 0.004039), (25, 0.001239), (27, 0.024549), (28, 0.002514), (36, 0.015894), (38, 0.001239), (46, 0.006945), (47, 0.001198), (52, 0.005515), (74, 0.006945), (75, 0.002671), (77, 0.008394), (78, 0.03558), (79, 0.017139), (81, 0.005525), (82, 0.011245), (83, 0.020307)),
        ((1, 0.017099), (2, 0.052518), (4, 0.043033), (5, 0.011657), (10, 0.026678), (11, 0.030275), (13, 0.063433), (14, 0.006838), (17, 0.035189), (18, 0.047947), (19, 0.013084), (20, 0.009709), (21, 0.02834), (22, 0.044663), (23, 0.030035), (26, 0.002209), (27, 0.027469), (29, 0.024755), (30, 0.001664), (31, 0.007556), (34, 0.004049), (35, 0.03483), (36, 0.000837), (50, 0.016315), (53, 0.004608), (60, 0.008864), (68, 0.001356), (69, 0.02188), (70, 0.012146), (77, 0.027345), (78, 0.02775), (79, 0.049584), (81, 0.021417), (82, 0.011198), (83, 0.040792)),
        ((1, 0.010277), (2, 0.015399), (3, 0.017693), (4, 0.002345), (5, 0.002345), (9, 0.089634), (10, 0.031947), (11, 0.045242), (12, 0.036496), (13, 0.034324), (17, 0.002345), (21, 0.007874), (25, 0.015017), (28, 0.013699), (31, 0.024666), (33, 0.020212), (34, 0.015748), (35, 0.016874), (36, 0.015399), (37, 0.017143), (38, 0.007612), (44, 0.024762), (49, 0.002747), (57, 0.030967), (60, 0.000206), (68, 0.000206), (70, 0.023622), (71, 0.001196), (76, 0.038752), (77, 0.017693), (78, 0.030246), (79, 0.003245), (82, 0.012058), (83, 0.01045)),
        ((1, 0.000519), (2, 0.018144), (3, 0.00695), (4, 0.022346), (5, 0.030044), (7, 0.089634), (10, 0.051427), (11, 0.04086), (12, 0.04032), (13, 0.051661), (14, 0.021694), (17, 0.014648), (18, 0.002749), (19, 0.02403), (22, 0.009515), (23, 0.011246), (25, 0.013089), (28, 0.00079), (31, 0.023284), (35, 0.00446), (36, 0.026465), (38, 0.013089), (42, 0.006369), (44, 0.006686), (46, 0.005181), (49, 0.022099), (50, 0.022256), (51, 0.021296), (52, 0.013391), (60, 0.005668), (61, 0.000998), (65, 0.002519), (66, 0.00617), (68, 0.020509), (71, 0.00277), (76, 0.015224), (78, 0.00666), (79, 0.015304), (82, 0.024797), (83, 0.021694)),
        ((2, 0.010417), (4, 0.015894), (5, 0.030805), (6, 0.026678), (7, 0.031947), (9, 0.051427), (11, 0.038821), (12, 0.028526), (13, 0.04225), (15, 0.020916), (17, 0.000984), (18, 0.015928), (19, 0.028105), (20, 0.029289), (21, 0.005693), (23, 0.008347), (27, 0.00053), (31, 0.008878), (33, 0.001832), (36, 0.002563), (43, 0.039265), (44, 0.006027), (50, 0.012619), (51, 0.000215), (53, 0.001417), (57, 0.002124), (71, 0.039731), (77, 0.045716), (78, 0.03157), (81, 0.00401), (83, 0.033871)),
        ((4, 0.005474), (5, 0.005474), (6, 0.030275), (7, 0.045242), (9, 0.04086), (10, 0.038821), (12, 0.044364), (13, 0.054173), (15, 0.043056), (20, 0.012146), (21, 0.020561), (25, 0.011475), (27, 0.008864), (29, 0.003592), (31, 9.9e-05), (34, 0.005607), (35, 0.012149), (39, 0.016384), (42, 0.004859), (43, 0.005887), (44, 0.002293), (49, 0.002558), (51, 0.003592), (52, 0.007396), (53, 0.002209), (61, 0.008167), (77, 0.005474), (78, 0.03167), (83, 0.014536)),
        ((3, 0.001071), (4, 0.008416), (7, 0.036496), (9, 0.04032), (10, 0.028526), (11, 0.044364), (13, 0.056974), (15, 0.010474), (17, 0.008416), (20, 0.02008), (21, 0.01676), (28, 0.002676), (30, 0.007867), (31, 0.008618), (33, 0.007239), (35, 0.008761), (43, 0.018844), (44, 0.012346), (46, 0.006699), (47, 0.025455), (49, 0.022901), (50, 0.026434), (51, 0.023344), (52, 0.002259), (55, 0.012608), (58, 0.013292), (59, 0.004773), (60, 0.013353), (61, 0.016219), (62, 0.007121), (63, 0.0), (65, 0.018182), (66, 0.005595), (67, 0.00167), (75, 0.000898), (79, 0.007867), (82, 0.01479)),
        ((2, 0.024007), (3, 0.029595), (4, 0.021508), (5, 0.029595), (6, 0.063433), (7, 0.034324), (9, 0.051661), (10, 0.04225), (11, 0.054173), (12, 0.056974), (15, 0.058618), (17, 0.005335), (19, 0.002854), (20, 0.048128), (21, 0.014737), (22, 0.018325), (27, 0.002959), (33, 0.002089), (35, 0.025395), (36, 0.024007), (37, 0.0005), (38, 0.012808), (39, 0.013953), (43, 0.008543), (52, 0.002095), (60, 0.005155), (77, 0.005335), (78, 0.016446), (79, 0.015342)),
        ((1, 0.03109), (2, 0.010497), (3, 0.012885), (4, 0.005463), (6, 0.006838), (9, 0.021694), (15, 0.065255), (17, 0.049995), (18, 0.027226), (19, 0.016275), (20, 0.041322), (22, 0.038656), (23, 0.035714), (25, 0.021247), (26, 0.005721), (27, 0.021698), (28, 0.025056), (29, 0.007268), (31, 0.01095), (33, 0.00346), (34, 0.018868), (36, 0.010497), (37, 0.017544), (38, 0.006747), (39, 0.013209), (41, 0.016935), (42, 0.021696), (43, 0.008028), (44, 0.00277), (55, 0.002272), (61, 0.012823), (65, 0.00978), (66, 0.004777), (67, 0.014806), (68, 0.021247), (73, 0.00868), (75, 0.016275), (76, 0.000881), (77, 0.012885), (78, 0.02467), (79, 0.01842), (82, 0.014399), (83, 0.002363)),
        ((1, 0.005422), (3, 0.016173), (4, 0.008487), (10, 0.020916), (11, 0.043056), (12, 0.010474), (13, 0.058618), (14, 0.065255), (17, 0.023859), (18, 0.01172), (19, 0.010157), (20, 0.013699), (26, 0.001497), (27, 0.014243), (28, 0.003208), (30, 0.009264), (33, 0.012464), (34, 0.013807), (36, 0.004322), (38, 0.006641), (39, 0.020112), (43, 0.017008), (44, 0.000477), (50, 0.017465), (51, 0.015593), (53, 0.0031), (59, 0.002445), (61, 0.00868), (65, 0.011321), (67, 0.030456), (75, 0.00277), (76, 0.009957), (77, 0.008487), (79, 0.032667), (81, 0.02445), (82, 0.003434), (83, 0.007858)),
        ((1, 0.013166), (2, 0.008439), (4, 0.008394), (5, 0.001156), (6, 0.035189), (7, 0.002345), (9, 0.014648), (10, 0.000984), (12, 0.008416), (13, 0.005335), (14, 0.049995), (15, 0.023859), (18, 0.019918), (19, 0.016919), (20, 0.003861), (21, 0.012797), (22, 0.003975), (23, 0.004039), (25, 0.015507), (26, 0.028074), (27, 0.009881), (29, 0.001028), (30, 0.009859), (31, 0.008362), (33, 0.012588), (34, 0.034735), (38, 0.001239), (39, 0.002702), (43, 0.006948), (44, 0.011276), (50, 0.000801), (51, 0.015827), (54, 0.006946), (57, 0.002414), (58, 0.015696), (59, 0.005525), (60, 0.001239), (65, 0.001198), (68, 0.001239), (73, 0.006945), (74, 0.014039), (75, 0.009795), (76, 0.012976), (79, 0.017139), (81, 0.012628), (82, 0.01841), (83, 0.012885)),
        ((1, 0.009808), (2, 0.038126), (3, 0.027124), (4, 0.012711), (5, 0.012711), (6, 0.047947), (9, 0.002749), (10, 0.015928), (14, 0.027226), (15, 0.01172), (17, 0.019918), (19, 0.012433), (21, 0.025455), (22, 0.011171), (23, 0.045016), (25, 0.018279), (26, 0.0084), (28, 0.001733), (29, 0.029808), (30, 0.007546), (31, 0.003338), (37, 0.000898), (39, 0.004608), (41, 0.005686), (42, 0.001862), (43, 0.007203), (44, 0.000898), (45, 0.000787), (47, 0.011933), (49, 0.009852), (50, 0.01172), (52, 0.01575), (54, 0.010654), (55, 0.013075), (57, 0.018931), (66, 0.013209), (70, 0.018182), (74, 0.009142), (75, 0.005329), (76, 0.014108), (79, 0.01479), (81, 0.007874), (82, 0.021623), (83, 0.005118)),
        ((1, 0.018587), (2, 0.042611), (3, 0.002671), (4, 0.009795), (5, 0.03829), (6, 0.013084), (9, 0.02403), (10, 0.028105), (13, 0.002854), (14, 0.016275), (15, 0.010157), (17, 0.016919), (18, 0.012433), (20, 0.02963), (23, 0.007143), (25, 0.018535), (26, 0.005254), (27, 0.008977), (29, 0.018051), (36, 0.013599), (37, 0.011556), (42, 0.009107), (46, 0.00088), (58, 0.005367), (66, 0.000886), (73, 0.00088), (76, 0.014599), (77, 0.024043), (81, 0.014085), (82, 0.00885), (83, 0.001808)),
        ((2, 0.029289), (3, 0.050193), (4, 0.027027), (5, 0.011583), (6, 0.009709), (10, 0.029289), (11, 0.012146), (12, 0.02008), (13, 0.048128), (14, 0.041322), (15, 0.013699), (17, 0.003861), (19, 0.02963), (21, 0.015873), (22, 0.029046), (23, 0.03125), (25, 0.011152), (26, 0.007194), (27, 0.064), (28, 0.044898), (29, 0.008197), (30, 0.011765), (33, 0.01083), (34, 0.031746), (36, 0.020921), (37, 0.003831), (39, 0.010909), (44, 0.003831), (46, 0.003663), (47, 0.007576), (58, 0.023622), (63, 0.008333), (73, 0.003663), (75, 0.014815), (78, 0.025862), (81, 0.007353), (82, 0.015038)),
        ((1, 0.008), (2, 0.005693), (3, 0.012797), (5, 0.02011), (6, 0.02834), (7, 0.007874), (10, 0.005693), (11, 0.020561), (12, 0.01676), (13, 0.014737), (17, 0.012797), (18, 0.025455), (20, 0.015873), (22, 0.024575), (23, 0.022059), (25, 0.008977), (26, 0.010601), (27, 0.018587), (28, 0.009381), (29, 0.022556), (30, 0.020258), (31, 0.010676), (33, 0.00885), (34, 0.037037), (35, 0.067164), (37, 0.016393), (38, 0.001795), (41, 0.014925), (45, 0.014286), (50, 0.005917), (53, 0.007194), (63, 0.007576), (69, 0.019011), (71, 0.001934), (73, 0.023173), (76, 0.015385), (79, 0.020258), (81, 0.007143), (82, 0.00361)),
        ((1, 0.0268), (2, 0.031303), (3, 0.011408), (4, 0.018841), (6, 0.044663), (9, 0.009515), (13, 0.018325), (14, 0.038656), (17, 0.003975), (18, 0.011171), (20, 0.029046), (21, 0.024575), (23, 0.049085), (26, 0.012336), (27, 0.027335), (28, 0.015122), (29, 0.035783), (30, 0.039249), (31, 0.01026), (33, 0.017162), (34, 0.024575), (36, 0.007866), (37, 0.001387), (38, 0.013052), (39, 0.005416), (41, 0.022486), (45, 0.020016), (49, 0.018182), (53, 0.010709), (54, 0.013547), (60, 0.013052), (63, 0.002597), (65, 0.001224), (68, 0.005795), (71, 0.005957), (73, 0.007938), (76, 0.045515), (77, 0.018841), (78, 0.013699), (82, 0.005989), (83, 0.015398)),
        ((1, 0.032343), (2, 0.01586), (3, 0.018578), (5, 0.004039), (6, 0.030035), (9, 0.011246), (10, 0.008347), (14, 0.035714), (17, 0.004039), (18, 0.045016), (19, 0.007143), (20, 0.03125), (21, 0.022059), (22, 0.049085), (25, 0.012719), (26, 0.005486), (27, 0.011475), (28, 0.010744), (29, 0.016556), (30, 0.019512), (31, 0.013407), (39, 0.007874), (44, 0.007246), (45, 0.010285), (63, 0.01), (66, 0.0136), (71, 0.006791), (73, 0.011848), (75, 0.014286), (76, 0.004223), (77, 0.018578), (78, 0.011824), (82, 0.022364), (83, 0.013289)),
        ((3, 0.008373), (4, 0.015507), (5, 0.001239), (7, 0.015017), (9, 0.013089), (11, 0.011475), (14, 0.021247), (17, 0.015507), (18, 0.018279), (19, 0.018535), (20, 0.011152), (21, 0.008977), (23, 0.012719), (28, 0.009589), (29, 0.037588), (30, 0.018947), (31, 0.001075), (34, 0.001795), (35, 0.019624), (39, 0.002001), (42, 0.006393), (46, 0.000147), (51, 0.008643), (52, 0.001623), (54, 0.004985), (55, 0.007612), (57, 0.017215), (58, 0.010851), (66, 0.006848), (67, 0.012092), (74, 0.000147), (75, 0.004413), (76, 0.004473), (77, 0.008373), (79, 0.004625), (82, 0.007776)),
        ((1, 0.012415), (2, 0.01136), (6, 0.002209), (14, 0.005721), (15, 0.001497), (17, 0.028074), (18, 0.0084), (19, 0.005254), (20, 0.007194), (21, 0.010601), (22, 0.012336), (23, 0.005486), (28, 9.8e-05), (30, 0.012055), (31, 0.000194), (33, 0.001648), (34, 0.010601), (36, 0.004248), (39, 0.007668), (41, 0.001574), (42, 0.001783), (44, 0.007916), (45, 0.006221), (53, 0.004285), (54, 0.001855), (55, 0.001994), (58, 0.011571), (65, 0.002342), (69, 0.010872), (70, 0.003534), (73, 0.013699), (74, 0.006704), (75, 0.012259), (76, 0.022205), (81, 0.006221), (82, 0.017352), (83, 0.005721)),
        ((2, 0.015789), (3, 0.009881), (4, 0.031883), (5, 0.024549), (6, 0.027469), (10, 0.00053), (11, 0.008864), (13, 0.002959), (14, 0.021698), (15, 0.014243), (17, 0.009881), (19, 0.008977), (20, 0.064), (21, 0.018587), (22, 0.027335), (23, 0.011475), (28, 0.027536), (29, 0.018044), (30, 0.02433), (33, 0.014864), (34, 0.033457), (39, 0.018279), (41, 0.003331), (45, 0.005564), (51, 0.010491), (54, 0.010209), (58, 0.007612), (63, 0.002538), (66, 0.013997), (73, 0.036001), (74, 0.007438), (76, 0.024936), (77, 0.009881), (78, 0.032674)),
        ((1, 0.035493), (2, 0.026035), (3, 0.024681), (4, 0.009903), (5, 0.002514), (7, 0.013699), (9, 0.00079), (12, 0.002676), (14, 0.025056), (15, 0.003208), (18, 0.001733), (20, 0.044898), (21, 0.009381), (22, 0.015122), (23, 0.010744), (25, 0.009589), (26, 9.8e-05), (27, 0.027536), (29, 0.05243), (30, 0.023451), (31, 0.020165), (33, 0.033509), (34, 0.054409), (35, 0.000421), (36, 0.018305), (37, 0.021609), (39, 0.000843), (41, 0.007993), (45, 0.001596), (46, 0.0109), (47, 0.00609), (50, 0.019416), (58, 0.013801), (61, 9.8e-05), (65, 0.00609), (68, 0.009589), (70, 0.016886), (71, 0.010935), (73, 0.03957), (74, 0.0109), (75, 0.02615), (78, 0.002182), (81, 0.023144), (82, 0.032239)),
        ((1, 0.039558), (2, 0.015716), (6, 0.024755), (11, 0.003592), (14, 0.007268), (17, 0.001028), (18, 0.029808), (19, 0.018051), (20, 0.008197), (21, 0.022556), (22, 0.035783), (23, 0.016556), (25, 0.037588), (27, 0.018044), (28, 0.05243), (30, 0.074051), (31, 0.02664), (33, 0.018764), (34, 0.045113), (35, 0.028668), (36, 0.046717), (37, 0.034991), (38, 0.015879), (39, 0.00734), (41, 0.021079), (43, 0.013379), (50, 0.007458), (59, 0.000799), (66, 0.007098), (67, 0.017315), (68, 0.008643), (70, 0.015038), (71, 0.015656), (73, 0.010161), (74, 0.010161), (75, 0.039711), (77, 0.038027), (78, 0.022767), (81, 0.000799), (82, 0.016592), (83, 0.03805)),
        ((1, 0.012508), (6, 0.001664), (12, 0.007867), (15, 0.009264), (17, 0.009859), (18, 0.007546), (20, 0.011765), (21, 0.020258), (22, 0.039249), (23, 0.019512), (25, 0.018947), (26, 0.012055), (27, 0.02433), (28, 0.023451), (29, 0.074051), (31, 0.019828), (33, 0.038728), (34, 0.049724), (35, 0.035847), (36, 0.013338), (37, 0.020408), (38, 0.011786), (39, 0.01433), (41, 0.013597), (45, 0.016627), (46, 0.011116), (47, 0.01083), (49, 0.0), (50, 0.009264), (51, 0.029247), (54, 0.001959), (62, 0.000991), (63, 0.007519), (67, 0.023655), (70, 0.020258), (71, 0.034329), (74, 0.004002), (75, 0.027703), (76, 0.001271), (79, 0.010536), (81, 0.023753), (82, 0.021289)),
        ((1, 0.011793), (2, 0.008878), (3, 0.015445), (4, 0.001279), (6, 0.007556), (7, 0.024666), (9, 0.023284), (10, 0.008878), (11, 9.9e-05), (12, 0.008618), (14, 0.01095), (17, 0.008362), (18, 0.003338), (21, 0.010676), (22, 0.01026), (23, 0.013407), (25, 0.001075), (26, 0.000194), (28, 0.020165), (29, 0.02664), (30, 0.019828), (34, 0.017794), (35, 0.022205), (36, 0.001696), (38, 0.001075), (41, 0.000793), (57, 0.00261), (61, 0.000194), (62, 0.003054), (63, 0.016746), (65, 0.004706), (66, 0.012442), (67, 0.004022), (68, 0.015144), (69, 0.008185), (70, 0.010676), (71, 0.009141), (73, 0.0038), (74, 0.0038), (75, 0.008787), (76, 0.00401), (77, 0.008362), (78, 0.011227), (81, 0.00312), (82, 0.00607), (83, 0.018117)),
        ((7, 0.020212), (10, 0.001832), (12, 0.007239), (13, 0.002089), (14, 0.00346), (15, 0.012464), (17, 0.012588), (20, 0.01083), (21, 0.00885), (22, 0.017162), (26, 0.001648), (27, 0.014864), (28, 0.033509), (29, 0.018764), (30, 0.038728), (34, 0.023009), (35, 0.020891), (36, 0.016091), (37, 0.006605), (39, 0.021027), (41, 0.006701), (43, 0.001288), (45, 0.012451), (46, 0.005979), (47, 0.001172), (49, 0.009501), (50, 0.005256), (51, 0.004543), (54, 0.000635), (57, 0.00408), (60, 0.010864), (61, 0.001648), (62, 0.000302), (65, 0.001172), (67, 0.000635), (68, 0.003849), (71, 0.017872), (73, 0.012979), (74, 0.005979), (75, 0.011394), (81, 0.005447)),
        ((1, 0.024), (2, 0.005693), (4, 0.02011), (6, 0.004049), (7, 0.015748), (11, 0.005607), (14, 0.018868), (15, 0.013807), (17, 0.034735), (20, 0.031746), (21, 0.037037), (22, 0.024575), (25, 0.001795), (26, 0.010601), (27, 0.033457), (28, 0.054409), (29, 0.045113), (30, 0.049724), (31, 0.017794), (33, 0.023009), (35, 0.022388), (36, 0.020873), (37, 0.038251), (38, 0.008977), (39, 0.012433), (41, 0.007463), (43, 0.003802), (46, 0.016043), (50, 0.005917), (51, 0.007519), (54, 0.005445), (58, 0.01107), (62, 0.004016), (68, 0.016158), (70, 0.02963), (71, 0.009671), (73, 0.008913), (74, 0.001783), (75, 0.010753), (77, 0.005484), (79, 0.012891), (81, 0.007143), (82, 0.00361)),
        ((4, 0.006947), (6, 0.03483), (7, 0.016874), (9, 0.00446), (11, 0.012149), (12, 0.008761), (13, 0.025395), (21, 0.067164), (25, 0.019624), (28, 0.000421), (29, 0.028668), (30, 0.035847), (31, 0.022205), (33, 0.020891), (34, 0.022388), (36, 0.002983), (37, 0.062271), (38, 0.005206), (41, 0.014214), (42, 0.054264), (45, 0.003981), (52, 0.001206), (53, 0.00321), (59, 0.011146), (60, 0.012415), (63, 0.012755), (67, 0.000405), (70, 0.007463), (71, 0.004789), (75, 0.007194), (77, 0.021659), (79, 0.00618), (81, 0.018312), (83, 0.016935)),
        ((1, 0.00964), (2, 0.010417), (5, 0.015894), (6, 0.000837), (7, 0.015399), (9, 0.026465), (10, 0.002563), (13, 0.024007), (14, 0.010497), (15, 0.004322), (19, 0.013599), (20, 0.020921), (22, 0.007866), (26, 0.004248), (28, 0.018305), (29, 0.046717), (30, 0.013338), (31, 0.001696), (33, 0.016091), (34, 0.020873), (35, 0.002983), (37, 0.035698), (38, 0.011159), (39, 0.011294), (41, 0.041329), (43, 0.023515), (57, 0.002124), (58, 0.003251), (59, 0.025662), (60, 0.011159), (61, 0.01136), (63, 0.013055), (66, 0.015962), (68, 0.01843), (73, 0.00645), (74, 0.013649), (78, 0.007559), (79, 0.013338), (83, 0.010497)),
        ((1, 0.00578), (2, 0.006027), (4, 0.00406), (7, 0.017143), (13, 0.0005), (14, 0.017544), (18, 0.000898), (19, 0.011556), (20, 0.003831), (21, 0.016393), (22, 0.001387), (28, 0.021609), (29, 0.034991), (30, 0.020408), (33, 0.006605), (34, 0.038251), (35, 0.062271), (36, 0.035698), (38, 0.024477), (41, 0.032967), (42, 0.043884), (43, 0.004643), (46, 0.001328), (47, 0.003584), (57, 0.000939), (60, 0.010236), (62, 0.002899), (66, 0.004922), (68, 0.003115), (70, 0.001821), (75, 0.011556), (77, 0.00406), (78, 0.011236), (81, 0.021277), (83, 0.00277)),
        ((3, 0.001239), (4, 0.001239), (5, 0.001239), (7, 0.007612), (9, 0.013089), (13, 0.012808), (14, 0.006747), (15, 0.006641), (17, 0.001239), (21, 0.001795), (22, 0.013052), (29, 0.015879), (30, 0.011786), (31, 0.001075), (34, 0.008977), (35, 0.005206), (36, 0.011159), (37, 0.024477), (41, 0.026832), (43, 0.002931), (45, 0.013312), (46, 0.000147), (63, 0.004843), (65, 0.013018), (68, 0.003485), (69, 0.024765), (70, 0.001795), (71, 0.00158), (73, 0.000147), (74, 0.000147), (75, 0.004413), (76, 0.026434), (77, 0.022641)),
        ((3, 0.009775), (11, 0.016384), (13, 0.013953), (14, 0.013209), (15, 0.020112), (17, 0.002702), (18, 0.004608), (20, 0.010909), (22, 0.005416), (23, 0.007874), (25, 0.002001), (26, 0.007668), (27, 0.018279), (28, 0.000843), (29, 0.00734), (30, 0.01433), (33, 0.021027), (34, 0.012433), (36, 0.011294), (41, 0.024139), (42, 0.000898), (43, 0.017823), (45, 0.010912), (46, 0.011538), (50, 0.012853), (51, 0.000198), (52, 0.014807), (54, 0.005242), (61, 0.000679), (66, 0.006508), (67, 0.005242), (68, 0.009029), (70, 0.005329), (71, 0.004858), (73, 0.004528), (75, 0.002634), (77, 0.002702), (78, 0.006797), (79, 0.01433), (82, 0.007139), (83, 0.013209)),
        ((2, 0.002983), (14, 0.016935), (18, 0.005686), (21, 0.014925), (22, 0.022486), (26, 0.001574), (27, 0.003331), (28, 0.007993), (29, 0.021079), (30, 0.013597), (31, 0.000793), (33, 0.006701), (34, 0.007463), (35, 0.014214), (36, 0.041329), (37, 0.032967), (38, 0.026832), (39, 0.024139), (42, 0.03876), (43, 0.054654), (45, 0.003981), (46, 0.013111), (47, 0.024272), (57, 0.022589), (58, 0.026424), (59, 0.003981), (65, 0.009709), (66, 0.033508), (67, 0.014998), (69, 0.031597), (71, 0.020461), (73, 0.013111), (74, 0.005959), (76, 0.019031), (83, 0.024555)),
        ((2, 0.004975), (9, 0.006369), (11, 0.004859), (14, 0.021696), (18, 0.001862), (19, 0.009107), (25, 0.006393), (26, 0.001783), (35, 0.054264), (37, 0.043884), (39, 0.000898), (41, 0.03876), (45, 0.007246), (49, 0.007937), (50, 0.034921), (52, 0.008264), (54, 0.019499), (55, 0.07173), (57, 0.034765), (58, 0.04), (59, 0.007246), (61, 0.008913), (63, 0.02381), (65, 0.007407), (66, 0.017544), (70, 0.019157), (74, 0.017164), (75, 0.009107), (77, 0.008451), (83, 0.005917)),
        ((10, 0.039265), (11, 0.005887), (12, 0.018844), (13, 0.008543), (14, 0.008028), (15, 0.017008), (17, 0.006948), (18, 0.007203), (29, 0.013379), (33, 0.001288), (34, 0.003802), (36, 0.023515), (37, 0.004643), (38, 0.002931), (39, 0.017823), (41, 0.054654), (44, 0.012071), (46, 0.012909), (47, 0.04914), (49, 0.026178), (51, 0.005611), (55, 0.00323), (57, 0.023099), (58, 0.009036), (59, 0.039326), (61, 0.003756), (62, 0.00784), (63, 0.034031), (65, 0.004914), (66, 0.022159), (67, 0.02453), (71, 0.020353), (75, 0.005445), (79, 0.019176), (81, 0.00321), (82, 0.002648), (83, 0.015839)),
        ((7, 0.024762), (9, 0.006686), (10, 0.006027), (11, 0.002293), (12, 0.012346), (14, 0.00277), (15, 0.000477), (17, 0.011276), (18, 0.000898), (20, 0.003831), (23, 0.007246), (26, 0.007916), (43, 0.012071), (45, 0.007092), (46, 0.008411), (49, 0.014815), (50, 0.000477), (51, 0.012891), (52, 0.007586), (53, 0.016043), (54, 0.009421), (55, 0.04), (57, 0.023474), (59, 0.014184), (62, 0.002899), (65, 0.003584), (66, 0.012081), (67, 0.009421), (73, 0.008411), (76, 0.003745), (79, 0.005896)),
        ((3, 0.005525), (4, 0.012628), (18, 0.000787), (21, 0.014286), (22, 0.020016), (23, 0.010285), (26, 0.006221), (27, 0.005564), (28, 0.001596), (30, 0.016627), (33, 0.012451), (35, 0.003981), (38, 0.013312), (39, 0.010912), (41, 0.003981), (42, 0.007246), (44, 0.007092), (46, 0.009368), (47, 0.002358), (49, 0.004808), (50, 0.00978), (51, 0.007987), (52, 0.011765), (53, 0.005486), (54, 0.015736), (55, 0.017915), (57, 0.004039), (58, 0.001585), (59, 0.008594), (60, 0.006265), (62, 0.002463), (65, 0.002358), (67, 0.001574), (71, 0.010509), (73, 0.002342), (81, 0.015625)),
        ((4, 0.014039), (5, 0.006945), (9, 0.005181), (12, 0.006699), (19, 0.00088), (20, 0.003663), (25, 0.000147), (28, 0.0109), (30, 0.011116), (33, 0.005979), (34, 0.016043), (37, 0.001328), (38, 0.000147), (39, 0.011538), (41, 0.013111), (43, 0.012909), (44, 0.008411), (45, 0.009368), (47, 0.024735), (49, 0.007194), (51, 0.010161), (52, 0.005729), (54, 0.009872), (55, 0.006695), (58, 0.003263), (60, 0.007188), (62, 0.00644), (66, 0.004267), (67, 0.016944), (70, 0.023173), (71, 0.01345), (73, 0.017113), (77, 0.006945), (78, 0.001206), (83, 0.00868)),
        ((3, 0.001198), (4, 0.015569), (5, 0.001198), (12, 0.025455), (18, 0.011933), (20, 0.007576), (28, 0.00609), (30, 0.01083), (33, 0.001172), (37, 0.003584), (41, 0.024272), (43, 0.04914), (45, 0.002358), (46, 0.024735), (49, 0.022059), (50, 0.018868), (53, 0.004739), (54, 0.020262), (55, 0.012563), (57, 0.054591), (58, 0.031325), (59, 0.016509), (61, 0.002342), (62, 0.022901), (66, 0.015458), (67, 0.013111), (70, 0.014493), (71, 0.008696), (74, 0.010601), (79, 0.00361)),
        ((7, 0.002747), (9, 0.022099), (11, 0.002558), (12, 0.022901), (18, 0.009852), (22, 0.018182), (30, 0.0), (33, 0.009501), (42, 0.007937), (43, 0.026178), (44, 0.014815), (45, 0.004808), (46, 0.007194), (47, 0.022059), (50, 0.033058), (51, 0.072165), (52, 0.043796), (53, 0.038835), (54, 0.034398), (55, 0.06044), (57, 0.053476), (58, 0.012563), (59, 0.004808), (60, 0.041162), (61, 0.018957), (62, 0.042373), (63, 0.046875), (65, 0.036765), (68, 0.012107), (69, 0.002618), (70, 0.007576), (82, 0.019512)),
        ((4, 0.016173), (6, 0.016315), (9, 0.022256), (10, 0.012619), (12, 0.026434), (15, 0.017465), (17, 0.000801), (18, 0.01172), (21, 0.005917), (28, 0.019416), (29, 0.007458), (30, 0.009264), (33, 0.005256), (34, 0.005917), (39, 0.012853), (42, 0.034921), (44, 0.000477), (45, 0.00978), (47, 0.018868), (49, 0.033058), (51, 0.080678), (52, 0.051797), (53, 0.017979), (54, 0.038031), (55, 0.111444), (58, 0.029038), (59, 0.02445), (60, 0.006641), (61, 0.00868), (62, 0.043376), (65, 0.026415), (68, 0.006641), (70, 0.013807), (73, 0.013248), (74, 0.020557), (75, 0.00277), (83, 0.024257)),
        ((4, 0.008428), (9, 0.021296), (10, 0.000215), (11, 0.003592), (12, 0.023344), (15, 0.015593), (17, 0.015827), (25, 0.008643), (27, 0.010491), (30, 0.029247), (33, 0.004543), (34, 0.007519), (39, 0.000198), (43, 0.005611), (44, 0.012891), (45, 0.007987), (46, 0.010161), (49, 0.072165), (50, 0.080678), (52, 0.033313), (53, 0.028203), (54, 0.024649), (55, 0.050496), (57, 0.018038), (58, 0.012058), (59, 0.029553), (60, 0.023116), (62, 0.018031), (63, 0.025773), (65, 0.012195), (67, 0.002648), (68, 0.015879), (69, 0.013379), (70, 0.030075), (71, 0.023594), (73, 0.002989), (74, 0.024507), (79, 0.006845), (82, 0.002023), (83, 0.022659)),
        ((3, 0.005515), (5, 0.005515), (9, 0.013391), (11, 0.007396), (12, 0.002259), (13, 0.002095), (18, 0.01575), (25, 0.001623), (35, 0.001206), (39, 0.014807), (42, 0.008264), (44, 0.007586), (45, 0.011765), (46, 0.005729), (49, 0.043796), (50, 0.051797), (51, 0.033313), (53, 0.021855), (54, 0.009648), (55, 0.04537), (57, 0.011374), (60, 0.001623), (61, 0.017831), (62, 0.012253), (63, 0.007299), (65, 0.017794), (68, 0.001623), (75, 0.002653), (79, 0.001347), (83, 0.009408)),
        ((1, 0.003331), (2, 0.001417), (6, 0.004608), (10, 0.001417), (11, 0.002209), (15, 0.0031), (21, 0.007194), (22, 0.010709), (26, 0.004285), (35, 0.00321), (44, 0.016043), (45, 0.005486), (47, 0.004739), (49, 0.038835), (50, 0.017979), (51, 0.028203), (52, 0.021855), (54, 0.010872), (55, 0.033856), (57, 0.021659), (58, 0.009174), (59, 0.005486), (60, 0.016703), (62, 0.01627), (63, 0.009709), (65, 0.011848), (67, 0.003756), (68, 0.009629), (69, 0.014998), (70, 0.007194), (71, 0.005932), (76, 0.016313), (79, 0.002989)),
        ((17, 0.006946), (18, 0.010654), (22, 0.013547), (25, 0.004985), (26, 0.001855), (27, 0.010209), (30, 0.001959), (33, 0.000635), (34, 0.005445), (39, 0.005242), (42, 0.019499), (44, 0.009421), (45, 0.015736), (46, 0.009872), (47, 0.020262), (49, 0.034398), (50, 0.038031), (51, 0.024649), (52, 0.009648), (53, 0.010872), (55, 0.046859), (57, 0.021881), (58, 0.007945), (59, 0.001574), (62, 0.011179), (65, 0.005959), (66, 5e-05), (67, 0.004724), (74, 0.002799)),
        ((12, 0.012608), (14, 0.002272), (18, 0.013075), (25, 0.007612), (26, 0.001994), (42, 0.07173), (43, 0.00323), (44, 0.04), (45, 0.017915), (46, 0.006695), (47, 0.012563), (49, 0.06044), (50, 0.111444), (51, 0.050496), (52, 0.04537), (53, 0.033856), (54, 0.046859), (57, 0.056694), (58, 0.046461), (59, 0.032573), (60, 0.007612), (61, 0.023534), (62, 0.066874), (65, 0.005025), (66, 0.008554), (67, 0.001471), (69, 0.00323), (74, 0.021302)),
        ((7, 0.030967), (10, 0.002124), (17, 0.002414), (18, 0.018931), (25, 0.017215), (31, 0.00261), (33, 0.00408), (36, 0.002124), (37, 0.000939), (41, 0.022589), (42, 0.034765), (43, 0.023099), (44, 0.023474), (45, 0.004039), (47, 0.054591), (49, 0.053476), (51, 0.018038), (52, 0.011374), (53, 0.021659), (54, 0.021881), (55, 0.056694), (58, 0.002345), (59, 0.018578), (61, 0.028379), (62, 0.058475), (63, 0.07754), (66, 0.020307), (67, 0.006948), (68, 0.002547), (69, 0.023099), (74, 0.028692), (75, 0.005484), (76, 0.028362), (79, 0.005419)),
        ((2, 0.018353), (3, 0.015696), (12, 0.013292), (17, 0.015696), (19, 0.005367), (20, 0.023622), (25, 0.010851), (26, 0.011571), (27, 0.007612), (28, 0.013801), (34, 0.01107), (36, 0.003251), (41, 0.026424), (42, 0.04), (43, 0.009036), (45, 0.001585), (46, 0.003263), (47, 0.031325), (49, 0.012563), (50, 0.029038), (51, 0.012058), (53, 0.009174), (54, 0.007945), (55, 0.046461), (57, 0.002345), (59, 0.008716), (62, 0.028534), (63, 0.005025), (65, 0.009639), (66, 0.018543), (70, 0.01845), (71, 0.000534), (73, 0.031741), (74, 0.038861), (77, 0.008405)),
        ((12, 0.004773), (15, 0.002445), (17, 0.005525), (29, 0.000799), (35, 0.011146), (36, 0.025662), (41, 0.003981), (42, 0.007246), (43, 0.039326), (44, 0.014184), (45, 0.008594), (47, 0.016509), (49, 0.004808), (50, 0.02445), (51, 0.029553), (53, 0.005486), (54, 0.001574), (55, 0.032573), (57, 0.018578), (58, 0.008716), (60, 0.006265), (61, 0.006221), (62, 0.024631), (63, 0.019231), (65, 0.016509), (66, 0.010212), (67, 0.022817), (68, 0.006265), (71, 0.010509), (79, 0.002375), (83, 0.0136)),
        ((2, 0.011159), (4, 0.015507), (6, 0.008864), (7, 0.000206), (9, 0.005668), (12, 0.013353), (13, 0.005155), (17, 0.001239), (22, 0.013052), (33, 0.010864), (35, 0.012415), (36, 0.011159), (37, 0.010236), (45, 0.006265), (46, 0.007188), (49, 0.041162), (50, 0.006641), (51, 0.023116), (52, 0.001623), (53, 0.016703), (55, 0.007612), (59, 0.006265), (61, 0.00477), (62, 0.012777), (65, 0.020118), (68, 0.031755), (69, 0.010209), (70, 0.008977), (71, 0.008922), (73, 0.000147), (74, 0.000147), (79, 0.004625), (82, 0.021951)),
        ((9, 0.000998), (11, 0.008167), (12, 0.016219), (14, 0.012823), (15, 0.00868), (28, 9.8e-05), (31, 0.000194), (33, 0.001648), (36, 0.01136), (39, 0.000679), (42, 0.008913), (43, 0.003756), (47, 0.002342), (49, 0.018957), (50, 0.00868), (52, 0.017831), (55, 0.023534), (57, 0.028379), (59, 0.006221), (60, 0.00477), (62, 0.018641), (63, 0.011848), (66, 0.009851), (68, 0.011778), (74, 0.006704), (75, 0.005254), (76, 0.000793)),
        ((12, 0.007121), (30, 0.000991), (31, 0.003054), (33, 0.000302), (34, 0.004016), (37, 0.002899), (43, 0.00784), (44, 0.002899), (45, 0.002463), (46, 0.00644), (47, 0.022901), (49, 0.042373), (50, 0.043376), (51, 0.018031), (52, 0.012253), (53, 0.01627), (54, 0.011179), (55, 0.066874), (57, 0.058475), (58, 0.028534), (59, 0.024631), (60, 0.012777), (61, 0.018641), (63, 0.016949), (65, 0.007634), (67, 0.011179), (69, 0.00784), (70, 0.028112), (71, 0.000372), (74, 0.013799), (76, 0.032258), (81, 0.002463)),
        ((2, 0.005222), (3, 0.01737), (12, 0.0), (20, 0.008333), (21, 0.007576), (22, 0.002597), (23, 0.01), (27, 0.002538), (30, 0.007519), (31, 0.016746), (35, 0.012755), (36, 0.013055), (38, 0.004843), (42, 0.02381), (43, 0.034031), (49, 0.046875), (51, 0.025773), (52, 0.007299), (53, 0.009709), (57, 0.07754), (58, 0.005025), (59, 0.019231), (61, 0.011848), (62, 0.016949), (65, 0.022059), (66, 0.00978), (67, 0.004914), (68, 0.012107), (69, 0.034031), (73, 0.014388), (75, 0.028986), (76, 0.042553), (78, 0.00266), (81, 0.012019), (82, 0.004878), (83, 0.005181)),
        ((1, 0.002538), (3, 0.001198), (4, 0.008383), (9, 0.002519), (12, 0.018182), (14, 0.00978), (15, 0.011321), (17, 0.001198), (22, 0.001224), (26, 0.002342), (28, 0.00609), (31, 0.004706), (33, 0.001172), (38, 0.013018), (41, 0.009709), (42, 0.007407), (43, 0.004914), (44, 0.003584), (45, 0.002358), (49, 0.036765), (50, 0.026415), (51, 0.012195), (52, 0.017794), (53, 0.011848), (54, 0.005959), (55, 0.005025), (58, 0.009639), (59, 0.016509), (60, 0.020118), (62, 0.007634), (63, 0.022059), (66, 0.008323), (67, 0.013111), (68, 0.020118), (70, 0.007246), (71, 0.008696), (73, 0.010601), (77, 0.001198), (81, 0.009434), (82, 0.009501), (83, 0.017115)),
        ((1, 0.00653), (2, 0.001275), (9, 0.00617), (12, 0.005595), (14, 0.004777), (18, 0.013209), (19, 0.000886), (23, 0.0136), (25, 0.006848), (27, 0.013997), (29, 0.007098), (31, 0.012442), (36, 0.015962), (37, 0.004922), (39, 0.006508), (41, 0.033508), (42, 0.017544), (43, 0.022159), (44, 0.012081), (46, 0.004267), (47, 0.015458), (54, 5e-05), (55, 0.008554), (57, 0.020307), (58, 0.018543), (59, 0.010212), (61, 0.009851), (63, 0.00978), (65, 0.008323), (67, 0.007192), (68, 0.021036), (69, 0.014806), (70, 0.001808), (71, 0.004283), (73, 0.011329), (74, 0.025452), (77, 0.016991), (78, 0.000411), (83, 0.02673)),
        ((1, 0.006158), (3, 0.006946), (12, 0.00167), (14, 0.014806), (15, 0.030456), (25, 0.012092), (29, 0.017315), (30, 0.023655), (31, 0.004022), (33, 0.000635), (35, 0.000405), (39, 0.005242), (41, 0.014998), (43, 0.02453), (44, 0.009421), (45, 0.001574), (46, 0.016944), (47, 0.013111), (51, 0.002648), (53, 0.003756), (54, 0.004724), (55, 0.001471), (57, 0.006948), (59, 0.022817), (62, 0.011179), (63, 0.004914), (65, 0.013111), (66, 0.007192), (69, 0.017141), (71, 0.013135), (73, 0.002799), (74, 0.002799), (75, 0.01331), (76, 0.002069), (77, 0.006946), (79, 0.016423), (81, 0.001574), (82, 0.001288)),
        ((1, 0.02965), (2, 0.003888), (4, 0.001239), (6, 0.001356), (7, 0.000206), (9, 0.020509), (14, 0.021247), (17, 0.001239), (22, 0.005795), (28, 0.009589), (29, 0.008643), (31, 0.015144), (33, 0.003849), (34, 0.016158), (36, 0.01843), (37, 0.003115), (38, 0.003485), (39, 0.009029), (49, 0.012107), (50, 0.006641), (51, 0.015879), (52, 0.001623), (53, 0.009629), (57, 0.002547), (59, 0.006265), (60, 0.031755), (61, 0.011778), (63, 0.012107), (65, 0.020118), (66, 0.021036), (69, 0.024765), (70, 0.037702), (71, 0.00158), (73, 0.014229), (74, 0.007188), (76, 0.004473), (82, 0.000689), (83, 0.028497)),
        ((6, 0.02188), (21, 0.019011), (26, 0.010872), (31, 0.008185), (38, 0.024765), (41, 0.031597), (49, 0.002618), (51, 0.013379), (53, 0.014998), (55, 0.00323), (57, 0.023099), (60, 0.010209), (62, 0.00784), (63, 0.034031), (66, 0.014806), (67, 0.017141), (68, 0.024765), (70, 0.003802), (71, 0.012257), (73, 0.027319), (74, 0.020114), (76, 0.036574), (79, 0.019176), (83, 0.000217)),
        ((1, 0.04), (2, 0.013283), (4, 0.005484), (6, 0.012146), (7, 0.023622), (18, 0.018182), (26, 0.003534), (28, 0.016886), (29, 0.015038), (30, 0.020258), (31, 0.010676), (34, 0.02963), (35, 0.007463), (37, 0.001821), (38, 0.001795), (39, 0.005329), (42, 0.019157), (46, 0.023173), (47, 0.014493), (49, 0.007576), (50, 0.013807), (51, 0.030075), (53, 0.007194), (58, 0.01845), (60, 0.008977), (62, 0.028112), (65, 0.007246), (66, 0.001808), (68, 0.037702), (69, 0.003802), (71, 0.032882), (73, 0.001783), (74, 0.044563), (75, 0.003584), (77, 0.02011), (78, 0.030769), (79, 0.005525), (81, 0.014286), (82, 0.039711), (83, 0.026415)),
        ((1, 0.043072), (2, 0.007453), (3, 0.000894), (4, 0.008463), (7, 0.001196), (9, 0.00277), (10, 0.039731), (21, 0.001934), (22, 0.005957), (23, 0.006791), (28, 0.010935), (29, 0.015656), (30, 0.034329), (31, 0.009141), (33, 0.017872), (34, 0.009671), (35, 0.004789), (38, 0.00158), (39, 0.004858), (41, 0.020461), (43, 0.020353), (45, 0.010509), (46, 0.01345), (47, 0.008696), (51, 0.023594), (53, 0.005932), (58, 0.000534), (59, 0.010509), (60, 0.008922), (62, 0.000372), (65, 0.008696), (66, 0.004283), (67, 0.013135), (68, 0.00158), (69, 0.012257), (70, 0.032882), (73, 0.020704), (75, 0.019213), (76, 0.00872), (77, 0.000894), (81, 0.003234), (83, 0.065143)),
        ((1, 0.007961), (3, 0.006945), (14, 0.00868), (17, 0.006945), (19, 0.00088), (20, 0.003663), (21, 0.023173), (22, 0.007938), (23, 0.011848), (26, 0.013699), (27, 0.036001), (28, 0.03957), (29, 0.010161), (31, 0.0038), (33, 0.012979), (34, 0.008913), (36, 0.00645), (38, 0.000147), (39, 0.004528), (41, 0.013111), (44, 0.008411), (45, 0.002342), (46, 0.017113), (50, 0.013248), (51, 0.002989), (58, 0.031741), (60, 0.000147), (63, 0.014388), (65, 0.010601), (66, 0.011329), (67, 0.002799), (68, 0.014229), (69, 0.027319), (70, 0.001783), (71, 0.020704), (74, 0.024133), (75, 0.00088), (76, 0.022919), (82, 0.012055), (83, 0.00868)),
        ((1, 0.000612), (2, 0.00645), (4, 0.006945), (5, 0.006945), (17, 0.014039), (18, 0.009142), (25, 0.000147), (26, 0.006704), (27, 0.007438), (28, 0.0109), (29, 0.010161), (30, 0.004002), (31, 0.0038), (33, 0.005979), (34, 0.001783), (36, 0.013649), (38, 0.000147), (41, 0.005959), (42, 0.017164), (47, 0.010601), (50, 0.020557), (51, 0.024507), (54, 0.002799), (55, 0.021302), (57, 0.028692), (58, 0.038861), (60, 0.000147), (61, 0.006704), (62, 0.013799), (66, 0.025452), (67, 0.002799), (68, 0.007188), (69, 0.020114), (70, 0.044563), (73, 0.024133), (75, 0.00088), (76, 0.008444), (79, 0.004002), (82, 0.004999)),
        ((1, 0.003717), (3, 0.002671), (5, 0.002671), (12, 0.000898), (14, 0.016275), (15, 0.00277), (17, 0.009795), (18, 0.005329), (20, 0.014815), (23, 0.014286), (25, 0.004413), (26, 0.012259), (28, 0.02615), (29, 0.039711), (30, 0.027703), (31, 0.008787), (33, 0.011394), (34, 0.010753), (35, 0.007194), (37, 0.011556), (38, 0.004413), (39, 0.002634), (42, 0.009107), (43, 0.005445), (50, 0.00277), (52, 0.002653), (57, 0.005484), (61, 0.005254), (63, 0.028986), (67, 0.01331), (70, 0.003584), (71, 0.019213), (73, 0.00088), (74, 0.00088), (78, 0.014599), (81, 0.021127), (82, 0.015929), (83, 0.001808)),
        ((1, 0.012621), (7, 0.038752), (9, 0.015224), (14, 0.000881), (15, 0.009957), (17, 0.012976), (18, 0.014108), (19, 0.014599), (21, 0.015385), (22, 0.045515), (23, 0.004223), (25, 0.004473), (26, 0.022205), (27, 0.024936), (30, 0.001271), (31, 0.00401), (38, 0.026434), (41, 0.019031), (44, 0.003745), (53, 0.016313), (57, 0.028362), (61, 0.000793), (62, 0.032258), (63, 0.042553), (67, 0.002069), (68, 0.004473), (69, 0.036574), (71, 0.00872), (73, 0.022919), (74, 0.008444)),
        ((1, 0.028707), (2, 0.008439), (4, 0.001156), (5, 0.008394), (6, 0.027345), (7, 0.017693), (10, 0.045716), (11, 0.005474), (13, 0.005335), (14, 0.012885), (15, 0.008487), (19, 0.024043), (22, 0.018841), (23, 0.018578), (25, 0.008373), (27, 0.009881), (29, 0.038027), (31, 0.008362), (34, 0.005484), (35, 0.021659), (37, 0.00406), (38, 0.022641), (39, 0.002702), (42, 0.008451), (46, 0.006945), (58, 0.008405), (65, 0.001198), (66, 0.016991), (67, 0.006946), (70, 0.02011), (71, 0.000894), (78, 0.028045), (81, 0.012628), (82, 0.00408), (83, 0.020307)),
        ((1, 0.030097), (2, 0.05558), (3, 0.005442), (5, 0.03558), (6, 0.02775), (7, 0.030246), (9, 0.00666), (10, 0.03157), (11, 0.03167), (13, 0.016446), (14, 0.02467), (20, 0.025862), (22, 0.013699), (23, 0.011824), (27, 0.032674), (28, 0.002182), (29, 0.022767), (31, 0.011227), (36, 0.007559), (37, 0.011236), (39, 0.006797), (46, 0.001206), (63, 0.00266), (66, 0.000411), (70, 0.030769), (75, 0.014599), (77, 0.028045), (79, 0.016518), (81, 0.020161), (82, 0.010664), (83, 0.02467)),
        ((1, 0.028308), (5, 0.017139), (6, 0.049584), (7, 0.003245), (9, 0.015304), (12, 0.007867), (13, 0.015342), (14, 0.01842), (15, 0.032667), (17, 0.017139), (18, 0.01479), (21, 0.020258), (25, 0.004625), (30, 0.010536), (34, 0.012891), (35, 0.00618), (36, 0.013338), (39, 0.01433), (43, 0.019176), (44, 0.005896), (47, 0.00361), (51, 0.006845), (52, 0.001347), (53, 0.002989), (57, 0.005419), (59, 0.002375), (60, 0.004625), (67, 0.016423), (69, 0.019176), (70, 0.005525), (74, 0.004002), (78, 0.016518), (81, 0.016627), (82, 0.042879), (83, 0.003434)),
        ((1, 0.011475), (2, 0.011227), (5, 0.005525), (6, 0.021417), (10, 0.00401), (15, 0.02445), (17, 0.012628), (18, 0.007874), (19, 0.014085), (20, 0.007353), (21, 0.007143), (26, 0.006221), (28, 0.023144), (29, 0.000799), (30, 0.023753), (31, 0.00312), (33, 0.005447), (34, 0.007143), (35, 0.018312), (37, 0.021277), (43, 0.00321), (45, 0.015625), (62, 0.002463), (63, 0.012019), (65, 0.009434), (67, 0.001574), (70, 0.014286), (71, 0.003234), (75, 0.021127), (77, 0.012628), (78, 0.020161), (79, 0.016627), (82, 0.025118)),
        ((1, 0.018044), (2, 0.01109), (3, 0.00408), (5, 0.011245), (6, 0.011198), (7, 0.012058), (9, 0.024797), (12, 0.01479), (14, 0.014399), (15, 0.003434), (17, 0.01841), (18, 0.021623), (19, 0.00885), (20, 0.015038), (21, 0.00361), (22, 0.005989), (23, 0.022364), (25, 0.007776), (26, 0.017352), (28, 0.032239), (29, 0.016592), (30, 0.021289), (31, 0.00607), (34, 0.00361), (39, 0.007139), (43, 0.002648), (49, 0.019512), (51, 0.002023), (60, 0.021951), (63, 0.004878), (65, 0.009501), (67, 0.001288), (68, 0.000689), (70, 0.039711), (73, 0.012055), (74, 0.004999), (75, 0.015929), (77, 0.00408), (78, 0.010664), (79, 0.042879), (81, 0.025118), (83, 0.021699)),
        ((1, 0.047796), (2, 0.010497), (3, 0.005463), (5, 0.020307), (6, 0.040792), (7, 0.01045), (9, 0.021694), (10, 0.033871), (11, 0.014536), (14, 0.002363), (15, 0.007858), (17, 0.012885), (18, 0.005118), (19, 0.001808), (22, 0.015398), (23, 0.013289), (26, 0.005721), (29, 0.03805), (31, 0.018117), (35, 0.016935), (36, 0.010497), (37, 0.00277), (39, 0.013209), (41, 0.024555), (42, 0.005917), (43, 0.015839), (46, 0.00868), (50, 0.024257), (51, 0.022659), (52, 0.009408), (59, 0.0136), (63, 0.005181), (65, 0.017115), (66, 0.02673), (68, 0.028497), (69, 0.000217), (70, 0.026415), (71, 0.065143), (73, 0.00868), (75, 0.001808), (77, 0.020307), (78, 0.02467), (79, 0.003434), (82, 0.021699)),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        ()
    ),
    "formations": (
        (1, 2, 7, 17, 19, 20, 26, 34, 36, 78, 95),
        (1, 2, 3, 4, 11, 17, 26, 27, 33, 36),
        (3, 4, 6, 15, 23, 25, 31),
        (1, 3, 5, 7, 15, 20, 23, 26, 31),
        (1, 2, 3, 6, 7, 22, 89, 93),
        (7, 10, 12, 13, 15, 19, 25, 31),
        (5, 9, 10, 11, 12, 13, 14, 15),
        (2, 17, 18, 20, 37, 39, 91, 92, 94),
        (5, 12, 13, 15, 18, 19, 23, 31),
        (2, 3, 7, 17, 20, 21, 26, 27, 34, 36, 94, 95),
        (1, 3, 17, 23, 26, 31, 33, 34, 39, 92),
        (7, 13, 19, 25, 34, 36, 39, 41),
        (3, 7, 17, 23, 26, 27, 31, 34, 90, 92),
        (13, 18, 19, 28, 31, 37, 39, 45),
        (25, 29, 33, 34, 35, 36, 39, 41),
        (2, 18, 20, 21, 30, 35, 86, 93, 94),
        (1, 2, 17, 19, 33, 34, 39, 43, 83),
        (23, 25, 26, 31, 33, 34, 35, 38, 39),
        (17, 19, 28, 33, 34, 36, 37, 39, 78),
        (3, 17, 27, 31, 34, 37, 89, 90, 92),
        (34, 35, 39, 41, 42, 43, 44, 47),
        (41, 43, 47, 49, 55, 95),
        (25, 36, 44, 47, 79, 82, 84, 87),
        (17, 34, 42, 45, 47, 50, 76, 95),
        (1, 18, 33, 39, 45, 46, 58, 83),
        (47, 50, 54, 55, 66, 74, 76, 77),
        (49, 55, 62, 63, 66, 67, 68, 76),
        (51, 52, 57, 59, 60, 61, 62, 63, 66, 71),
        (35, 45, 52, 65, 74, 82),
        (33, 35, 53, 54, 81, 82),
        (51, 52, 57, 58, 59, 60, 61, 62, 63, 71, 75),
        (52, 57, 58, 59, 62, 63, 65, 67, 68, 71),
        (50, 51, 57, 59, 63, 65, 66, 68, 71),
        (50, 55, 57, 66, 73, 74),
        (49, 58, 62, 63, 65, 67, 85),
        (42, 50, 55, 68, 73, 76, 95),
        (14, 55, 57, 59, 69, 75, 94),
        (50, 57, 63, 65, 66, 68, 70, 71),
        (55, 66, 73, 74, 76, 77, 79),
        (55, 57, 59, 68, 73, 75, 78),
        (35, 38, 47, 52, 53, 54, 74, 77, 82),
        (2, 19, 34, 36, 37, 78, 81, 82, 84, 95),
        (46, 58, 59, 65, 67, 71, 81, 85),
        (2, 3, 34, 36, 76, 78, 81, 82, 95),
        (58, 62, 63, 65, 71, 83, 85),
        (25, 31, 33, 34, 37, 38, 39, 84),
        (23, 27, 29, 33, 34, 35, 45, 86, 94),
        (10, 20, 25, 26, 27, 31, 84, 87),
        (3, 6, 7, 12, 27, 89, 90, 95),
        (2, 3, 6, 7, 27, 89, 90, 94, 95),
        (7, 25, 61, 79, 91),
        (17, 23, 27, 33, 34, 86, 90, 92, 94),
        (18, 23, 30, 33, 35, 39, 86, 93, 94),
        (14, 37, 55, 59, 77, 91, 94)
    ),
}
# [TIER] The tier's internal sub-term weights (the label-free
# product rule: reliability x distinctiveness, measured per
# sub-term by diagnostics/_tier_calibrate.py on the dev split).
#   formation  reliability 0.5477 distinctiveness 0.5941 -> 0.4570
#   msmap      reliability 0.7083 distinctiveness 0.2256 -> 0.2244
#   readout    reliability 0.8225 distinctiveness 0.2110 -> 0.2437
#   pairs      EXCLUDED: the pair histogram DILUTES the tier
#              (the label-free margin 0.29896 without it against 0.27724 with it).
ASPECT_TIER_WEIGHTS: dict = {'formation': 0.45696, 'msmap': 0.224397, 'readout': 0.243747}
# [TIER] The difference-like sub-terms' tolerances (the p90 of the
# raw difference over the corpus's random pairs, label-free).
ASPECT_TIER_TOLERANCE: dict = {'cohesion': 0.008586, 'texture2': 0.358152}
# [DEPTH] The cues that passed their OWN repeatability gate (the
# integer shift and the 0.9 scale; the gate is the null-corrected
# repeatability against the cue's own corpus null).
#   blur               shift 0.9972 scale 0.9843 null 0.8638 corrected +0.9322 -> ADMITTED
#   falloff            shift 0.9323 scale 0.9191 null 0.8520 corrected +0.4979 -> ADMITTED
#   shading            shift 0.8251 scale 0.9111 null 0.8057 corrected +0.3213 -> ADMITTED
#   texture_gradient   shift 0.9388 scale 0.8771 null 0.8170 corrected +0.4973 -> ADMITTED
ASPECT_DEPTH_CUES: tuple = ('texture_gradient', 'shading', 'blur', 'falloff')
ASPECT_DEPTH_WEIGHTS: dict = {'blur': 0.230391, 'falloff': 0.241338, 'shading': 0.248026, 'texture_gradient': 0.280244}
# [DEPTH] The cues' tolerances (the p90 of the raw difference over
# the corpus's random pairs, label-free).
ASPECT_DEPTH_TOLERANCE: dict = {'blur': 0.12321, 'falloff': 0.263551, 'shading': 0.263566, 'texture_gradient': 1.472701}
# [DEPTH] The shading field's smoothing scale: the 1/e correlation
# length of the interior luminance (68855 fragment pairs, measured on the dev corpus).
ASPECT_DEPTH_SHADING_SCALE: float = 0.45
# [DEPTH] The blur level reads the FINEST band(s) of the surface's
# spatial-frequency distribution.
ASPECT_DEPTH_BLUR_BANDS: int = 2
# The depth group's own corpus null (recorded, not applied -- the
# per-group null correction is OFF by the measured decision).
ASPECT_DEPTH_CORPUS_NULL: float = 0.83462
# tier: reliability 0.6591 distinctiveness 0.3616 raw product 0.2383; the calibrated weight on the shipped scale is 0.2971 (the corpus null 0.4064)
# tier is RETIRED to weight 0 (RETIRED).
#   the decision (diagnostics/_tier_decision.json, the dev
#   split, the paired instance-bootstrap, 2000 draws):
#     fused+tier 0.8264 against fused 0.8750: -0.0487 [-0.1042, +0.0069] p>0 0.028
#     the quadruped trio 0.6111 against 0.7963: -0.1866 [-0.3333, -0.0556] p>0 0.001
#   the reason: the group DILUTES the fusion: the paired interval is negative overall (-0.0487 [-0.1042, +0.0069]) and clearly negative on the quadruped trio (-0.1866 [-0.3333, -0.0556], which EXCLUDES 0 on the wrong side); the tier alone scores 0.2500 against the 0.125 chance, so the mechanism carries little identity information
ASPECT_V3_W_TIER: float = 0.0
# depth: reliability 0.9320 distinctiveness 0.1360 raw product 0.1268; the calibrated weight on the shipped scale is 0.1581 (the corpus null 0.8328)
# depth is RETIRED to weight 0 (RETIRED (BORDERLINE: the point estimate is positive in both measures and the paired interval is non-negative, but its lower bound is exactly 0.0000, so it does not EXCLUDE 0)).
#   the decision (diagnostics/_tier_decision.json, the dev
#   split, the paired instance-bootstrap, 2000 draws):
#     fused+depth 0.8889 against fused 0.8750: +0.0139 [+0.0000, +0.0347] p>0 0.883
#     the quadruped trio 0.8148 against 0.7963: +0.0188 [+0.0000, +0.0556] p>0 0.671
#   the reason: the group does NOT dilute (the point estimate is positive overall AND on the quadruped trio, and the paired interval is non-negative on both), but its lower bound sits exactly at 0.0000 in both measures (2000 draws and, as a sensitivity check, 20000 draws), so the stated admission bar (an interval EXCLUDING 0) is not met; it is retired to weight 0 with this entry, and it is the strongest candidate of the two for a pre-registered confirmation read
ASPECT_V3_W_DEPTH: float = 0.0
# The two new groups' corpus nulls (RECORDED, not applied -- the
# per-group null correction stays OFF by the measured decision).
ASPECT_V3_NEW_GROUP_NULL: dict = {'tier': 0.406377, 'depth': 0.832839}
# [TIER+DEPTH CONSTANTS END]
# The layout group's kernels: the centroid distance is in units of the energy
# radius (so the reading is scale-free) and the radius ratio is read on a
# log scale.
#
# DERIVED FROM MEASURED REPEATABILITY (diagnostics/_repeatability.py, the
# pre-registered perturbations shift / scale / gain / blur / noise, measured
# on the dev split), not chosen:
#   the energy centroid moves by p50 0.027 and p90 0.326 ENERGY RADII under
#   those perturbations -> the centre tolerance is the p90 (0.3264), so a
#   perturbed view keeps ~exp(-0.5) of its centre agreement;
#   the energy radius changes by p50 0.028 and p90 |log ratio| 0.3513 ->
#   the radius tolerance is that p90 (0.3513).
# Date 2026-10-20; mechanism version descriptor-v3.
ASPECT_V3_LAYOUT_CENTER_SIGMA: float = 0.3264
ASPECT_V3_LAYOUT_RADIUS_SIGMA: float = 0.3513
# Coarse (all-six-slot) bins contribution inside the shape group. Small on
# purpose: the coarse index may stabilise identity, but the graded fragment
# populations must carry the information.
ASPECT_V2_COARSE_BIN_WEIGHT: float = 0.20
# Soft polar arrangement. Radial sigma is in RMS-spread units; angular sigma
# is in degrees on the circular upright axis.
#
# DERIVED FROM MEASURED REPEATABILITY (diagnostics/_repeatability.py): under
# the pre-registered perturbations (shift, scale, gain, blur, noise) the
# part's polar address moves by p50 0.060 / p90 0.301 in the radial
# coordinate and p50 4.7 deg / p90 33.1 deg in the angle (the parts matched
# by position, never by index). The tolerances are the p90 with a 1.4x
# margin: 0.301 * 1.5 = 0.45 and 33.1 * 1.36 = 45 deg.
# Date 2026-10-20; mechanism version descriptor-v3.
# [W2/F] The arrangement kernel's sigmas are CALIBRATED, not guessed:
# they are the measured spread of the matched pairs' radius and bearing
# over the dev corpus (median |dr| 0.1013, median |dtheta| 9.85 deg,
# 1504 matched pairs), scaled so the kernel is 0.5 at a typical matched
# pair. The previous hand-set values (0.45 / 45 deg) were too LOOSE: a
# typical matched pair scored 0.87 under them, so the group barely
# discriminated. Producer: diagnostics/_calibrate_arrangement.py.
# Date: 2026-10-21. Mechanism version: descriptor-v5.
ASPECT_V2_POS_SIGMA_R: float = 0.1217
ASPECT_V2_POS_SIGMA_THETA_DEG: float = 11.83
# View-level soft angular mass histogram (circular, normalised). It adds the
# global arrangement (mass distribution / symmetry) that a per-part kernel
# cannot see; multiplied into resonance, never used as a gate.
ASPECT_V2_POLAR_BINS: int = 12
ASPECT_V2_W_POLAR_HIST: float = 0.15
# Surface sampling window, in pixels: radius = clip(frac * R, min, max). R is
# the view's wavelength/extraction scale, so the window scales with the frame.
ASPECT_V2_SURFACE_RADIUS_FRAC: float = 0.35
ASPECT_V2_SURFACE_RADIUS_MIN: float = 6.0
ASPECT_V2_SURFACE_RADIUS_MAX: float = 18.0
# Surface slot kernels. Hue is circular in degrees; saturation and luminance
# are in [0, 1]; texture orientation/scale are graded populations compared by
# chance-corrected overlap (no kernel width).
ASPECT_V2_SURFACE_SIGMA_HUE_DEG: float = 30.0
ASPECT_V2_SURFACE_SIGMA_SAT: float = 0.25
ASPECT_V2_SURFACE_SIGMA_LUM: float = 0.25

# =============================================================================
# [Q3] NEGATIVE (ANTI-PHASE) COUPLING
# =============================================================================
# The coupled-phase oscillator separates two cliques only when the coupling
# matrix can say NO (dynamics/oscillator.py: negative edge weights drive the
# phase difference toward pi instead of 0). Measured before: the observed
# graph of a two-panel composite carried 146 edges, ALL sign=+1, 118 of them
# EDGE_CONTACT proximity noise, so two-panel separation could not be positive
# for a structural reason. These switches enable the two anti-phase sources in
# perception/anti_phase.py, wired by orchestration/binder.py:
#
#   1. opposite flank owners across a shared boundary (mutual
#      BorderOwnership claims) -- the figure-ground / occlusion relation
#   2. closed-chain interior vs exterior (a ring PEAK_CHAIN enclosing another
#      detection's anchor)
#
# Ships DARK by default for the same reason the other scoring-facing work
# does: the gate is the two-panel separation measurement, not an assertion.
ANTIPHASE_ENABLED: bool = False
#: Minimum confidence (weakest-claim strength / containment depth) for an
#: anti-phase edge to be emitted. Calibrated by a rule written before looking
#: (diagnostics/_q3_strength.py): "p10 of the observed BorderOwnership
#: strength distribution rounded down to two decimals". Measured over 552
#: detections on 3 composites + 2 real images: p10 = 0.200, p50 = 0.214,
#: max = 0.347, so the threshold is 0.19 and ~90% of claims pass.
ANTIPHASE_MIN_CONFIDENCE: float = 0.19
#: Per-node cap on EDGE_CONTACT edges (proximity-only, so its count, not its
#: structure, is what dominated the coupling matrix). Distance decay stays;
#: the cap only bounds how many neighbours each node may bind to.
CONTACT_MAX_PER_NODE: int = 6
#: [Q3] Apply the per-node contact cap. Shadow of ANTIPHASE_ENABLED: the cap
#: is part of the same measured change set (it changes which edges reach the
#: coupling matrix), so it ships dark under its own switch.
CONTACT_CAP_ENABLED: bool = False

# --- ORIENTATION_CONTINUITY binder (v8) ---
# Angular tolerance: signal falls off as exp(-(Δθ/σ_θ)²), Δθ in radians.
ORIENTATION_CONTINUITY_SIGMA_THETA = 0.3
# 0.5 rad ~= 29deg: on a uniform 32px patch grid, the major-axis
# estimate rotates ~(pitch / radius_of_curvature) between adjacent
# rim patches (~0.4-0.5 rad for object-scale rims), so 0.3 let only
# straight background lines chain while every curved object rim fell
# under THETA_CONTOUR_EDGE -- but loosening also merged rim and
# background chains into 10-node comps and slightly worsened class
# gaps, so the change was REVERTED to 0.3 after measurement.
# Perpendicular distance tolerance as a fraction of scene L_med. On a
# uniform grid (pitch = L_med), 0.5*L_med structurally rejects every
# DIAGONAL edge continuation (perp = L_med/sqrt(2) ~ 0.71*L_med ->
# exp(-2) = 0.135), so smooth curves could never chain; 1.0 restores
# diagonal continuations (exp(-0.5) ~ 0.61) while the collinearity
# term (sigma_theta) remains the semantic gate.
ORIENTATION_CONTINUITY_SIGMA_P_FRAC = 1.0
# Along-axis gap decay as a multiple of scene L_med.
ORIENTATION_CONTINUITY_TAU_GAP_FRAC = 2.0

# [Pass 21] Per-port competition and bipole role parameters.
# All five values are UNMEASURED initial estimates. Calibrate against ETH-80.
ORIENTATION_CONTINUITY_RANGE_TAU_MULT = 2.0      # search radius = this x tau_gap
ORIENTATION_CONTINUITY_PORT_MIN_SIGNAL = 0.05    # floor for a port to count as supported
ORIENTATION_CONTINUITY_MUTUAL_FLOOR = 0.7        # signal factor for non-mutual winners
ORIENTATION_CONTINUITY_BIPOLE_STRONG = 0.4       # sqrt(s_fwd*s_bwd) cut for "strong interior"
ORIENTATION_CONTINUITY_POLARITY_FLIP_GAIN = 1.0  # 1.0 = polarity is slot-only, no signal effect

# v32: shared Gaussian sigma for ORIENTATION_BEND and
# ORIENTATION_JUNCTION. Approximately 15° (π/12). The angle-range
# membership test uses a discrete n_theta=24 lattice with 7.5° bin
# spacing, so sigma=π/12 gives ~0.78 similarity to adjacent bins
# and ~0.37 to two-apart bins.
ORIENTATION_ANGLE_RELATION_SIGMA_THETA = math.pi / 12.0
# Spatial horizon for ORIENTATION_BEND / ORIENTATION_JUNCTION, as a
# multiple of the step's EXTRACTION SCALE (patch radius), not of
# _step_l_med. A bend or junction is a LOCAL relation: on a uniform grid,
# _step_l_med collapses to the grid pitch (32px) and the old
# TAU_GAP_FRAC*l_med = 64px horizon welded patches up to ~3 cells apart --
# with no hard cutoff -- into one image-spanning angle-relation component.
# 1.0 * extraction_scale (~22.6px at 8x8/256) keeps nearest-neighbour
# locality (dist 32px ~> e^-1.41) and kills diagonal/long-range welds.
ORIENTATION_ANGLE_RELATION_KAPPA = 1.0

# =============================================================================
# v32.2 GRADED CORE DESIGN
# =============================================================================
# Thresholds removed from the flow:
#   - intersection's min_positive_ratio → replaced by min_frequency (default 0.0)
#   - filter's theta_pos_min / theta_disc_norm → unused
#   - mcs_score's flat count → replaced by mass-weighted, smaller-side numerator
#
# Mass pipeline:
#   1. Intersection emits per-element consistency ∈ [0, 1] = mean pairwise
#      signature similarity among aligned positives (nodes), or presence
#      frequency (edges). Stored in Subgraph.node_weights / Subgraph.weights.
#   2. Filter multiplies by max(0, disc)**DISC_EXPONENT and overwrites the
#      same fields with the final mass.
#
# final_mass = consistency * max(0, disc)**DISC_EXPONENT
#
# SEMANTIC SHIFT: Subgraph.node_weights / Subgraph.weights mean
#   - consistency prior ∈ [0, 1]  before the filter
#   - final mass ∈ [0, 1]         after the filter

# [FIX 2 / Item round 3] Recurrence rule: single source of truth is
# _wilson_min_recurrence() in memory/distributions.py. It computes the
# minimum recurrence count pc such that Wilson-lower(pc/n_pos) >
# Wilson-upper(0/n_neg). For n_pos=7, n_neg=8 this gives pc=4
# (frequency 4/7 ≈ 0.571).
#
# THETA_MIN_FREQ previously defaulted to 0.0 (dead knob: no filtering)
# then 0.5 (hand-set constant competing with the Wilson rule). It is now
# env-override only, labeled diagnostic, with no default. Callers that do
# not pass min_frequency explicitly get the Wilson-computed value from
# _structural_intersection.
#
# THETA_MIN_CONSISTENCY was a dead 0.0 knob (0.0 = no consistency
# filtering at the intersection step). Removed; the Wilson recurrence
# test subsumes its purpose.
THETA_MIN_FREQ: Optional[float] = (
    float(os.environ.get("THETA_MIN_FREQ")) if os.environ.get("THETA_MIN_FREQ") else None
)

# =============================================================================
# v32.3 CONTENT-TIER SIMILARITY CLUSTERING
# =============================================================================
# Content-tier identity was previously hash(exact_signature). Two signatures
# one bin apart produced distinct nodes and were never treated as the same
# content. This caused the 1-bit quantization trap: a bin-boundary wobble
# across the class appears as two disjoint class-specific nodes, inflating
# both `disc` values. Clustering content nodes by similarity θ_cluster
# merges near-identical signatures into one node.
#
# Semantics: when a content signature arrives, resolve it to the nearest
# existing cluster centroid of the same lens type whose similarity is at
# least θ_cluster. If found, join that cluster (update centroid as running
# mean). If not found, create a new cluster seeded by this signature.
#
# θ_cluster is on TypedSignature.similarity's scale. Given the current
# flat similarity calibration (~0.05 per bin on a 20-range), θ_cluster=0.90
# merges signatures within ~2 total bin units and is the default.
# Env-overridable so the sweep does not require an edit per point.
#
# Instance-tier identity (§1.11 `(lens_type, signature_hash)`) is
# unchanged. Only the content tier clusters.
# [v32.3] Content-tier similarity clustering. Set to 0.99 to disable
# merging in practice (current `similarity` scale cannot exceed 0.95
# for adjacent bins, so 0.99 thresholds effectively yield hash-deterministic
# IDs). The v32.4 distribution classifier does NOT rely on clustering;
# this default preserves the pre-clustering behavior.
THETA_CLUSTER_CONTENT: float = float(os.environ.get("THETA_CLUSTER_CONTENT", "0.99"))

# [v32.5] Distribution normalization mode for the content distribution
# classifier.
#   'per_lens'     — each lens type normalized to mass 1.0 (v32.4 behavior).
#   'global'       — raw disc values across all lens types, globally normalized.
#   'log'          — log(1+disc) per key, globally normalized. Transform
#                    applies to class-side disc only; query side is the
#                    same globally normalized presence distribution as
#                    'global'.
#   'class_global' — class side uses 'global' (flattened __global__),
#                    query side is flattened __global__ but per-lens
#                    normalized. Diagnostic mode for isolating class-side
#                    effects from query-side dilution.
# Env-overridable. Read once at import; not mutated at runtime.
VALID_DIST_NORM_MODES = ("per_lens", "global", "log", "class_global")

DIST_NORM_MODE: str = os.environ.get("DIST_NORM_MODE", "per_lens")
if DIST_NORM_MODE not in VALID_DIST_NORM_MODES:
    raise ValueError(
        f"unknown DIST_NORM_MODE: {DIST_NORM_MODE!r}; "
        f"expected one of {VALID_DIST_NORM_MODES}"
    )

GRADED_TOP_K_NODES: int = int(os.environ.get("GRADED_TOP_K_NODES", "100"))
GRADED_TOP_K_EDGES: int = int(os.environ.get("GRADED_TOP_K_EDGES", "100"))
DISC_EXPONENT: float = float(os.environ.get("DISC_EXPONENT", "2.0"))

# [Pass 20] OBJECT_SHAPE_COARSEN_FACTOR and OBJECT_SHAPE_CONTENT_MODE removed
# with the ObjectShape oracle. Nothing reads them now that ObjectShapeLens is
# excised and content_signature no longer has an ObjectShape branch.

# --- Phase A (Step 1 + Step 2) gating / calibration ---
# SIMILARITY_MODE selects TypedSignature.similarity implementation.
# Read at import of parameters.py (restart required to change).
SIMILARITY_MODE: str = os.environ.get("SIMILARITY_MODE", "gaussian")
# [POPULATION CODING] Master A/B switch: when True (default) a CANONICAL
# signature that carries a graded `population` for a registered class is
# compared by chance-corrected activity-vector OVERLAP instead of the per-slot
# Gaussian over bins (bins remain only the lookup/index key). This is the
# representation thesis; the switch exists so the whole population path can be
# A/B'd against the bins path without a code change. It does NOT gate whether a
# mechanism is built -- every registered class emits its full population
# unconditionally; only the matcher's read of it is switchable.
POPULATION_CODING_ENABLED: bool = os.environ.get(
    "POPULATION_CODING_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
# [ROUND 10 / A] Graded populations on EDGE parameters (relative angle,
# distance, scale ratio) for the contained/parallel/symmetric relations.
# Same contract as POPULATION_CODING_ENABLED: every such edge derives its
# population unconditionally; this only switches whether the matcher READS it
# (off -> the bins Gaussian, for A/B).
EDGE_POPULATION_CODING_ENABLED: bool = os.environ.get(
    "EDGE_POPULATION_CODING_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
# Sigma fallback for range-0 slots (e.g. BOUNDARY_ANCHORED slot 2).
SIGMA_FALLBACK_FOR_RANGE0: float = float(os.environ.get("SIGMA_FALLBACK_FOR_RANGE0", "5.0"))
# Scale applied to graded disc when exact disc is zero (dual filter fallback).
GRADED_FALLBACK_SCALE: float = float(os.environ.get("GRADED_FALLBACK_SCALE", "0.5"))
# Per-image edge cap for graded edge fallback (bounds O(core_edges * img_edges)).
MAX_GRADED_EDGES_PER_IMAGE: int = int(os.environ.get("MAX_GRADED_EDGES_PER_IMAGE", "200"))
# Master gate for dual (exact-preferred, graded-fallback) discriminative filter.
# Env default "1" (enabled); runtime-overridable via P.DUAL_FILTER_ENABLED assignment.
DUAL_FILTER_ENABLED: bool = os.environ.get("DUAL_FILTER_ENABLED", "1") != "0"

# --- Phase B (Representation tier) — futurework3.md B.4–B.16 ---
#
# Master switch for the six higher lenses, their binder plumbing, and
# surround suppression. Starts FALSE (B.16 / II.46) and is flipped TRUE in
# Phase B Step 4, when BorderOwnershipLens is the only implemented lens —
# the other five remain uncalled skeletons, so the tier is live but
# single-lens. Roll back by setting this back to False.
HIGHER_LENS_ENABLED: bool = True

# --- [U20] migration flags -------------------------------------------------
# Named, ledgered switches (diagnostics/_u20_migration_flags.json). Both default
# ON because the U20 gate is zero-delta on the behavioural fields; set the
# environment variable to "0" to reproduce the pre-U20 behaviour exactly.
#
# SPINE_IDENTITY_UNIFIED: one physical contour gets ONE identity -- where an
# orientation-spine Contour is positionally the same contour as a field-spine
# PEAK_CHAIN, the Contour is emitted with the chain's canonical anchor
# (LEAD 2).
# JUNCTION_FAMILY_SPLIT: JunctionLens emits Corner / T / Y as separate node
# kinds, with the generic "Junction" kind kept as the compatibility ALIAS in
# the shared detection dict (LEAD 3).
SPINE_IDENTITY_UNIFIED: bool = os.environ.get(
    "SPINE_IDENTITY_UNIFIED", "1") != "0"
JUNCTION_FAMILY_SPLIT: bool = os.environ.get(
    "JUNCTION_FAMILY_SPLIT", "1") != "0"
#
# [U21] ORIENTATION_SPINE_WIRED: ContourLens reads the GRAPH the binder
# actually wrote. Two fractures kept the orientation spine inert on the 2D
# path (diagnostics/_u21_id_mismatch.json): the continuity edges connect
# ORIENTATION_POL nodes while the lens minted ORIENTATION nodes (0 of 48 ids
# shared), and the lens gated on edge.weight, which the weight EMA only
# writes in Phase 5, AFTER the higher-lens tier (0 of 56 edges above the
# threshold during the step that created them). Set the environment variable
# to "0" to reproduce the pre-U21 (inert) behaviour exactly.
ORIENTATION_SPINE_WIRED: bool = os.environ.get(
    "ORIENTATION_SPINE_WIRED", "1") != "0"
#
# [U25] BORDER_SUPPORT_GATE: extract_peaks keeps a candidate that lies
# inside the kernel-support border band when the raw luminance has a real
# edge at its pixel (the reflect-padding ghosts the band exists to remove
# have none).  SHIPPED DEFAULT 0: with the gate ON the field spine's 121 px
# clean-condition gap closes (121 -> 9 px at S1:L0) and its coverage rises
# 0.554 -> 0.683, BUT the recovered border peaks let a T-junction edge form
# on the faded box scenes (V1 evidence + ledger row 18: see
# diagnostics/_u25_gap_fix.md).  Under the SESSION U25 (RESUMED v2) new plan,
# Lead 1 re-enabled the mechanism (shipped default 1, `_BORDER_SUPPORT_MIN_GRAD`
# = 0.02); "0" reproduces the pre-U25 hard discard byte-for-byte.
BORDER_SUPPORT_GATE: int = int(os.environ.get("BORDER_SUPPORT_GATE", "1"))
BORDER_SUPPORT_MIN_GRAD: float = float(os.environ.get("BORDER_SUPPORT_MIN_GRAD", "0.02"))
#
# [U25 LEAD 6 / PATH A] ORIENTATION_PIXEL_ANCHORS: a GABOR Orientation
# anchor (region u, theta index, sf index, cell kind) is positioned at the
# spatial maximum of its OWN channel inside the region's footprint, read from
# the pixel-level Gabor field (the EndStoppedLens precedent).  Set to "0" to
# reproduce the pre-U25 behaviour exactly (every GABOR anchor at its region
# CENTROID, verified by diagnostics/_u25_orient_patha.json).
ORIENTATION_PIXEL_ANCHORS: bool = os.environ.get(
    "ORIENTATION_PIXEL_ANCHORS", "1") != "0"
# [U26] The footprint of that spatial maximum, as a fraction of `radius_px`.
# `V1RetinotopicLattice` sets radius_px = (stride/2) * overlap_factor with
# overlap_factor = 1.6, so radius_px / 1.6 is exactly the half-width of the
# region's OWN Voronoi tile.  1.5 (the U25 value) is 2.4 tiles wide: the
# channel argmax then lands on a NEIGHBOUR's stronger edge, and on box-over-box
# that pulled every bottom-edge anchor onto the front box's silhouette 12 px
# away (S2:L0 arc-2 coverage 0.168, diagnostics/_u26_probe_anchors.json).  The
# tile value keeps the anchor inside the region it belongs to; S2:L0 anchor
# coverage 0.443 -> 0.515 at the unchanged threshold
# (diagnostics/_u26_sweep.json).
ORIENTATION_ANCHOR_WINDOW_FRAC: float = float(os.environ.get(
    "ORIENTATION_ANCHOR_WINDOW_FRAC", "0.625"))
# The RF footprint is the U25 value (1.5x).  [U26] The tile is a CLIPPED maximum.
# When the same edge peak lies just outside the tile, the tile still sees most of
# its response and the peak is continuous with the region; when a much stronger
# peak from another structure lies in the receptive field, the tile sees only a
# small fraction of it.  So the receptive-field maximum is followed only when the
# tile response is at least FOLLOW_RF_RATIO of it (a factor-2 continuity test),
# otherwise the anchor stays in its own tile.  This keeps S2:L0's occlusion-edge
# anchors in their tile (tile/RF ~0.2) while restoring S1:L0's
# sphere-silhouette anchors to the RF peak (tile/RF ~0.9).
# [U27 LEAD 3] The same-connected-component alternative was implemented and
# measured: it recovered S1's precision completely (S1:L0 P 0.716, S1:L1 P 0.755)
# but broke S2:L0 coverage (0.481 vs the 0.50 bar), because a monotone shading
# ramp is one connected component.  Rejected; the ratio rule ships.
# See diagnostics/_u27_final_report.md and diagnostics/_u27_lead3_comp.py.
ORIENTATION_ANCHOR_RF_FRAC: float = float(os.environ.get(
    "ORIENTATION_ANCHOR_RF_FRAC", "1.5"))
ORIENTATION_ANCHOR_FOLLOW_RF_RATIO: float = float(os.environ.get(
    "ORIENTATION_ANCHOR_FOLLOW_RF_RATIO", "0.5"))
#
# [U25 LEAD 6 / F4] CONTOUR_SPINE_MAP: the Contour lens receives the binder's
# MERGED anchor map (region keys + GABOR tuple keys) instead of the
# region-keyed map.  The orientation spine's members ARE tuple anchors; with
# the region map only 0.004-0.168 of them resolved (U24, U25 card), so the
# lens fell back to uniform spacing and skipped its figure-ground side test.
# Set to "0" to reproduce the pre-U25 wiring exactly.
CONTOUR_SPINE_MAP: bool = os.environ.get("CONTOUR_SPINE_MAP", "1") != "0"

# Phase B spatial kappas: multipliers of extraction quantum.
# extraction quantum = median V1 region radius in observation units
# (pixels in 2D, meters in 3D), same units as anchor_positions.
# Effective radius = KAPPA * extraction_scale.
# Scene-stable (does not change with scene crowding), resolution-
# independent, modality-agnostic. Same spirit as V1 receptive fields.
#
# WHY multipliers and not absolute radii: the Phase B draft specified
# absolute radii (1.5-3.0) that read as if they were in grid-cell units,
# but anchor_positions is in PIXELS (2D) or METERS (3D). On the Phase-A
# grid (256px image, GRID_N=8) the minimum anchor separation is 32px, so
# every absolute radius below 32 made its distance gate unsatisfiable —
# the whole higher tier was silently inert (verified: BorderOwnership
# emitted 0 at R=3.0, 128 at R=48.0; surround_suppress suppressed 0 at
# R=1.5). Kappas restore the spec's intended *relative* neighbourhood
# sizes while staying correct across resolutions and modalities.
KAPPA_SURROUND_SUPPRESS = 1.5
KAPPA_BORDER_OWNERSHIP = 3.0
KAPPA_CONTOUR_CONTINUITY = 2.0
KAPPA_JUNCTION_CONTOUR = 2.5
KAPPA_CLUSTER_PART = 2.0
KAPPA_CLUSTER_OBJ = 3.0

# B.4 surround suppression similarity gate (dimensionless; not a radius).
THETA_SURROUND_SUPPRESS: float = 0.9

# B.6 / B.7 edge-weight gates for the Orientation-neighbour graphs. These
# are thresholds on edge WEIGHT, not distances, so they need no kappa.
# R_CONTOUR_EDGE / R_JUNCTION_EDGE are spec names the draft never bound to
# an edge; retained at the local-neighbourhood scale.
R_CONTOUR_EDGE: float = 2.0
THETA_CONTOUR_EDGE: float = 0.15
R_JUNCTION_EDGE: float = 2.0
THETA_JUNCTION_EDGE: float = 0.15

# B.12 per-step pair caps that bound the O(n^2) emitter loops.
MAX_CONTOUR_PAIRS_PER_STEP: int = 2000
MAX_JUNCTION_CONTOUR_PAIRS: int = 2000

# B.14 recurrent contour grouping (ContourGainState).
CONTOUR_GAIN_INCREMENT: float = 0.1
CONTOUR_GAIN_DECAY: float = 0.9
CONTOUR_GAIN_FLOOR: float = 0.001
MAX_CONTOUR_GAIN_ENTRIES: int = 5000
MAX_CONTOUR_GAIN_VALUE: float = 1.0
# [Phase B] Step 9. Flipped True at the end of Step 9, after validating the
# threading against a flag-off V1 baseline (20/20 gate) and confirming on the
# image probe that gain actually reduces Orientation emissions.
RECURRENT_GROUPING_ENABLED: bool = True

# --- [B5] Contour curvature-sequence signature ---------------------------
CONTOUR_N_SEQ: int = 20
CONTOUR_SEQ_LEVELS: int = 9                 # odd: zero sits on the centre level
CONTOUR_CORNER_TURN_FRAC: float = 0.25      # corner if |turn| > 0.25 * pi
CONTOUR_TOTAL_TURN_MAX: float = 8.0 * math.pi
CONTOUR_GEOM_BAND_BINS: int = 3             # corners within this many bins of 90 deg use geometry
# [Pass 17.3] Default "summary": with CONTOUR_CONTENT_MODE="full" the 27-slot
# raw curvature sequence enters the content tier and THETA_CLUSTER_CONTENT=0.99
# fragments it across viewpoints (benchmark regressed 88.9% -> 66.7%). "summary"
# projects to the 5 invariant slots (length, closure, total_turn, corners,
# chirality) at the content tier; the rich 27-slot sequence stays at instance tier.
CONTOUR_CONTENT_MODE: str = os.environ.get("CONTOUR_CONTENT_MODE", "summary")

# --- [B6] Border Ownership Field -----------------------------------------
BO_TEXTURE_BIN_MAX: float = 9.0
BO_RICHNESS_FALLBACK: float = 0.0
BO_FIELD_SAMPLE_COUNT: int = 8
BO_CURVATURE_WEIGHT: float = 0.4
BO_TEXTURE_WEIGHT: float = 0.3
BO_COLOR_WEIGHT: float = 0.2
BO_GRAVITY_WEIGHT: float = 0.1
BO_CUE_SCALE: float = 2.0
BO_GRAVITY_SCALE: float = 10.0

# --- [ROUND 10 / C] Per-flank fragment ownership --------------------------
# Ownership is a property of an oriented edge fragment (one per chain member,
# i.e. a peak) with a signed side bit: + means the owner (figure) lies on the
# LEFT of the chain's TRAVEL direction. Edges are derived from the intrinsic
# ownership afterwards. The old entity-based path (segmenter Boundary / field
# regions) is legacy and OFF by default.
BO_FRAGMENT_OWNERSHIP_ENABLED: bool = True
# Chain-wide consensus: the per-member estimate is blended along the contour
# (moving average) before it may switch side. A flip is allowed only at a
# corner (|signed turn| above BO_CORNER_FLIP_MIN_TURN) or a junction member;
# elsewhere a lone opposite member is absorbed into its neighbours.
BO_CONSENSUS_WINDOW_FRAC: float = 0.25
BO_CORNER_FLIP_MIN_TURN: float = 0.5
# Partner search (edge derivation). Radius is a multiple of the fragment's own
# flank offset (scale-relative); close same-chain neighbours inside
# BO_PARTNER_EXCLUDE_NEIGHBOURS are never partners, but a circle's far side
# (many members away) still pairs.
BO_PARTNER_RADIUS_SCALE: float = 2.0
BO_PARTNER_EXCLUDE_NEIGHBOURS: int = 2
# Enclosure (interior-field difference across the edge) is sampled at several
# scales, in multiples of INTERIOR_PROBE_PX. This replaces a separate
# surroundedness weight with a multi-scale enclosure term.
BO_SURROUND_PROBES: tuple = (1.0, 2.0, 4.0)
# Log-odds weights for the ownership vote. Derived by logistic fit on a
# synthetic ground-truth corpus (diagnostics/_round10_C_fit.py), never from a
# score. Contrast polarity is deliberately NOT a voter: V2 ownership is largely
# polarity-independent, so polarity stays a linking/descriptor property.
BO_LOGODDS_WEIGHTS: dict = {"convexity": 1.0, "enclosure": 1.2, "t_junction": 0.8}

# DEPRECATED — retained for env overrides and probe call-site compatibility.
# The graded filter accepts and ignores these.
THETA_DISC_NORM: float = float(os.environ.get("THETA_DISC_NORM", "0.10"))
THETA_POS_MIN_CONTENT: float = 0.30
POOL_SIGMA_FLOOR: float = 0.10


# --- SURFACE_GROUPING binder (v8) ---
# Color distance scale (RGB in [0, 1]³). Signal term exp(-d/σ_c).
SURFACE_GROUPING_SIGMA_C = 0.2
# Texture distance scale (2-dim quantized roughness/isotropy in [0, 1]²).
SURFACE_GROUPING_SIGMA_T = 0.3
# Absolute floor: candidate pairs below this raw signal are dropped
# before max-normalization.
SURFACE_GROUPING_MIN_SIGNAL = 0.05
# Spread gate: if max(signal) - min(signal) across candidates is below
# this, no edges are emitted (all-equal means nothing to discriminate).
SURFACE_GROUPING_SPREAD_MIN = 0.05

# =============================================================================
# 13. EXTRACTION: BINDERS (order-1)
# =============================================================================
# §2.3. Constants for order-1 binder signal functions.

TRANSITION_ONSET_LAG = 1  # Fixed at 1 step (§1.12)
CO_MOTION_MIN_FRAMES = 3  # v30 §2.1.4 θ_comotionFrames: min co-occurrence frames
CONTACT_R_MED_BOOTSTRAP = 1.0  # §1.12 CONTACT bootstrap default
TOPOLOGY_K_MAX = 4  # K_TOPOLOGY
TOPOLOGY_RADIUS_MULT = 2
DEGREE_RADIUS_MULT = 1
# [D2/Seam-4] Scale-adaptive DEGREE radius: R_eff = max(κ·L, floor) with L
# the median nearest-neighbour distance among positioned anchors. The law
# is scale-invariant (grid units, meters, pixels) — no absolute clamp
# bounds, which would re-hardcode a single metric scale. The floor only
# guards the degenerate co-located-anchors case (L = 0).
DEGREE_KAPPA = 1.5
DEGREE_RADIUS_FLOOR = 1e-6
ETA_SEQ_AGE = 0.1  # §1.12 SEQUENCE age decay
CONTRAST_MIN_OBS = 5  # Min observations for contrast signal
PERIODICITY_MIN_INTERVALS = 3  # Min intervals for periodicity signal
PERSISTENCE_CHANGE_THRESHOLD = 0.1  # "B changes" magnitude threshold

# =============================================================================
# 14. EXTRACTION: BINDERS (order-2) — REMOVED
# =============================================================================
# Structural similarity is no longer a binder: comparing edge-neighborhoods is
# a measurement, not an observed relation, so it emits no §2.8 edge type and
# lives in structural_similarity.py as a pure operator (§1.5, §4.12, §6.4.1).
# The order-2 binder's constants (STRUCT_SIM_NEIGHBORHOOD_DEPTH,
# STRUCT_SIM_MIN_OVERLAP, STRUCT_SIM_MAX_NODES) were deleted with it; the
# operator's own weights/threshold are defined once, under §20.

# =============================================================================
# 15. EXTRACTION: SELF-BINDERS (DEPRECATED / REMOVED)
# =============================================================================

# =============================================================================
# 16. EXTRACTION: QUANTIZATION
# =============================================================================
# Q_* values drive the constitutional R_j ranges in primitives.py.

Q_P = 0.1  # Persistence quantization
Q_S = 0.1  # Feature-vector stability quantization
Q_R = 0.05  # Relational distance quantization
Q_D = 1.0  # Degree quantization
Q_DELTA = 0.5  # Structure feature-delta quantization
Q_T = 1.0  # Time quantization
PAD_DIST = 20  # Topology absent-neighbor sentinel
DEG_MAX_3D = 8  # 3D contact degree maximum
DEG_MAX = 50  # Legacy 2D degree maximum

# =============================================================================
# 17. BINDER THRESHOLD MECHANISM
# =============================================================================
# §2.7. Structurally identical to lens θ, applied per relation channel.

THETA_CHANNEL_BASE_INIT = 0.15  # Lowered from 0.5. §2.7 thresholds must
                                # overlap the emittable signal band, which
                                # for self-binders is [0, ~0.2]. A 0.5 start
                                # never intersects real signals before the
                                # channel decays to floor, and empty-candidate
                                # steps (now homeostatic) still need to reach
                                # an entry threshold below ~0.17 to fire.
DELTA_CHANNEL_HYST = 0.02
KAPPA_CHANNEL_THETA = 0.05
RHO_CHANNEL_TARGET = 0.2
ETA_CTX_CHANNEL = 0.05
ETA_FAT_CHANNEL = 0.05
BETA_F_CHANNEL = 0.95
KAPPA_C_CHANNEL = 0.05
KAPPA_R_CHANNEL = 0.05
THETA_CHANNEL_MIN = 0.05        # Lowered from 0.1. A silent channel's
                                # homeostatic floor must sit below the
                                # smallest meaningful signal. Low-signal
                                # candidates from a one-bin BOUNDARY shift
                                # carry signal ~0.056; the entry threshold
                                # (theta_eff + DELTA_CHANNEL_HYST = 0.07)
                                # must be reachable from theta_eff = MIN.
THETA_CHANNEL_MAX = 0.9

# =============================================================================
# 18. BUILDER
# =============================================================================
# §4.22. Every constant listed in the Builder parameter table.

# Freshness and selection
TAU_FRESH = 50
ETA_FRESH = 0.2
THETA_CURIOUS = 0.3
THETA_STERILE = 0.1
ETA_SURPRISE = 0.3

# Stall detection
W_STALL = 20
STALL_THRESHOLD = 0.7
KAPPA_REF = 0.1
EPS_REF = 0.01

# Excursion bounds
TAU_EXCURSION_MAX = 100
TAU_EPISTEMIC_MAX = 50

# Basin and Spine maintenance
K_MAIN = 500
THETA_BASIN_FULL_RECOMPUTE = 0.5

# Frontier lifecycle
THETA_PRUNE_FRONTIER = 0.01
K_PRUNE_FRONTIER = 30

# Value function inputs
F_RECURSIVE_DEPTH = 3
LAMBDA_RECURSIVE = 0.7
THETA_FERTILE_PERCENTILE = 90  # §4.7 dynamic θ_fertile
RF_MIN_FERTILE = 1.0  # §4.7 θ_fertile floor
K_FERTILE = 100  # §4.7 θ_fertile recomputation cadence
FERTILE_MIN_POPULATION = 20
# [P0-2] Cold-start rescue: guarantee a non-empty fertile set when at
# least one crystallized concept has positive EF but sits below the
# percentile-derived candidate threshold. The 5% margin prevents ULP
# chatter near top_val; the 10% retighten margin prevents mode
# oscillation when top_val hovers near the candidate boundary.
FERTILE_ADMIT_MARGIN = 0.05
FERTILE_RETIGHTEN_MARGIN = 0.10

# [P0-3] Bootstrap-aware membership re-observation. A candidate is
# partially re-observed when at least this fraction of its members are
# present this step AND every edge with both endpoints present is
# active per §1.12. Fractional credit is accumulated in a per-candidate
# member_accum; each time it crosses 1.0, N_conf increments by 1.
THETA_MIN_PRESENT_RATIO = 0.5  # [P0-3] partial re-observation presence floor
TAU_MIN = 1.0  # §4.7 proximity smoothing floor
TAU_SAMPLE_SIZE = 100  # §4.7 pairwise sample cap

# Value function weights (§4.6.2)
BUILDER_ALPHA = 0.4  # F_expected weight
BUILDER_BETA = 0.4  # F_recursive weight
BUILDER_GAMMA = 0.2  # W weight

# Path/outcome pairs (§4.15)
W_SUCCESSOR = 20  # Steps to observe outcome after path
THETA_PAIR_TRIALS = 3  # Trials before emitting (P, S) pair

# Seeding, expansion, MCS backtracking
W_SEEDING_MAX = 500
BUILDER_EXPANSIONS_PER_STEP = 1
K_AMB = 100  # Typed-constrained MCS backtracking cap

# Analogical registration cadence (§8.11)
K_ANALOGICAL = 500

# Initial reliability scaling (§5.6, §4.22)
KAPPA_INIT = 0.8

# Surprise interaction with CF (§8.6.3)
ETA_CF_SURPRISE = 0.3

# [Phase 1 Decision 1] ConceptGraph three-signal pruning thresholds.
THETA_A = 0.05
THETA_RF = 0.001
# Fertility-protected pruning (§P3). Concepts with EF_combined >= this
# threshold are exempt from pruning: they are currently fertile, so
# culling them destroys downstream structure.
# NOTE: the spec (§P3) writes this as 0.1 * RF_MAX. With RF_MAX = 1e9
# this resolves to 1e8, a very high bar against realistic EF_combined
# values (clamped at RF_MAX but typically far below it). Expect the
# gate to be a no-op on the first landing; calibrate from
# `pruning_ef_max` in the concept_graph diagnostics once pruning
# activates.
THETA_FERTILE_PROTECT = 0.1 * RF_MAX
# §P3 specificity gate. spec = 1 / (1 + KAPPA_SPEC · avg_activation_ema)
# with KAPPA_SPEC = 4 and avg ∈ [0, 1] (activation is binary via
# heaviside → avg is a convex combination over {0,1}), so
# spec ∈ [0.2, 1.0]. Prunable requires spec < THETA_PRUNE_SPECIFICITY,
# i.e. avg > 0.25. Concepts firing less than a quarter of steps are
# protected as selective; concepts firing a quarter or more are treated
# as hub-like and are pruning-eligible (subject to the other gates).
THETA_PRUNE_SPECIFICITY = 0.5
# [Phase 1 Decision 1] Per-step eviction cap: prevents mass extinction
# when many dormant concepts cross the three-signal threshold in one step.
MAX_PRUNE_PER_STEP = 10

# =============================================================================
# 19. CONSOLIDATOR
# =============================================================================
# §5.13. Constants not already defined above (THETA_PAIR_TRIALS, K_CONSOLIDATE,
# CORE_SIZE_CAP_*, KAPPA_INIT are shared and defined in §5/§18).

# [DESIGN OBSERVATION] K_CONSOLIDATE = 500 means zero transfer ever
# fires in episode one on any episodic benchmark (rooms 490, tool 98,
# minigrid 597 — all end before the first tick). This is correct for
# continuous/long-running workloads. For episodic benchmarks, consider
# firing the first tick earlier or at episode end. Not fixing now;
# flagging for future design review.
K_CONSOLIDATE = 500  # §4.22, §5.13
# [G.2.2] Maximum number of real extraction attempts for a concept
# whose category-core extraction returned None. A budget-starved
# extraction succeeds on the next sweep; a genuinely unextractable
# concept is abandoned after this many attempts. Bounds sweep cost
# without sacrificing the starvation recovery this parameter exists
# to enable. The abandonment happens on the (N+1)-th sweep encounter
# without calling the extractor again.
MAX_CATEGORY_EXTRACTION_ATTEMPTS = 3
THETA_CATEGORY_INSTANCES = 5
THETA_CATEGORY_DIVERSITY = 0.5
THETA_CONTAINMENT = 0.95
THETA_GROWTH_PLATEAU = 0.01
M_PLATEAU = 100
KAPPA_RELIABILITY = 0.05
# [K.7] Core minimization floor. _minimize_core's only constraint is
# _classify_prefiltered (containment against every positive, no negative);
# a tiny core satisfies that vacuously -- a 2-edge core scores 1.0 against
# every positive -- so the floor is what stops the minimizer collapsing a
# core to noise. floor = max(MIN_CORE_STRUCTURAL_ELEMENTS,
# int(MIN_CORE_SIZE_RATIO * pre_edges)).
MIN_CORE_STRUCTURAL_ELEMENTS = 3      # absolute edge floor
MIN_CORE_SIZE_RATIO = 0.3             # retain >= 30% of original edges

# [G.2.1] Despite the name, this bounds MCS calls per EXTRACTION,
# not per step. Multiple extractions at one step each get a full
# budget (see consolidator._begin_extraction_budget for why). Renaming
# deferred to avoid a cross-module parameter change.
#
# [K.7 -> Phase L] The K.7 branch carried a raise to 100000 to test whether
# the budget blocked classification. It does not: the classify-only probe
# (K7_classify_probe_report.md) showed the entry classification is False for
# structural reasons, and the budget-10000 harness run died of OOM (the memo
# key is ~302 KB/entry, K7_budget10k_report.md). Reverted to 1000 here, per
# the standing constraint, while restoring the K.7 minimizer itself.
MCS_BUDGET_PER_STEP = 1000
# [L.1] Near-miss minimization. When the extension loop exits without
# classifying, _minimize_core can still run in "near_miss" mode: a removal is
# accepted iff it strictly increases min_pos - max_neg, subject to no negative
# crossing THETA_CONTAINMENT - NEAR_MISS_NEGATIVE_MARGIN. Entry requires a
# core that is close (every positive >= NEAR_MISS_THRESHOLD, no negative
# within the margin).
NEAR_MISS_THRESHOLD = 0.70            # min_pos must exceed this to enter
NEAR_MISS_NEGATIVE_MARGIN = 0.10      # max_neg must be below THETA_CONTAINMENT - this
NEAR_MISS_MCS_BUDGET = 4000           # per-extraction MCS call cap for near-miss mode

# --- DETERMINISM GUARD -- NOT A TUNING KNOB --------------------------------
# _EVID_TIE_GRID is the bucket width of the lexicographic tie-break in
# consolidator._find_best_extension (node-extension AND edge-extension sites):
#     key = (round(score / _EVID_TIE_GRID), tb)
# It must sit strictly BETWEEN float noise and the smallest real score gap:
#   * above float jitter (~1e-11), so nondeterministic last-bit differences
#     land in the same bucket and the content hash resolves them;
#   * below the smallest observed positive top-two gap
#     (2.7604159041905163e-08 at seeds 0 and 42, on the graded-param tree
#     after the Pass-23 edge vocabulary landed: probe's distinct margin
#     values = 17 per seed, so param_sim genuinely varies), so genuine
#     distinctions stay in different buckets and the hash never overrules
#     the score.
# The earlier coarse value (1.38e-08, from a pre-calibration gap of
# 1.38e-07) left only a 2x margin below the then-current min gap -- and a
# coarser grid (e.g. 0.0017857) would bucket both members of such a
# top-two pair together and hand the decision back to the hash --
# reintroducing the bug for exactly the near-ties that matter most.
# Derivation: min_positive_gap / 10 = 2.76e-08 / 10. DO NOT tune this
# against benchmark numbers -- it is a determinism guard. A grid at or above
# the minimum real gap collapses every score into one bucket and turns
# selection into a pure hash lottery, reintroducing (worsening) the bug this
# exists to fix. If a future probe run reports a smaller minimum positive
# gap, re-derive it (min_gap / 10) instead of hardcoding.
# Sanity: grid 2.76e-09 >> float noise 1e-11 (276x), and << min gap
# 2.76e-08 (10x) -- strictly inside the required window.
_EVID_TIE_GRID: float = 2.76e-09

# [Bundle A] Per-call MCS memo cap (discriminative_core scope). 1.26M
# submissions/tick collapse to ~196 unique pairs; 100k entries bounds the
# memo far above any single-tick unique set while capping memory.
MCS_MEMO_MAX_ENTRIES = 100_000
# [Bundle B0] Three-tier classification prefilter master switch. Verification
# toggle: running with False reproduces the unfiltered baseline for core
# library identity comparison.
B0_PREFILTER_ENABLED = False

# [B3/§5.8.2] Observation-window ring buffer (discriminative trigger).
# Capacity must admit θ_instances distinct positives with headroom for
# diversity; max age prevents fossil windows from ever firing the trigger
# (K2). Windows are content-deduplicated, so capacity bounds STRUCTURALLY
# DISTINCT situations per label, not raw steps.
LABEL_WINDOW_CAPACITY = 8
LABEL_MAX_AGE_STEPS = 1000
LABEL_MIN_INSTANCES = THETA_CATEGORY_INSTANCES

# [Batch D3] Category-producer unification. §5.8.2's diversity condition is
# no longer a hard TRIGGER predicate: the ObservationWindowBuffer's own
# diversity gate duplicated THETA_CATEGORY_DIVERSITY and, because buffered
# positives are content-deduplicated windows that frequently sit in one
# spatial region, it silently blocked every category core on structurally
# uniform label windows (the gate could never be satisfied for the very
# labels the §5.8.2 trigger exists to consolidate).
#
# Diversity now SOFTENS the new core's initial reliability instead:
#     factor      = clamp(div(W+) / TARGET_DIVERSITY, MIN_RELIABILITY_FACTOR, 1)
#     reliability = KAPPA_INIT * factor
# so a uniform positive set still produces a core, but one that starts
# with MIN_RELIABILITY_FACTOR of the reference reliability and must earn
# reliability (§5.6 κ_reliability) before it can drive the Mapper.
#
# MIN_DIVERSITY_FLOOR remains a hard floor on degenerate positive sets
# (structural spread ≈ 0), which carry no discriminating structure at all.
TARGET_DIVERSITY = 0.30
MIN_RELIABILITY_FACTOR = 0.25
MIN_DIVERSITY_FLOOR = 0.01

# §5.3.2 autonomous template materialization — reserved node-ID namespace.
# Materialized template nodes are NOT pattern nodes in the graph's store;
# they must never alias a live pattern-node ID (§1.11's single monotonic
# counter allocates upward from 0 and cannot reach this band), nor collide
# with previously materialized blocks (the consolidator-owned allocator
# hands out strictly increasing blocks from this base). Within signed
# 64-bit serialization range (§2.6 byte-level closure).
TEMPLATE_NODE_ID_BASE = 1 << 62

# =============================================================================
# 20. MAPPER
# =============================================================================
# §6.8. THETA_CONTAINMENT and MAX_SITUATION_SIZE are defined above.

H_SITUATION = 2
THETA_SITUATION_EDGE = 0.2
TAU_TRANSFER = 0.4
QUERY_CACHE_TTL = 10
# [Batch D2] Per-step cap on §6.6.2 "What is this?" queries issued by the
# Pipeline — the category-query CALLER. `Mapper.query_category` existed with
# no caller at all: CATEGORY cores were extracted (Trigger 2A and the
# label-window trigger) but never queried, so their reliability was never
# exercised and `category_labels_returned` could only ever be 0. The cap
# bounds per-step caller cost (each query runs a §6.4 identification over
# the situation subgraph) while covering every informative label in use
# (≤ 4 distinct effect atoms are simultaneously present on the environments
# in use). Deterministic order: labels are sorted by serialize_value.
K_CATEGORY_QUERY_MAX = 4
THETA_CATEGORY_MIN_POSITIVES = 3

# [DEPRECATED in Step 6; will be deleted in Step 8] Legacy MCS thresholds
THETA_MCS_BY_FAMILY = {
    "EVENT_CAUSAL": 0.6,
    "TEMPORAL_CHAIN": 0.7,
    "SPATIAL_OBJECT": 0.5,
    "RECURSIVE_COMPOSITION": 0.6,
    "CROSS_DOMAIN_PATTERN": 0.5,
    "CHUNK": 0.5,
}
THETA_MCS_DEFAULT = 0.5
THETA_MCS_CATEGORY = 0.6

# [Step 6a] Evidence readout thresholds
# DEPRECATED (Stage 2): Superseded by EVIDENCE_LAMBDA_NODE + EVIDENCE_LAMBDA_EDGE.
# Retained for backward compatibility with external probe scripts.
EVIDENCE_LAMBDA_ABSENCE: float = float(
    os.environ.get("EVIDENCE_LAMBDA_ABSENCE", "0.0")
)
if EVIDENCE_LAMBDA_ABSENCE < 0.0:
    raise ValueError("EVIDENCE_LAMBDA_ABSENCE must be >= 0")

THETA_EVIDENCE_POS: float = float(
    os.environ.get("THETA_EVIDENCE_POS", "0.35")
)
THETA_EVIDENCE_NEG: float = float(
    os.environ.get("THETA_EVIDENCE_NEG", "0.60")
)
EVIDENCE_GAP_THRESHOLD: float = float(
    os.environ.get("EVIDENCE_GAP_THRESHOLD", "0.05")
)
EVIDENCE_MIN_BALANCED: float = float(
    os.environ.get("EVIDENCE_MIN_BALANCED", "0.15")
)

THETA_EVIDENCE_BY_FAMILY: Dict[str, float] = {
    "EVENT_CAUSAL": 0.30,
    "TEMPORAL_CHAIN": 0.35,
    "SPATIAL_OBJECT": 0.25,
    "RECURSIVE_COMPOSITION": 0.30,
    "CROSS_DOMAIN_PATTERN": 0.25,
    "CHUNK": 0.25,
}
THETA_EVIDENCE_DEFAULT: float = 0.25
THETA_EVIDENCE_CATEGORY: float = 0.35

# Phase 2/4 — identity tier.
# Strong: score >= (family_threshold + THETA_IDENTIFY_DELTA).
# Weak:   passes match_all's family filter but < strong.
# Delta = 0.0 disables the tier (every match strong); 4-mode
# classification degenerates to FIRED-only. Useful diagnostic fallback
# if the coverage gate proves too aggressive on a given benchmark.
THETA_IDENTIFY_DELTA = 0.1

# Replaces THETA_FIRED_COVERAGE as the trivial-match gate. A match
# must have BOTH:
#   (a) at least MIN_MATCHED_ELEMENTS matched elements
#       (matched nodes + matched edges), and
#   (b) at least MIN_MATCHED_EDGES matched edges.
# Query-size-independent. SC11's query-relative coverage gate blocked
# legitimate small-core matches: 2-node/5-edge cores scored 0.714 on
# 30-node queries with coverage 2/30 = 0.067, below the 0.15 gate.
#
# Audit C2: matched_elements >= 4 alone admits 4-node/0-edge
# co-occurrence matches — the exact trivial class SC11 blocked, just
# larger. The edge companion closes that hole without affecting the
# rooms target (2 nodes / 3 edges = 5 elements / 3 edges).
# DEPRECATED (H.4b.7): superseded by MIN_MATCHED_ELEMENTS_PCT and
# MIN_MATCHED_EDGES_PCT. Retained for external reference and for mapper.py's
# import-time range guards; no longer consumed by the identity gates in
# query_identification.
MIN_MATCHED_ELEMENTS = 4
MIN_MATCHED_EDGES = 2

# [DEPRECATED in Step 8: superseded by evidence readout in query_identification]
# [H.4b.7] Percentage-based identity gates. The old absolute gates were
# calibrated for Hungarian's denser correspondences. The sparse greedy
# matcher can produce fewer matched pairs on large cores, so an absolute
# integer gate falsely rejects valid matches.
#
# FLOOR = 0 for edges: edgeless cores/queries must not permanently fail
# identity. The gate reads: matched_edges >= max(FLOOR, int(PCT * min_size)).
MIN_MATCHED_ELEMENTS_PCT = 0.15
MIN_MATCHED_ELEMENTS_FLOOR = 2
MIN_MATCHED_EDGES_PCT = 0.10
MIN_MATCHED_EDGES_FLOOR = 0


# DEPRECATED. Retained for external reference. No longer used as a
# gate; coverage is computed per-match as a diagnostic scalar
# (`identity_coverage_max` on the Mapper, cumulative across episodes).
THETA_FIRED_COVERAGE = 0.15

# Per-bucket cap on Identity objects. match_all sorts by score
# descending; the top-N per bucket survive. Dropped matches increment
# `identities_capped_this_call`.
MAX_IDENTITIES_PER_BUCKET = 20

# Phase 3 (opt-in; not wired by default).
H_FIRE = 2
LAMBDA_ACT = 0.7
THETA_SITUATION = 0.2

# [Batch E] Near-miss edge decay / recency inclusion in situation subgraphs.
# An edge below THETA_SITUATION_EDGE is admitted into the induced situation-
# subgraph edge set iff it carries recent POSITIVE evidence via one of:
#   (2) e.observed_at >= t - W_SITUATION_RECENCY and e.pending_signal > 0.0
#   (3) e.time_since_coact <= TAU_SITUATION_COACT
# The pending_signal gate on clause (2) closes audit E6-03: observed_at is
# stamped on every upsert (including signal=0.0), so without the gate a
# freshly stamped zero-signal edge would enter the situation as structure
# for ~21 steps. Clause (3)'s counter resets only on signal > 0, so it is
# positive-evidence by construction.
# Spreading weight is unaffected (RULE 3).
W_SITUATION_RECENCY = 20
TAU_SITUATION_COACT = 5

# [Phase 1 Decision 4 / P2-7] Similarity-graph parameters.
SIMILARITY_EDGE_THRESHOLD = 0.60
MAX_SIMILARITY_NEIGHBORS = 20
N_SIMILARITY_ACTIVATE = 50
K_SEED = 3
K_NEIGHBOR_SCAN = 100
MAX_HOPS = 2
THETA_TRUST_NEIGHBORHOOD = 0.85
N_FORCE_FLAT = 10
# [v13 / SC-YY] Transient anti-ossification state cap; cleared on exceed.
SIMILARITY_STATE_CAP = 500
# [v13 / SC-XX] Persistent similarity-graph cap; liveness-filtered on exceed.
SIMILARITY_EDGE_CAP = 5000

# =============================================================================
# STRUCTURAL SIMILARITY OPERATOR (§1.5, §4.12, §6.4.1)
# =============================================================================
# Weights of the closed-form structural score. The operator itself lives in
# structural_similarity.py (pure, stateless — it is NOT a binder and emits no
# edges). Same weights are used for the node and the edge dimension.

STRUCT_SIM_W_SIZE = 0.3
STRUCT_SIM_W_TYPE = 0.3
STRUCT_SIM_W_ALIGN = 0.4
STRUCT_SIM_ALPHA = 0.5
STRUCT_SIM_MIN_OVERLAP = 0.5

# Bounded fixed-point budget for organization_alignment's relational
# refinement (rounds of Hungarian re-assignment over grounded pair quality
# q = sig_sim × relational agreement; iteration exits early once the
# correspondence stabilizes, so typical cost is 2-3 rounds). Coupled
# cross-type correspondences relax roughly one type-layer per sweep, so
# the budget scales as ~2 sweeps over the lens-type count; polynomial per
# round, and multi-start selection (organization_alignment) arbitrates
# rival correspondence basins by total grounded quality.
STRUCT_ALIGN_REFINE_ROUNDS = 6

# =============================================================================
# PHASE 5 -- FUZZY GRADED MATCHER (matching/alignment.py, calibration.py,
# consensus.py, search.py, fuzzy.py)
# =============================================================================
# The robust fuzzy graded matcher. Node similarity is the population-overlap
# similarity already produced by TypedSignature.similarity; this layer adds
# global-consistency alignment (seeded from the approximate Hungarian),
# an explicit unmatched cost, symmetric coverage, systematicity weighting,
# null-calibrated mass with the 0.85 conjunction floor REMOVED, background-
# calibrated per-class scoring, multi-vote consensus with stored disagreement,
# and best-first extension search with anytime budgets.

# Use global-consistency alignment for correspondence in the live MCS path.
# DARK by default (A/B in the Phase 6 protocol): it changes every MCS score, so
# it is built and switchable now and validated before any default flip. The
# mechanism is fully live via matching/fuzzy.py regardless of this flag.
MATCH_GLOBAL_CONSISTENCY_ENABLED = False
# Explicit price of leaving a node unmatched. 0.0 = the classic "any positive
# similarity may match" behaviour; > 0 makes unmatched a priced decision.
MATCH_UNMATCHED_COST = 0.0
# Weight of the relational (systematicity) modulation of the alignment.
MATCH_SYSTEMATICITY_WEIGHT = 0.5
# Bounded fixed-point rounds for the global alignment refinement.
MATCH_ALIGN_ROUNDS = 4
# Pair-evaluation budget for the refinement. None = unbounded.
MATCH_ALIGN_MAX_PAIR_EVALS = 200000
# Weight of the relational term in the combined fuzzy score (the node term is
# 1 - this). This is a blend weight, NOT the removed 0.85 additive floor.
MATCH_EDGE_WEIGHT = 0.5
# Null-calibrated mass + floor-free conjunction in the live evidence path.
# When True, evidence_score calibrates the node coverage against the null and
# uses conjunction() with floor 0.0 (the historical 0.85 floor is retired).
MATCH_NULL_CALIBRATED = True
# Minimum support weight for a consensus edge to be kept (no invented edges).
MATCH_MIN_CONSENSUS_SUPPORT = 0.5
# Best-first extension search budgets.
MATCH_SEARCH_MAX_EXPANSIONS = 4096
MATCH_SEARCH_MAX_OPS = 200000

# =============================================================================
# LOCOMOTION BRIDGE & ANCHOR FALLBACK
# =============================================================================
# Maximum anchor distance (Euclidean, metres/cells) at which the UtilityEngine's
# targeted anchor fallback may hand the reflex a direction proposal. Beyond it
# the concept is "far away" and the AEM locomotion bridge must be used instead,
# so a distant anchor cannot masquerade as a valid terminal affordance.

ANCHOR_FALLBACK_MAX_DIST = 1.5

# =============================================================================
# 21. EXECUTOR
# =============================================================================
# §7.8.

N_EDGE = 5  # Minimum edge mass for traversability
EDGE_COST = 1.0  # Unit cost
MASS_TIEBREAKER = True  # Prefer higher total mass among equal-cost
MAX_EXPANSIONS = 10000  # A* expansion bound
THETA_TARGET_CONTAINMENT = 0.95

# =============================================================================
# 22. PREDICTION
# =============================================================================
# §8.8. W_CONFIRM, MAX_PENDING, MAX_SURPRISE_QUEUE, MAX_SURPRISE_PER_STEP,
# ETA_SURPRISE, ETA_CF_SURPRISE, THETA_PROTO_EF, TAU_PROTO_DECAY,
# N_PROTO_MAX_EBAF are defined in earlier sections.

GAMMA_CF = 0.99  # Counterfactual decay per step
RHO_SMOOTH = 0.9  # EMA for smoothed causal probabilities
N_CF_MIN = 5.0
C_HYST = 0.5

# [Batch D1] Tiered necessity commit (§8.3.3). N_cf = min(C_opp, C_opp_not)
# grows only as fast as the action is executed: a sparse/one-shot effect
# (rooms toggles fire ~26x per 2000 steps; the tool's effects are
# one-shot) never reaches N_CF_MIN = 5 within an episode, so H_nec could
# never be 1 and no binding was ever committed -- even when a single
# execution produced decisive contrast evidence. The commit point is now
# THETA_CF_COMMIT; the required contrast then ramps with
# sqrt((N_cf - THETA_CF_COMMIT) / (N_CF_MIN - THETA_CF_COMMIT)) from
# THETA_CONTRAST_COMMIT to THETA_CONTRAST_STABLE, so:
#   * 1-2 confirmations commit only against an unambiguous contrast;
#   * at N_cf >= N_CF_MIN the criterion is the historical
#     THETA_CONTRAST_STABLE reading (Gate_bind / Gate_undo still demand
#     N_cf >= N_CF_MIN, unchanged).
# Recalibrate THETA_CF_COMMIT upward if the 2000-step watch shows
# spurious commits (weak links entering the library).
THETA_CF_COMMIT = 0.50
THETA_CONTRAST_COMMIT = 0.30
THETA_CONTRAST_STABLE = 0.40

# [Batch D1b] Deferred causal-binding re-attribution (design draft §4).
#   W_REATTRIB     — steps a VOID entry (rise buffered with cid=None
#                    because no causal concept was ACTIVE to claim it)
#                    stays claimable. Measured need on tool: claim window
#                    [62, 76] = 15-17 steps; 60 caps memory with wide
#                    margin. Do NOT extend beyond 60 without a fresh
#                    measurement.
#   W_HOMED_TTL    — steps a homed-but-uncommitted triple waits for H_nec
#                    before the claim is released (its own clock, not the
#                    void clock).
#   K_REATTRIB_MAX — per-step cap on homing claims; bounds the
#                    O(voids x onsetting) scan (deterministic FIFO —
#                    oldest voids homed first).
W_REATTRIB = 60
W_HOMED_TTL = 20
K_REATTRIB_MAX = 16

KAPPA_REL = 0.05  # Transfer reliability EMA rate
W_VALIDATION = 20  # Transfer prediction horizon
W_STALE = 100  # Pending staleness pruning
THETA_SURPRISE = 0.4  # Alarm threshold

# -----------------------------------------------------------------------------
# 22.1 CAUSAL COUNTERFACTUAL HYSTERESIS GATES (§8.3.3)
# -----------------------------------------------------------------------------
# Single source of truth for the necessity-gate hysteresis constants.
# (N_CF_MIN itself is declared in Section 22 above; not duplicated here.)

BAR_DP_ENTER_THRESHOLD = 0.40
BAR_P_A_ENTER_THRESHOLD = 0.60
BAR_P_NOT_A_ENTER_THRESHOLD = 0.20
C_HYST_NECESSITY = 0.5
EPS_HYST_NEC_MIN = 0.02
EPS_HYST_NEC_MAX = 0.20

# =============================================================================
# 23. UTILITY ENGINE (v30 §2.20)
# =============================================================================
# §2.20.3 is a CLOSED FORM: U_raw = U_cap·[Σ ŵᵢ·Normᵢ + ŵ_persist·PlanPersist
# − stay_discount(a_null)] − η_conflict·Conflict. The weights below are the
# full set of constitutional utility constants. (The former "§23.1 formula
# coefficients" — reversal penalty, plan bonus, orient gain, destination
# richness, adaptive inertia — had no blueprint basis and were removed as
# drift. The stay discount is REINSTATED by auditor ruling R5 §2.4 at its
# standard v30 value/placement: it is the constitutional bias against the
# §2.6 null action, ensuring exploratory movement always beats passive
# idling when every other utility source is tied at zero.)

U_CAP = 50.0
# v30 §23.1 standard stay discount (auditor ruling R5 §2.4). Applied inside
# the U_cap bracket for exactly the constitutional no-op class (a_null:
# None/"null"/"stay", §2.6 — primitives.is_noop_action), never by env action
# names. Effective magnitude U_STAY_DISCOUNT × U_cap = 5.0 utility points,
# strictly above the maximum a targetless action can score at a zero tie
# (w6·MotorNorm·U_cap/W_total ≤ 0.4·50/4.1 ≈ 4.88): idling is a last resort
# while goal/frontier-driven utilities (≫ 0.1 in bracket terms) keep full
# weight — no hard bypass, no forced action.
U_STAY_DISCOUNT = 0.10
OMEGA_PROBE = 100.0
ETA_PLAN_PERSIST = 0.1
SIGMA_NOISE = 0.01
W_FERT = 1.0
W_ABSTRACT = 0.8
W_REAL = 0.7
W_PROTECT = 0.6
W_CONSOL = 0.5
W_MOTOR = 0.4

# v30 §2.1.12 — action selection and utilities (single source of truth).
ETA_LAYER = 0.1            # η_layer abstraction bias
M_LAYER_MAX = 2.0          # M_layer^max abstraction multiplier cap
THETA_ABSTRACT = 0.1       # θ_abstract (FertNorm floor for goal proposals)
ETA_HYSTERESIS = 0.1       # η_hysteresis plan-switch hysteresis
ETA_FATIGUE = 0.1          # η_fatigue probe fatigue weight
ETA_FATIGUE_DECAY = 0.95   # η_fatigueDecay
TAU_MEANING = 200          # τ_meaning meaningful-child window
XI_ATTEMPT = 1.0           # ξ_attempt motor attempt weight
XI_ENTROPY = 0.5           # ξ_entropy motor entropy weight
ETA_CONFLICT = 5.0         # η_conflict soft conflict penalty
TAU_PROVISIONAL = 500      # τ_provisional provisional ratification timeout
# [H.4b.6] Matcher calibration. The sparse greedy + 2-opt matcher (H.4b)
# scores systematically lower than the historical Hungarian because the
# greedy phase can leave some high-mass nodes unmatched. The following
# drop was measured by H.4b.5 on the ETH-80 9-image battery:
#
#   weighted mean drop = -0.0004  (greedy minus Hungarian; NEGATIVE means
#   the new matcher scored marginally HIGHER, not lower -- the Gestalt
#   boost is active on both arms per PF-8, and the per-lens-type matrices
#   are small enough that greedy loses almost no mass there)
#
# Template-construction thresholds (THETA_SIM, THETA_SIM_CONSENSUS) are
# recalibrated by this drop so that template nodes that would have
# entered under Hungarian still enter under greedy. At 4 decimals the
# recalibration is a no-op (0.7000 -> 0.7004); it is applied as specified
# rather than special-cased, and the stop trigger (drop > 0.20) is not met.
#
# NOTE: THETA_CONTAINMENT and THETA_MCS_BY_FAMILY (core classification
# and Mapper retrieval gates) are NOT recalibrated here. If the harness
# run after this change shows reduced full_identities_fired, those
# thresholds require a follow-up calibration.
_MATCHER_SCORE_DROP: float = -0.0004  # from H.4b.5 weighted mean drop

_OLD_THETA_SIM = 0.7
THETA_SIM = max(0.1, _OLD_THETA_SIM - _MATCHER_SCORE_DROP)   # θ_sim signature similarity threshold
THETA_EXPLORATION = 0.2    # θ_exploration frontier potential threshold
KAPPA_PROTECT = 1.0        # κ_protect consolidation protection weight
PHI_MAX = 1e6              # Φ_max structural potential cap
THETA_DOMINANCE = 0.8      # θ_dominance edge dominance threshold
N_OUTCOME_MAX = 10         # N_outcomeMax max tracked outcomes per edge
KAPPA_EMA = 0.05           # κ_ema spawn-rate EMA rate (§2.20.2)
RHO_SPAWN_STAR_DEFAULT = 0.2  # ρ_spawn* default (env-scaled per v30 §2.1.16)

# --- Frontier-gated goal proposal (cures compulsive re-visitation) ---
# [Drive Remediation Phase 0 / Strategic Fix 1] §2.20.1 goal-subset gate:
# a causal goal concept is proposed only if pursuing it can REACH
# unexplored territory (Φ > THETA_FRONTIER within FRONTIER_HORIZON AEM
# hops of one of its post/enabling states). Reachability — not the local
# post-state Φ alone — so the concept reactivates when the world changes
# (e.g. a door closes and re-opening unlocks new territory).
THETA_FRONTIER = 0.15      # Phi threshold: state counts as "unexplored territory"
# [Batch 4 / P1-2] Homeostatic lower bound for THETA_FRONTIER.
# Frozen at 0.10 after the four-way isolation run: θ = 0.10 pins to the
# safe trajectory (rooms 425). θ = 0.02 admits low-Φ goals and fails
# (rooms stalls at (7,5)). θ = 0.0734 (the live controller's equilibrium)
# also fails — the controller was overshooting into the harmful band.
# See futurework3.md, "Frontier Gate Homeostasis — Partially Resolved".
THETA_FRONTIER_FLOOR = 0.10

THETA_FRONTIER_MAX = 0.40
KAPPA_EMA_FRONTIER = 0.05      # EMA smoothing rate
KAPPA_FRONTIER = 0.05          # theta update rate
RHO_GOAL_TARGET = 0.30         # target pool occupancy
# [Patch L1a] Force-refresh period for the cached frontier-reachability
# gate: bounds staleness from Phi decay so exhausted goals cannot creep
# back in between AEM-growth invalidations.
FRONTIER_CACHE_TTL = 8     # steps before frontier cache is force-refreshed (Phi decay guard)
FRONTIER_HORIZON = 6       # max AEM hops to search for frontier from a post-state
K_PLAN_SCAN = 30           # [L2 REVERTED] trim to 15 was NOT behavior-
                           # preserving: rooms late-game has >15 gated causal
                           # concepts, so the pool content changed and the
                           # solve step moved 329 -> 583 (measured). Phase-0
                           # mandate: latency patches must not change which
                           # actions are selected. Scan depth restored; the
                           # L1 strict-topology cache carries the latency win.
                           # scan this many top-EF concepts before frontier filtering

# --- Fertility-to-locomotion inheritance ---
# [Drive Remediation Phase 0 / Strategic Fix 2] Actions that reduce the AEM
# geodesic distance toward a high-EF goal concept's enabling states earn an
# additive utility bonus (fert_norm × approach_gain × W_FERT_INHERIT).
# Replaces the reflex as the approach mechanism for conceptual goals
# (Strategic Fix 3 decouples the reflex from abstract locomotion).
W_FERT_INHERIT = 0.5       # utility weight for approaching a high-EF concept
FERT_INHERIT_TOP_K = 4     # only inherit from the K strongest active goal concepts

# v30 §2.1.9 — structural potential map (§2.19 update rule).
ETA_DECAY = 0.95           # η_decay potential decay
EPS_EDGE = 0.1             # ε_edge new confirmed edge bonus
NU_NOVEL = 0.2             # ν_novel novelty potential weight
OMEGA_FRONTIER = 0.5       # ω frontier weight
THETA_CLEANUP = 0.01       # θ_cleanup potential cleanup threshold
N_POT_MAX = 10000          # N_potMax max potential entries

# =============================================================================
# 24. INNATE REFLEX & KINEMATIC SALIENCE (12_Update §2.3–§2.5, 13_Update §6–§7)
# =============================================================================

REFLEX_ALPHA_P = 0.20
REFLEX_ETA_DEAD = 1e-6
REFLEX_EPS_M = 1e-9
REFLEX_L_REF = 1.0
REFLEX_K_DIR_MAX = 2
REFLEX_DELTA_RANK = 0.05
REFLEX_T_RETAIN = 10
REFLEX_V_BASE = 1.0
REFLEX_KAPPA_V = 0.001
REFLEX_EPS_V = 1e-9
REFLEX_V_MAX = 1e3
REFLEX_R_ENV_MIN = 1.0
REFLEX_M_EXP = 50.0
REFLEX_W_D = 0.30
REFLEX_W_V = 0.30
REFLEX_W_C = 0.25
REFLEX_W_N = 0.15
REFLEX_V0 = 0.15
REFLEX_THETA_G = 0.25
REFLEX_DELTA_HYST = 0.05
REFLEX_BETA_G = 10.0
REFLEX_GAMMA_MIN = 0.60
REFLEX_KAPPA_C = 0.10
REFLEX_T_DWELL = 2
REFLEX_BETA_VEL = 50.0
REFLEX_THETA_VEL = 0.05
REFLEX_V_THRESH = 1e-2
REFLEX_TAU_KIN_SAT = 100
REFLEX_PSI_CLIP = 0.999999
REFLEX_EPS_DEAD_MOTOR = 0.04

# =============================================================================
# 25. SATIETY & LOCKOUTS (v30 §2.1.12)
# =============================================================================

TAU_SATIETY = 30
TAU_FAIL = 20
TAU_FAIL_MAX = 200
TAU_NAV_REFRACTORY = 5
N_MOTOR_FAIL_MAX = 3

# =============================================================================
# 26. EDGE TYPE CLASSIFICATION
# =============================================================================
# §2.8 and §4.14. These frozensets are used by:
#   - binder.py: enforcement of self-loop permission (§3.4)
#   - builder.py: emission restriction (§4.14)
#   - concept_graph.py: per-type edge dispatch (§1.12)

# The consolidated spatial edge categories. Kept as names so parameter
# imports do not need to import the EdgeType enum.
EDGE_TYPE_NAMES = frozenset({
    "PROXIMITY",
    "CONTINUITY",
    "CONTAINMENT",
    "OCCLUSION",
    "JUNCTION",
    "SURFACE",
    "SHAPE",
    "MATERIAL",
})

# §3.4: only these types may have parent == child (none under v31 without self-binders).
SELF_BINDER_TYPE_NAMES = frozenset()

# Every canonical spatial relation may contribute to a family; there are no
# scaffold-only edge categories in the consolidated taxonomy.
NON_FAMILY_FORMING_EDGE_TYPES = frozenset()
FAMILY_FORMING_EDGE_TYPES = EDGE_TYPE_NAMES

# §2.3: order-2 binder edge types. The structural-similarity comparison was
# moved out of the Binder into structural_similarity.py (a pure operator that
# emits no edges), so there are NO order-2 edge types in v31. The set is kept
# as the single extension point for a future order-2 binder.

# §3.3: layer computation uses edge types, not member layers.
# edge type exists; the duplication with ORDER_2_EDGE_TYPES is intentional so
# that the layer computation and the emission restriction can be tuned
# independently if the architecture diverges.
EDGE_TYPES_WITH_LAYER_1 = frozenset()

# =============================================================================
# 27. LEGACY (v30 constants retained for backward compatibility)
# =============================================================================
# These constants are used by v30 code paths that have not yet been
# migrated. They will be deleted once v31 migration is complete.
# Values are the v30 §2.1 canonical defaults.

BFS_MAX = 10          # v30 §2.1.7 BFS_max
D_MAX = 10
C_PLAN_MAX = 100      # v30 §2.1.12 C_planMax
TAU_PLAN_MAX = 200    # v30 §2.1.12 τ_planMax
STUCK_PLAN_STEPS = 100
K_PLAN = 20
B_MAX = 3
N_SLOT_CAND_MAX = 5
LAMBDA_DIST = 0.15
PI_ABANDON_BASELINE = 0.5
PI_ABANDON_MIN = 0.1
PI_ABANDON_MAX = 1.0

# =============================================================================
# 28. 3D SENSORY ADAPTER (13_Update §3–§4) — CANONICAL VALUES
# =============================================================================
# Layer-0 preprocessor constants. These are the 13_Update specification
# values; they are scale-appropriate for metric workspaces and MUST NOT be
# re-tuned per environment (the earlier drift — R_CLUSTER 0.12, θ_depth
# 0.035/0.018, R_MIN_SEP 0.05, τ_persist 30, grid 128×96 — was tabletop
# hyper-tuning and has been reverted to spec).

ADAPTER_M_KEYPOINTS = 100        # §3.2 M ≤ 100 sparse corner keypoints
ADAPTER_R_CLUSTER = 0.20         # §3.2 R_cluster (m) common-fate co-locality
ADAPTER_EPS_VEL = 0.03           # §3.2 ε_vel (m/s) common-fate velocity coherence
ADAPTER_V_THRESH = 0.02          # §3.2 v_thresh (m/s) kinematic instantiation
ADAPTER_MIN_CLUSTER_SIZE = 4     # §3.2 |K_u| ≥ 4 keypoints
ADAPTER_R_MIN = 0.04             # §3.2 r_min (m) minimum bounding radius
ADAPTER_R_MAX_OBJ = 0.50         # §4.3 R_max_obj (m) normalization reference
ADAPTER_THETA_DEPTH_HIGH = 0.25  # §3.3 Schmitt upper threshold (m), room-scale default
ADAPTER_THETA_DEPTH_LOW = 0.12   # §3.3 Schmitt lower threshold (m), room-scale default
# ARCHITECT RULING (Item 2): θ_depth is a PHYSICAL UNIT CALIBRATION of the
# depth-discontinuity detector, not a cognitive constant — it must match the
# scene's metric scale. A 6 cm cube cannot produce a 25 cm depth step, so the
# tabletop deployment declares its own calibration through AdapterConfig.
THETA_DEPTH_HIGH_3D_TABLETOP = 0.035  # m — tabletop-scale upper threshold
THETA_DEPTH_LOW_3D_TABLETOP = 0.018   # m — tabletop-scale lower threshold
ADAPTER_R_MIN_SEP = 0.25         # §3.3 R_min_sep (m) 3D NMS merge separation
ADAPTER_EPS_GATE = 0.15          # §3.4 ε_gate (m) predictive centroid gating
ADAPTER_TAU_PERSIST = 10         # §3.4 τ_persist (frames) occlusion memory
ADAPTER_ALPHA_P = 0.20           # §6.1 α_p position low-pass coefficient
ADAPTER_BETA_T = 0.90            # §4.3 Ch3 β_T temporal-coherence EMA
ADAPTER_DEPTH_GRID_W = 64        # §3.3 decimated depth grid width
ADAPTER_DEPTH_GRID_H = 48        # §3.3 decimated depth grid height
ADAPTER_CYCLE_THRESH_PX = 1.0    # §3.2 bidirectional LK cycle-consistency (px)
ADAPTER_ANCHOR_PRIOR_MARGIN = 0.02  # margin on anchor bounding sphere (m)
ADAPTER_RGB_WINDOW_RADIUS = 2    # per-anchor mean-RGB window radius (px), v31 Color lens

# Optical-flow / corner-detector algorithm internals (implementation tuning;
# not constitutional, but centralized here per the single-source rule).
ADAPTER_LK_WIN = 21
ADAPTER_LK_LEVELS = 3
ADAPTER_LK_MAX_ITER = 30
ADAPTER_LK_EPS = 0.01
ADAPTER_ST_QUALITY = 0.01
ADAPTER_ST_MIN_DIST = 7
ADAPTER_ST_BLOCK = 3

# [PHASE F] The Size lens is PURGED: on a uniform patch grid every region
# has identical radial extent, so the lens emitted one constant bin per
# patch (a tautology) and polluted the Part signature. The lens body is
# retained in lenses.py but run_all only calls it when this flag is set.
SIZE_LENS_ENABLED: bool = False
# Size-lens normalization constants were only consumed by the purged lens;
# retained for reference (adapter radii remain the single definition).
# SIZE_RADIUS_MIN = ADAPTER_R_MIN
# SIZE_RADIUS_MAX = ADAPTER_R_MAX_OBJ

# =============================================================================
# 28.1 RECOGNITION / CATEGORIZATION (v30 §2.1.21, §2.31)
# =============================================================================
# Consumed by derived_lenses (recognition homeostasis) and concept_graph
# (disablement cooldowns). Values are the v30 canonical defaults.

THETA_CATEGORIZE = 0.7        # θ_categorize(t) initial categorization threshold
K_CATEGORIZE = 5              # K_categorize max recognition matches per step
KAPPA_CAT = 0.05              # κ_cat categorization homeostasis gain
RHO_CAT_TARGET = 0.2          # ρ_cat* target categorization rate
THETA_CATEGORIZE_MIN = 0.5
THETA_CATEGORIZE_MAX = 0.95
DELTA_THETA_CAT_MAX = 0.05    # Δθ_cat^max max threshold step
TAU_PROGRESS = 500            # τ_progress recognition progress failure window
RHO_PROGRESS_MIN = 0.1        # ρ_progress^min normalized fertility for progress
TAU_REVERT = 3                # τ_revert recognition failure streak threshold
DELTA_CAT = 0.02              # δ_cat recognition hysteresis margin

# =============================================================================
# 28.2 CONSOLIDATOR DISCRIMINATIVE MODE (v31 §5.3.2)
# =============================================================================

_OLD_THETA_SIM_CONSENSUS = 0.8
THETA_SIM_CONSENSUS = max(0.1, _OLD_THETA_SIM_CONSENSUS - _MATCHER_SCORE_DROP)     # consensus-signature graded similarity threshold

# =============================================================================
# 29. MISCELLANEOUS STRUCTURAL BOUNDS
# =============================================================================

TAU_RECOMPUTE_CADENCE = 100      # §4.7 τ recomputation cadence (steps)
TAU_ACTIVE_DRIFT_FRACTION = 0.2  # §4.7 active-set symmetric-diff drift trigger
STATE_PATTERN_INDEX_CAPACITY = MAX_EXPANSIONS  # pipeline_support StatePatternIndex;
                                 # derived: any state the Executor could expand
                                 # to (bounded by max_expansions) stays retained.
NODE_DETECTION_SWEEP_PERIOD = PATTERN_NODE_TTL * 10  # Builder node→detection map sweep
NODE_DETECTION_BLOAT_FACTOR = 2  # sweep triggers above factor × live node count
DIAG_INTERVAL = 25               # v30 §4.7 diagnostic sampling default (steps)
EBAF_TEST_MODE = False           # v30 §2.32.8 test-only credit injection hook
EBAF_T_INJECT = 1000             # injection step when EBAF_TEST_MODE is enabled
GAP_QUEUE_MAX = 100              # Mapper bounded GapSignal queue
CATEGORY_WINDOW_MAX = 10         # Trigger 2A: max positive (and negative)
                                 # situation windows fed to discriminative_core
                                 # per consolidation attempt (bounded compute;
                                 # windows arrive in temporal order)
LABEL_HISTORY_MAXLEN = 10        # Mapper per-label window history bound

# =============================================================================
# 27. BATCH D / P1-1 — ENDOGENOUS CADENCE BOUNDS
# =============================================================================
# Revision OR-bounds. graph_revision() = topology_revision + node_revision
# (ConceptGraph). Waiting-class cadences fire on steps OR graph churn;
# work-class cadences (basin recompute) are pure-revision (HIGH-03).

W_STALL_REVISION = 20
K_MAIN_REVISION = 500
TAU_EXCURSION_REVISION = 100
TAU_EPISTEMIC_REVISION = 50
K_PRUNE_FRONTIER_REVISION = 30
K_FERTILE_REVISION = 100
TAU_PROTO_DECAY_REVISION = 1000

# P1-3 credit propagation to preconditions
D_C = 0.7
MAX_HOPS_CREDIT = 3
# P1-4 partial-failure attribution
GAMMA_BLAME = 0.7
# P1-5 goal-abandon revival (graph revisions before a suppressed goal is
# reconsidered)
N_REVIVE_MIN = 5

# =============================================================================
# [V9 Track 2 — B.15] New constants for the v9 modules
#
# Applied as a DIFF, not as a block splice. Of the draft's 95 names:
#   4 collided with LIVE constants and are OMITTED here:
#       ORIENTATION_N_THETA              draft 12 vs live 24  (hash-changing:
#                                        binder.py:1007/2003/2175/2427 bin by it)
#       ORIENTATION_N_ELONG              draft  8 vs live  5  (hash-changing:
#                                        lenses.py:1297-1299 bin by it)
#       ORIENTATION_ANGLE_RELATION_KAPPA draft 2.0 vs live 1.0 (binder tau_gap)
#       ORIENTATION_ANGLE_RELATION_SIGMA_THETA
#                                        draft 0.15 vs live pi/12 (bend/junction
#                                        angular tolerance)
#     The draft's B.15 was written against a v9 Gabor bank that does not exist
#     in this tree. The v9 Gabor path uses GABOR_N_THETA below (12) and must
#     never shadow a live name; a helper needing a different theta count gets a
#     distinct name.
#   1 was dropped by ruling: MCS_SPECIFICITY_SCALE — B.13 forbids the
#     coverage x specificity form in the same document, and the constant's only
#     consumer was that form (tried in 619e16c, reverted in 6b18560).
#
# NAMING ADJACENCY — read before touching either constant:
#   JUNCTION_MAX_GAP (3.0) / JUNCTION_T_PERP_TOL (0.15) below are the v9
#   parameters for primitives.detect_junctions (the 5-tuple T/Y/X classifier,
#   B.1). They are NOT the tree's v9-era T-junction detection pair —
#   T_JUNCTION_MAX_GAP (6.0) / T_JUNCTION_MAX_PERP_DEV (0.5, parameters.py:312)
#   — which are read by sensory_floor._detect_t_junctions (the 4-tuple
#   detector used by binder._run_occlusion). Two functions, two schemas, two
#   thresholds. Fixing "the" junction constant without checking which detector
#   reads it is the ABSORPTION_MIN_MIN_SIM trap (B.27) again.
#
# Fingerprint expectation for this commit: EMPTY (constants only; nothing
# reads them yet).
#
# Section order follows the draft: V1 Gabor / pyramid, normalization, color,
# orientation, V2 binders, V4, IT, settle, matcher, core.
# =============================================================================

GABOR_N_THETA = 12
GABOR_N_SF = 5
ORIENTATION_RICH_MARGINAL_BINS: int = 60
MAX_M_PREV_SIZE: int = 50_000
GABOR_SF_MIN = 0.06
GABOR_SF_MAX = 0.35
GABOR_SIGMA_X_FRAC = 0.56
GABOR_ASPECT = 0.5
GABOR_KERNEL_MAX = 113

# [FIX 1 / Item round 3] Gabor kernel size from max(sigma_x, sigma_y).
# GABOR_ASPECT = 0.5 makes sigma_y = sigma_x / 0.5 = 2 * sigma_x, so the
# envelope is twice as long along the bar. The old size was based on
# 3 * sigma_x, clipping at 1.5 * sigma_y instead of 3 * sigma_y. The flag uses
# max(sigma_x, sigma_y) = sigma_y so no scale is clipped short along the bar.
# [ROUND 10 / B] Now ON by default: the truncation at ~1.55 sigma along the bar
# was a source of the diagonal/cardinal gain anisotropy; the build phase does
# not need the dark flag. Still env/toggle-overridable for A/B. The cap (113)
# was raised so the coarsest scale (sy ~ 18.7) reaches 3*sy without clipping.
GABOR_KERNEL_SIZE_FROM_MAX_SIGMA: bool = (
    os.environ.get("GABOR_KERNEL_SIZE_FROM_MAX_SIGMA", "1") == "1"
)

# --- [ROUND 10 / PSYCHOPHYSICS] measured V1 population width ---------------
# The graded width of the orientation population (core.population.PopAxis
# sigma_bins) and the width the assembly MDL charges against are NOT tuned by
# hand: they are MEASURED from the bank's OWN orientation tuning.
# diagnostics/_psychophysics.py renders a grating at each band's wavelength,
# reads the per-band orientation curve at the frame centre, and converts the
# interpolated half-width at half-maximum (10.0 deg) to a Gaussian sigma in
# bins (one channel = one bin, step 180/12 = 15 deg):
#     sigma = (HWHM_deg / step_deg) / sqrt(2 ln 2) = 0.6667 / 1.17741 = 0.56621.
# tests/perception/test_psychophysics.py re-measures it and asserts these
# parameters equal the measurement, so a Gabor-kernel or bank edit cannot
# silently change the representation's own resolution -- it fails the contract
# until the number is re-derived and logged (the same discipline the field
# constants use). Previously this width was declared flat at 1.0 bin.
POP_ORIENTATION_SIGMA_BINS: float = 0.566214533525346
# MDL graded-parameter width, in bins: the representation's own finest
# resolution (memory.assembly.parameter_precision_cost_bits charges
# log2(range / width)). Applied per graded internal relation by
# memory.assembly_readout._relation_parameter_widths.
RELATION_PARAMETER_WIDTH_BINS: float = POP_ORIENTATION_SIGMA_BINS

# --- [B.1] Rich complex-cell signature (ORIENTATION_RICH_{nt}_{nsf}) --------
# [B.1] SIFT-style histogram clamp threshold
B1_HIST_CLAMP = 0.2
# [B.1] Minimum phase resultant length for phase-difference computation.
# Below this, the phase is noise-dominated and differences are meaningless.
B1_PHASE_R_MIN = 0.1
# [B1 FIX — AUDIT #1] Front-end uses DIVISIVE normalization, NOT theta-smoothing.
# theta smoothing with sigma=pi/4 collapses 12 orientations to ~3 channels.
# RF_FRAC: pooling covariance = RF envelope covariance * RF_FRAC**2.
#   0.3 => pool over ~30% of the RF (position tolerance without merging
#   two distinct nearby features).
GABOR_COMPLEX_POOL_RF_FRAC: float = 0.3
# SIGMA_THETA: NO orientation smoothing in the front-end (0.0 = off).
#   Cross-orientation competition is divisive (B1_NORM_*), not convolutional.
GABOR_COMPLEX_POOL_SIGMA_THETA: float = 0.0
# PHASE_ACROSS_SIGMA: phase pooling width ACROSS spatial positions (pixels),
#   used to build pool_phase for the phase-consistency block.
B1_PHASE_ACROSS_SIGMA: float = 2.0
# [AUDIT #1] Heeger semisaturation on gain-normalised energy (fixed ruler,
#   not a running max). B1_NORM_SIGMA_THETA: w(90 deg)=0.42.
B1_NORM_SIGMA: float = 0.05
B1_NORM_SIGMA_THETA: float = 1.2

FFT_KERNEL_THRESHOLD = 31
# [O5] Thread count for the value-preserving field/V1 loops. This is a
# PERFORMANCE knob only: every task writes a disjoint output slice, so the
# produced arrays are byte-identical at any value (proven in the O5 harness
# and tests/perception/test_field_threads_and_cache.py). It is still folded
# into field_fingerprint() so a change cannot silently reuse a cache entry
# whose producing configuration differs.
FIELD_THREADS = 4
PYRAMID_N_LEVELS = 4
MIN_PYRAMID_DIM = 16
DIVISIVE_SIGMA = 0.1
DIVISIVE_POOL_RADIUS = 2.0
DIVISIVE_POOL_SIGMA = 2.0
DIVISIVE_W_SPATIAL = 0.7
DIVISIVE_W_CROSS = 0.3
# [B2b — AUDIT #2] Orientation pooling width for the lens across-anchor pool.
# Formula: w_theta = exp(-0.5*(d_theta / σ)²)  (standard Gaussian).
# With σ = 1.2: w(90°) = exp(-0.5*((π/2)/1.2)²) ≈ 0.424 — inside the
#   0.3–0.6 biological band.
DIVISIVE_SIGMA_THETA = 1.2          # orientation pooling width, radians
DIVISIVE_SIGMA_SF = 0.8             # log2 spatial-frequency width
DIVISIVE_ANISO_POOL_RADIUS = 16.0   # px; NOT DIVISIVE_POOL_RADIUS
                                    # (2.0, live, and bound as
                                    # normalize_2d's isotropic default)
COLOR_CONTRAST_RADIUS = 3
ILLUMINANT_TOP_FRAC = 0.1
LUM_SALIENCE_MULT = 0.25
CHROMA_SALIENCE_MIN = 0.10
ENERGY_MASK_MIN = 0.15
ORIENTATION_N_SF = 5
ORIENTATION_N_PHASE = 4
ORIENTATION_N_CONTRAST = 16
ORIENTATION_MIN_RESPONSE = 0.02
ORIENTATION_NEIGHBOR_MULT = 3.0
SIMPLE_CELL_GAIN = 0.8
SIMPLE_PHASE_SNR_MIN = 0.4
SIMPLE_SNR_FLOOR = 0.1
# [B.30 Tier-1 Fix A / B.15] |normalized signed phase-0 response|
# above this classifies the anchor as ON- or OFF-dominant; below, as
# balanced. Consumed by polarity_lens.PolarityLens._anchor_polarity.
POLARITY_ON_THRESHOLD = 0.35

# [B.46] Retinal ON/OFF front end (draft: retinal_frontend.py).
# Difference-of-Gaussians center/surround, then half-wave rectified
# into the ON and OFF channels the Gabor bank runs on separately.
RETINAL_CENTER_SIGMA = 1.0
RETINAL_SURROUND_MULT = 3.0

# [B.47] End-stopped (hypercomplex) cells, draft: end_stopped.py.
ENDSTOP_HALF_LENGTH_PX = 6.0
ENDSTOP_FLANK_K = 0.7
ENDSTOP_N_SAMPLES = 11
ENDSTOP_THRESHOLD = 0.05

# [B.48] Dimension-specific feature attention (Reynolds & Heeger 2009).
ATTENTION_BETA_FEAT = 0.4
TEXTURE_MIN_WINDOW = 8
EDGE_N_CONTRAST = 16
EDGE_N_SHARPNESS = 8
EDGE_N_ORIENT = 24
# [P1-5] THE ASSOCIATION FIELD'S ORIENTATION LIMIT, DERIVED FROM THE BANK.
#
# The cocircular fan's rays deviate from the reference axis by
# `half = asin(AF_PERP_FRAC / AF_LONGITUDINAL_MULT)` (perception/
# contour_field.py), and the alternate-segment theorem turns that deviation
# into a predicted orientation `theta_ref + 2*joff`: the fan's ORIENTATION
# EXCURSION is therefore 2*half.
#
# The limit used to be a FREE constant (0.35 rad) that no geometry
# produced, so the fan could be refused for being narrow (the comparison
# was inverted) or allowed for being wide, and the two numbers could
# disagree. The limit is now the BANK'S OWN ORIENTATION RESOLUTION: the
# angular gap between two adjacent Gabor orientation channels. A fan whose
# predicted orientation stays inside that gap names a direction the bank
# can still resolve as local to the reference; beyond the gap it reads a
# channel that is not the reference's neighbourhood at all.
#
# The two numbers now come from ONE geometry -- the excursion is derived
# from the AF constants, the limit is derived from the bank -- so they
# cannot disagree. The admission condition is exactly
#     AF_PERP_FRAC / AF_LONGITUDINAL_MULT <= sin(pi / (2 * GABOR_N_THETA))
# i.e. 0.1 <= 0.1305 for the shipped constants, so the derived fan
# (2*half = 0.2003 rad) is admitted by the derived limit (0.2618 rad).
# Proven by diagnostics/_p0_af_geometry.py (which also shows the field's
# output is byte-identical before and after the change).
#
# PROVENANCE: rule = "the limit is the bank's own orientation channel gap";
# data source = the Gabor bank's own geometry (GABOR_N_THETA; no corpus);
# date = 2026-10-22; mechanism version = association-field-v3 (the field's
# own formula version is contour_field._FIELD_FORMULA_VERSION = 2).
AF_CHANNEL_GAP_RAD = math.pi / GABOR_N_THETA
AF_MAX_DELTA_THETA = AF_CHANNEL_GAP_RAD
AF_PERP_FRAC = 0.3
AF_LONGITUDINAL_MULT = 3.0
AF_SIGMA_THETA = 0.15
# [AF RE-DERIVATION, Round 6] The association field is MODULATORY, not
# additive: integration = driving * (1 + AF_GAIN * facil/(driving+facil)).
# Facilitation can only amplify local (driving) evidence, never create it in a
# channel with no local energy, so a cross segment bridging two structures
# gets no support in the gap. AF_GAIN bounds the maximum amplification
# (1 -> up to 2x) and is a fixed principled value, not a tuned knob.
AF_GAIN = 1.0
# [P1-6] THE CONTOUR_PEAK GRADED SLOTS' SPANS -- MEASURED, NOT TUNED.
#
# The peak's STRENGTH and COHERENCE slots are graded quantities with a
# SPAN (the "clip"): the bin is floor(value / span * 8) and the graded
# population bump is centred on the same continuous position. The spans
# were free constants taken from a STALE measurement (5 images, an older
# field formula): STRENGTH_CLIP = 20 with a claimed p50 of 17.0 (so the
# top bin was a saturated plateau) and COHERENCE_CLIP = 0.15 with a
# claimed max of 0.147 (so the axis barely used its span).
#
# RULE: the span of a graded slot is the measured MAXIMUM of the quantity
# over the real corpus -- the quantity's own range end, which is what
# perception/contour_peak.py has always documented ("bin edges are chosen
# from these observed ranges; anything outside the range is clipped to the
# nearest edge"). Nothing is fabricated and nothing is tuned: a value
# outside the corpus's range (a new frame, a different corpus) is clipped
# to the edge, which is the declared behaviour of a clip.
#
# MEASUREMENT (the data source of the two constants below): 4200 field
# peaks over 16 real ETH-80 frames (8 classes x 2 instances, canon 256,
# the PRODUCTION luminance), measured by
# diagnostics/_p0_peak_distributions.py:
#   integration_strength: min 0.0194, p01 0.391, p10 2.802, p50 3.238,
#                         p90 3.996, p99 22.429, max 29.838
#   coherence:            min 0.0955, p01 0.1075, p10 0.1216, p50 0.1527,
#                         p90 0.2117, p99 0.3030, max 0.4925
# The OLD coherence clip saturated the population: with 0.15, 72 % of the
# measured peaks (p50 = 0.1527) landed in the TOP bin, so the coherence
# slot carried one level for most of the population. The strength
# distribution is strongly bimodal (a bulk at ~3.2 and a thin tail to
# 29.8), so its linear axis stays coarse in the bulk -- that is the
# measurement's shape, not a choice.
#
# PROVENANCE: rule = "the span is the measured maximum over the corpus";
# data source = diagnostics/_p0_peak_distributions.py over 16 real
# ETH-80 frames (4200 peaks); date = 2026-10-22; mechanism version =
# contour-field-v2 (field formula version 2) / contour-peak-v3.
CONTOUR_PEAK_STRENGTH_CLIP = 29.84
CONTOUR_PEAK_COHERENCE_CLIP = 0.4925
# [FIGURE-GROUND CUE WEIGHTS] RETIRED with the entity path. The weight
# vector named the RESERVED region/global keys (`symmetry`, `small_area`,
# `lower_region`, `familiarity`) -- a global shape/region decision the
# emergent-objecthood mandate forbids. The live weights are the FITTED
# log-odds file src/parameters/_ownership_weights.json (convexity,
# enclosure, t_junction), read by perception/fragment_ownership.py.
# [FIGURE-GROUND FUSION] A/B switch for the cue combiner on the per-vertex
# ownership signal. True: combine convexity + enclosure + texture + colour +
# gravity through BO_WEIGHTS (perception/ownership.py). False: the legacy
# four-term weighted sum (curvature/texture/colour/gravity). Structural
# switch only; the cues themselves are unchanged.
# [SIGNED EDGE VOCABULARY] A/B switch for the signed figure-ground edges:
# EDGE_BORDER_OWNERSHIP carries sign=+1 (the entity is the figure side) or
# sign=-1 (the entity is the ground side, i.e. the neighbour owns the border),
# and anti-phase OCCLUSION edges are emitted figure -> ground. False restores
# the prior unconditional sign=+1 on the ownership channel.
BO_SIGNED_EDGE_ENABLED: bool = True
# [INTERIOR FIELD] bounded-diffusion interior/exterior field
# (perception/interior_field.py). A bounded geodesic (min-plus) relaxation on a
# fixed-size working grid; deterministic, O((h+w)*hw). Enclosure is measured as
# the EXTRA path cost edges impose on escaping to the frame, so it fills
# interiors and reads 0 on a blank image. The one stated limit of the scheme:
# a boundary whose edge cost is below INTERIOR_EDGE_PENALTY is not seen as a
# barrier.
INTERIOR_EDGE_PENALTY: float = 6.0
INTERIOR_ENCLOSURE_SCALE: float = 3.0
INTERIOR_WORK_SIZE: int = 64
INTERIOR_PROBE_PX: float = 3.0
INTERIOR_FIELD_CACHE_SIZE: int = 8
# [INTERIOR FIELD -- NOISE GATE] The gradient magnitude is gated by its own
# NOISE FLOOR before it is normalised: only a gradient above the frame's
# median magnitude counts as an edge, so the background's gradient noise
# stops being a barrier. Provenance: the field's own contract is that a
# point whose cheapest escape to the frame crosses no edge must read 0
# (exterior) -- but the raw magnitude, normalised by its 95th percentile,
# puts the background's noise at ~0.3 of the scale, so EVERY escape paid
# extra cost and the field saturated (measured: 87% of a real frame at
# enclosure > 0.99, and 0 of 2137 captured parts had any flank support).
# Re-measured on synthetic corpora (diagnostics/enclosure_sat_probe.py),
# rule written first: "the exterior must reach the frame edge"; blank frame
# reads enclosure ~0 at every percentile, but a textured frame falsely
# encloses 70.8% of its exterior at the median (floor=50) vs 10.1% at floor
# 90, while the blank frame stays at 0 everywhere. The floor is therefore
# the 90th gradient percentile: the gradient magnitude gate must suppress
# background gradient noise so object exteriors are not falsely enclosed.
# mechanism version interior-field-v3, date 2026-10-22.
#
# [O4 RE-DERIVATION, 2026-10-25] The chamfer was made ISOTROPIC (8-neighbour,
# sqrt(2) diagonals; it was Manhattan). Re-running the same rule on the same
# corpora (5 texture seeds + disk/rect/blank) shows the Manhattan bias was
# itself suppressing part of the false enclosure, so the OLD value no longer
# meets its own recorded criterion: at floor 90 a textured frame now falsely
# encloses 22.9-25.0% of its exterior above 0.5 (was ~10%). Floor 95 restores
# the pre-registered ~10% (9.9-11.3% across the 5 seeds) while a compact
# object's interior still reads enclosure > 0.99 over 13.5% of the frame
# (disk r=25: 1963 px = 12.0%; rect 40x40: 1600 px = 9.8%), i.e. essentially
# the whole object, and the blank frame stays 0.0 everywhere. The floor is
# therefore the 95th gradient percentile under the isotropic metric.
# mechanism version interior-field-v4, date 2026-10-25.
INTERIOR_EDGE_FLOOR_PCT: float = 95.0
JUNCTION_MAX_GAP = 3.0
JUNCTION_T_PERP_TOL = 0.15
AMODAL_MIN_CONFIDENCE = 0.4
AMODAL_MIN_RELATABILITY = 0.5
AMODAL_MAX_THETA = 0.35
AMODAL_MAX_CONN = 0.5
AMODAL_SEARCH_RADIUS = 4.0
AMODAL_SEARCH_RADIUS_MIN_PX = 3.0
COCIRCULAR_R_MIN = 5.0
COCIRCULAR_R_MAX = 200.0
COCIRCULAR_SIGMA_KAPPA = 0.15
COCIRCULAR_SEARCH_RADIUS = 8.0
COCIRCULAR_MAX_PATH_ANGLE: float = math.pi / 6.0
PARALLEL_THETA_TOL = 0.15
PARALLEL_MIN_OFFSET = 0.5
PARALLEL_MAX_OFFSET = 5.0
PARALLEL_SIGMA_THETA = 0.1
CLOSURE_Z_BOOST = 1.4
ALPHA_ATTN = 0.4

CURVATURE_N_SAMPLES = 5
POOL_SHIFT = 2.0
POOL_SCALES = 3
K_IT_SPARSE = 5
IT_OVERLAP_MAX = 0.5
SYMMETRY_RADIUS = 0.3
MIRROR_SIMILARITY = 0.85
SETTLE_MAX_ITERS = 6
SETTLE_STABILITY_EPS = 0.02
SETTLE_TOP_K = 3
BETA_TOP_DOWN = 0.4
BETA_TEMPORAL = 0.2
GAIN_THETA_REDUCTION = 0.3
HYPOTHESIS_MAX_BOXES = 8
ORIENTATION_BEND_THETA_MIN = 0.15
ORIENTATION_BEND_THETA_MAX = 0.80
ORIENTATION_BEND_THETA_CENTER = 0.45
ORIENTATION_JUNCTION_THETA_MIN = 0.80
ORIENTATION_JUNCTION_THETA_MAX = 1.571
ORIENTATION_JUNCTION_THETA_CENTER = 1.20
THETA_GIST_MIN = 0.4
ABSORPTION_MIN_SIM = 0.6
REL_AGREE_W_REF = 0.6
REL_AGREE_W_TGT = 0.4
PARTIAL_OCCLUSION_CREDIT = 0.5
T_JUNCTION_RADIUS = 2.0
CORE_MIN_TOLERANCE = 0.02
CORE_MIN_SIZE = 3
NEGATIVES_PER_MIN = 100
HYPOTHESIS_MATCH_MIN = 0.75
SALIENCY_TOP_FRAC = 0.6
SPATIAL_CELL_MIN = 4.0
MAX_REGIONS = 64
MOTION_BOUNDARY_THRESHOLD = 0.15

# --- [S3/S5] Edge parameter mixture & relational matching constants ---
EDGE_MIX_GAMMA: float = 0.98            # Per-step temporal mass decay (half-life ~34 steps)
EDGE_MIX_MERGE_TOL: float = 1.0         # Bin tolerance for prototype clustering (L-infinity)
EDGE_MIX_MAX_MODES: int = 4             # Maximum joint modes per edge mixture
EDGE_MATCH_USE_CONFIDENCE: bool = False # Match pure relational geometry during MCS
EDGE_PARAM_ONESIDED_SIM: float = 0.5    # Neutral score when exactly one edge has parameters


# --- [B19] Luminance adaptation ---------------------------------------------
# Adaptation scale as a fraction of the short image side. SCENE-SCALE ON
# PURPOSE: an object must not be its own illuminant estimate. This is the same
# lesson as the B8 Retinex fix ([AUDIT #14]) -- tying the scale to
# extraction_scale made large regions divide by themselves.
LUM_ADAPT_FRAC: float = 0.15
# Semisaturation = L0 + A * local_mean. A=1 maps local-mean luminance to 0.5.
LUM_ADAPT_A: float = 1.0
# Absolute floor so TRUE BLACK STAYS BLACK (without it, 0 maps to 0/TINY=0
# anyway, but the floor is what bounds the response slope near zero).
LUM_ADAPT_L0: float = 0.02


# --- [B2] Broad cross-orientation suppression & pool normalization ----------
LENS_POOL_NORMALIZE: bool = True
LENS_POOL_PRIOR_MASS: float = 1.0


# --- [B2c] Front-end two-component cross-orientation pool -------------------
B2_COS_ENABLED: bool = False
B2_COS_ALPHA: float = 0.5
B2_COS_SIGMA_ISO: float = 0.3
B2_COS_SIGMA_BROAD: float = 1.2
B2_COS_SIGMA_SF: float = 0.8


# --- [B18] Contrast gain control --------------------------------------------
# [B18] Noise floor, expressed as a fraction of gain-normalised energy. Below
# this the filter is responding to numerical noise, not structure. MUST be
# re-measured after B19 lands: adaptation halves the mean energy, so a floor
# calibrated on raw luminance silently discards real signal.
B18_NOISE_FLOOR: float = 0.01
B18_SIGMA0: float = 0.02         # semisaturation floor
B18_W_CENTER: float = 1.0        # weight of the (theta, sf) cross-channel pool
B18_W_SURROUND: float = 1.0      # weight of the spatial surround
# Surround sigma = RATIO x typical RF sigma, derived from the bank rather than
# hard-coded, so it tracks multi-scale banks automatically.
B18_SURROUND_RATIO: float = 3.0


# --- [D1] End-Stopped (Hypercomplex) Lens -----------------------------------
ENDSTOP_FLANK_KAPPA: float = 1.5
ENDSTOP_REL_THRESHOLD: float = 0.15
ENDSTOP_ENERGY_FRAC: float = 0.25
ENDSTOP_DOUBLE_RATIO: float = 0.6
ENDSTOP_EPS: float = 1e-6
ENDSTOP_NMS_MULT: float = 2.0
ENDSTOP_RIM_FRAC: float = 1.0


# --- [B10] Orientation Geometry (Phase & Contrast) --------------------------
ORIENTATION_PHASE_BINS: int = 24
ORIENTATION_CONTRAST_BINS: int = 8
ORIENTATION_C50: float = 0.5


# --- [B11] Motion Direction Population --------------------------------------
MOTION_DIRECTION_BINS: int = 8
MOTION_COHERENCE_BINS: int = 8
MOTION_SPEED_BINS: int = 8
MOTION_TRANSPARENCY_BINS: int = 8
MOTION_C50: float = 0.5
MOTION_RF_SIGMA_PX: float = 3.0


# --- [B20] Texture-Defined Boundary Lens ------------------------------------
TB_SAMPLE_FRAC: float = 0.5
TB_OFFSET_FRAC: float = 1.0
TB_SAMPLE_COUNT: int = 8
TB_MIN_ENERGY: float = 0.01
TB_MIN_DIST: float = 0.1

# --- [Q1] Co-circularity denominator ---------------------------------------
# Two contour peaks are co-circular enough to chain when the deviation of
# the arc through them from its own chord, d*sin(dtheta/2), stays under this
# many MEDIAN WAVELENGTHS. The bound is denominated in the pipeline's own
# local contour scale, so it carries no absolute angular threshold and is
# resolution-free.
#
# 4.0 is derived, not tuned: it is the Nyquist-style criterion that the
# arc's sagitta (kappa*d^2/8 = d*sin(dtheta/2)/4) must not exceed half a
# wavelength, i.e. d*sin(dtheta/2) <= 4*lambda. Measured on 8 real images
# this admits pairs whose implied curvature is below ~1/lambda and rejects
# pairs above it, which is the same separation an angular cap cannot make.
COCIRCULAR_WAVELENGTHS: float = 4.0

# --- Depth / disparity honesty ---------------------------------------------
# binder.Binder._run_binocular_disparity derives EDGE_BINOCULAR_DISPARITY from
# the THIRD component (z) of 3D anchor positions. That z is a scene/depth
# proxy, not a binocular disparity measurement, so by DEFAULT the channel is
# BLOCKED: the edge may only be produced when this flag is explicitly set AND
# genuine 3D anchors exist. Enum value 50 stays reserved and the channel is
# DARK (not silently alive). [ROUND 11 / O4]
BINOCULAR_DISPARITY_FROM_Z: bool = False

# --- [W1] Entity source for the Shape/identity lenses ----------------------
# True  -> entities come from FIELD regions built on PEAK_CHAINs
#          (perception/chain_regions.py). No segmenter, no boundary tracer.
# False -> the legacy segmenter path (observation.get_boundary_polyline).
# [REPRESENTATION MISSION] TRUE, and not as an A/B arm: the legacy path is a
# SEGMENTER, and the mission's rules are "no segmentation or masks, objects
# emergent". The legacy path was also unreachable in practice -- the
# texture-boundary lens's legacy arm requires
# `observation.sample_texture_profile_at`, an API no observation implements,
# so it emitted nothing at all. The field arm measures the same material
# discontinuity on the front end's own planes and is the arm that fires.
# Provenance: rule = "no segmentation or masks"; date 2026-10-20;
# mechanism version representation-v4.
FIELD_REGION_SOURCE: bool = True
# Minimum enclosed area for a closed chain to count as a figure, expressed
# as a FRACTION OF IMAGE AREA rather than absolute pixels, so the gate is
# scale-free: the same criterion applies at 128 and 256 and at any crop.
# 1e-4 of 512^2 is ~26 px2, matching the previous absolute constant.


# --- [B7/B21] Junction Topology & Tolerances -------------------------------
JUNCTION_COLLINEAR_TOL: float = math.pi / 6.0  # radians (30 deg) for T/X collinearity
JUNCTION_STAR_TOL: float = math.pi / 12.0      # radians (15 deg) base tolerance for Star uniformity
JUNCTION_SATURATION_ARMS: int = 4              # Arm count where z saturates to 1.0
JUNCTION_ARM_ALIGN_TOL: float = math.pi / 6.0  # radians (30 deg) arm-orientation alignment tolerance
KAPPA_JUNCTION_RADIUS: float = 2.0


# --- [B8] Opponent Color & Retinex Constants -------------------------------
COLOR_LUM_BINS: int = 8
COLOR_RG_BINS: int = 16
COLOR_BY_BINS: int = 12
COLOR_SAT_BINS: int = 8
COLOR_HUE_BINS: int = 16  # Circular -> range 8.0

COLOR_RETINEX_SIGMA_FINE_RATIO: float = 1.0   # L/M channels (fine detail)
COLOR_RETINEX_SIGMA_COARSE_RATIO: float = 4.0 # S channels (coarse detail)

COLOR_LOG_OPPONENT_ALPHA_LM: float = 1.5
COLOR_LOG_OPPONENT_ALPHA_S: float = 2.5 * 1.8  # = 4.5

COLOR_BY_WEIGHT: float = 1.8
COLOR_SAT_SCALE: float = 3.0

COLOR_ILLUM_FRAC: float = 0.25
COLOR_ACHROMATIC_EPS: float = 0.02


# --- [B13] CurveComplex Enrichment --------------------------------------
CURVE_COMPLEX_PARALLEL_TOL_RAD: float = math.pi / 12.0
CURVE_COMPLEX_TEMPLATE_BINS: int = 8
KAPPA_CURVE_COMPLEX_RADIUS: float = 4.0

# --- [B14] RegionShape (V4/IT Boundary Conformation) --------------------
REGION_SHAPE_CURV_BINS: int = 16
REGION_SHAPE_HIST_LEVELS: int = 8
REGION_SHAPE_AREA_BINS: int = 8
REGION_SHAPE_ASPECT_BINS: int = 8

# --- [B22] Contour self-symmetry ----------------------------------------
SYM_RESAMPLE_N: int = 48
SYM_N_AXES: int = 12
SYM_TAU: float = 0.20
SYM_FOLD_THRESHOLD: float = 0.80
SYM_ISOTROPIC_SPREAD: float = 0.05
SYM_AXIS_UNDEFINED_BIN: int = 12

# --- [B16] ObjectShape Enrichment (aIT Gist/Symmetry) -------------------
OBJECT_SYMMETRY_AXES: int = 8
OBJECT_GIST_BINS: int = 16
OBJECT_GIST_RING_BINS: int = 8

# --- [B17] Illusory Contours & Modal Completion -------------------------
KAPPA_ILLUSORY_MAX_GAP: float = 6.0
ILLUSORY_ANGLE_TOL_RAD: float = math.pi / 12.0
ILLUSORY_MAX_CANDIDATES: int = 1000


# =============================================================================
# STAGE 2: CONJUNCTIVE RELATIONAL VERIFICATION & POLYMORPHIC SCHEMAS
# =============================================================================
# α bounds the maximum relational penalty. At 0.85, relations modulate
# the node-coverage score by at most 15%.
EVIDENCE_ALPHA_CONJUNCTION: float = float(
    os.environ.get("EVIDENCE_ALPHA_CONJUNCTION", "0.85")
)

# [PHASE 5] Null-calibrated conjunction floor for the DEFAULT evidence path.
# The historical 0.85 floor (EVIDENCE_ALPHA_CONJUNCTION) let an uninformative
# edge award 85% of its node coverage merely because both endpoints mapped.
# Phase 5 retires that floor: the relational multiplier's null is 0 (an
# uninformative edge set yields 0 graded similarity), so the default floor is
# 0.0. EVIDENCE_ALPHA_CONJUNCTION is retained for the A/B legacy path only,
# selected when MATCH_NULL_CALIBRATED is False.
EVIDENCE_CONJUNCTION_FLOOR: float = float(
    os.environ.get("EVIDENCE_CONJUNCTION_FLOOR", "0.0")
)

# [ITEM 3 / K2 + BUG-16] RULE K2-LCB, now shipped ON (default on) after
# the O5c mission re-derived the distribution key on the population
# vectors.  The gate's purpose is to stop an element that is as common in
# the negatives as in the positives from carrying mass; with the bins as
# the key, an element that merely straddled a bin edge looked rare, and
# the gate then removed REAL evidence.  With the key on the measurement,
# the gate sees the element it was written to see.
#   MASS_LCB_GATE       RULE K2-LCB: an element gets mass only when the
#                       one-sided 95% Wilson lower bound of its rate
#                       difference (positives vs negatives) beats chance;
#                       z = 1.645 and the sample sizes are the entire rule
#                       -- no tuned threshold. Enforced inside the ONE
#                       shared mass function (memory/distributions.py
#                       _mass_from_counts).
#   EVIDENCE_ONLY_EDGES RULE B16: cores.evidence_score counts each core
#                       edge only by its own graded evidence; unmatched or
#                       ungraded edges count 0, never the 0.85 conjunction
#                       floor.
MASS_LCB_GATE: bool = os.environ.get("MASS_LCB_GATE", "1") == "1"
EVIDENCE_ONLY_EDGES: bool = os.environ.get("EVIDENCE_ONLY_EDGES", "0") == "1"

# Cross-type polymorphic family projection penalty (10% discount)
EDGE_CROSS_FAMILY_PRIOR: float = float(
    os.environ.get("EDGE_CROSS_FAMILY_PRIOR", "0.90")
)

# Two-component structural absence (replaces scalar absence).
# λ_node penalizes unmatched core nodes; λ_edge penalizes verifiable
# core edges that fail to match in Q.
EVIDENCE_LAMBDA_NODE: float = float(
    os.environ.get("EVIDENCE_LAMBDA_NODE", "0.20")
)
EVIDENCE_LAMBDA_EDGE: float = float(
    os.environ.get("EVIDENCE_LAMBDA_EDGE", "0.15")
)


# =============================================================================
# BIOLOGICAL TWO-STREAM FRONT-END (V1 LATTICE & ASSOCIATION CONTOURS)
# =============================================================================
# [Scale-relative front end] The pixel constants below were calibrated at a
# 256x256 reference frame. _front_end_scale(h, w) returns diag/REF, and the
# PIXEL-denominated constants are multiplied by it at use time, so a
# half-resolution image (scale 0.5) uses half the pixel tolerances while the
# Gabor cycles-per-pixel band is DIVIDED by it (same physical tuning).
# Evidence: the A/B/C self-resonance experiment (diagnostics/stage5_gates.py
# `three`) gave resonance(A,B)=0.149 vs resonance(A,C)=0.591 on identical
# images, with mean contour count 36.8 -> 14.9 at half resolution processed
# natively. C high / B low implicates absolute-pixel thresholds, not
# information loss -- exactly what this scaling repairs. Normalised (0-1),
# angular, and count-based constants are unaffected.
FRONT_END_REF_DIAGONAL_PX: float = 362.0386719678744  # sqrt(256^2 + 256^2)
RETINOTOPIC_GRID_DENSITY: int = 16
RETINOTOPIC_RF_OVERLAP: float = 1.6
CONTOUR_SIGMA_THETA: float = 0.45
CONTOUR_SIGMA_DIST: float = 3.5
CONTOUR_COS_POWER: float = 2.0
# [ROUND 10 / C5] Audited against the v2 field (diagnostics/_field_calibration.py).
# This constant gates `strength_norm = strength_map / s_ref`, a NORMALISED
# Gabor-energy fraction in [0, inf) -- NOT an integration/driving value -- so the
# v2 (modulatory-field, sub-pixel-beam) change does not move its units. Rule:
# ridge seeds are the local maxima whose normalised strength reaches this
# fraction of the interior p99.5 reference; 0.12 keeps real ridges while the
# interior-percentile normalisation already removes the border artifacts.
CONTOUR_STRENGTH_THRESHOLD: float = 0.12
CONTOUR_MIN_LENGTH: int = 12
CONTOUR_MAX_TRACING_STEPS: int = 350
CONTOUR_MAX_SEEDS: int = 180
# [Front-end fix 2026-10-10] The reflect-padded convolution of the LOWEST-SF
# Gabor kernel manufactures spurious energy ridges at ~3*sigma_x inside the
# frame border. A global-max normalisation then lets those artifacts set the
# reference, diluting real objects below CONTOUR_STRENGTH_THRESHOLD: on the
# synthetic zoom sweep a disc at r<=0.20 of a 256^2 frame produced ZERO
# contours. Normalise against this percentile of the INTERIOR (border margin
# = the lowest-SF kernel's support radius) instead.
CONTOUR_INTERIOR_PERCENTILE: float = 99.5
# Absolute floor for the interior-percentile reference. A percentile alone has
# no notion of "is there any structure at all": on a uniform field it happily
# normalises numerical noise into ridges (measured: a blank 256^2 frame gives
# interior p99.5 = 6.9e-07 yet produced 180 contours). The separation from real
# structure is ~6 orders of magnitude (hardest small object: 2.75e-01), so any
# value between ~1e-5 and ~1e-2 is safe; 1e-3 is used.
# [ROUND 10 / C5] Audited against the v2 field (diagnostics/_field_calibration.py):
# this gates the ABSOLUTE interior p99.5 of raw Gabor energy, not an
# integration/driving value, so its units are unchanged by the v2 field formula.
# Same noise-floor rule as PEAK_MIN_INTEGRATION_ABS: sit above the structure-free
# maximum, far below the weakest real object.
CONTOUR_MIN_ABS_STRENGTH: float = 1e-3


# --- recovered from the original parameters.py (audit repair) ---
_OLD_THETA_SIM = 0.7


# --- recovered from the original parameters.py (audit repair) ---
_OLD_THETA_SIM_CONSENSUS = 0.8

# [Phase 2 / contour field] V1/V2/V4 field parameters.
# Declared here so they are auditable and tunable rather than buried as
# getattr defaults inside perception/contour_field.py.
PEAK_RIDGE_SPACING_FRAC: float = 1.0
# Relative peak gate (fraction of the summed-integration maximum).
PEAK_MIN_INTEGRATION_FRAC: float = 0.15
# Absolute peak gate. A relative threshold alone normalises float noise
# against itself and lets a structure-free frame through; this is the
# gate-1 analogue of CONTOUR_MIN_ABS_STRENGTH.
# Re-measured after the wide-context (cocircular) term became MODULATORY rather
# than additive (contour_field v2) and the beam steps became sub-pixel/equal-
# arc-length. Synthetic corpus, 256^2: blank grey interior max 1.7e-4, blank white
# interior max 3.4e-4, structured (arc+bar) interior max 6.37 (was 1.99e1 with the
# additive term). The floor is set near the blank/real geometric mean
# (sqrt(3.4e-4 * 6.37) ~ 0.047 -> 0.05): strict on structure-free frames,
# inert on real frames (carried by the relative term).
# [U25 LEAD 2 / F1-F3 RE-DERIVATION] Re-derived after the association-field beam
# became MODULATORY (F1: bent-beam total-turn distribution; F3: energy-weighted
# end-stop mean) changed the integration scale. Field calibration (v6): blank
# white interior max 3.98e-5, structured (arc+bar) interior max 1.08257, blank
# peaks 0 / structured peaks 116. Re-derived floor = geometric mean:
# sqrt(3.98e-5 * 1.08257) = sqrt(4.309e-5) ~ 0.0066 (was sqrt(3.4e-4 * 6.37) ~ 0.047
# -> 0.05, now stale). Behavioural re-check: PASS (blank rejected, structure
# admitted). Floor lowered accordingly.
# [U26 RE-DERIVATION, v7 field] The v6 calibration does NOT reproduce on the
# shipped `_FIELD_FORMULA_VERSION = 7` field (U25 excised the border gate, made
# the curvature denominator `hypot(x_fwd-x_bwd, y_fwd-y_bwd)` and moved the
# curvature floor to `I >= min_integration`), which changed the integration
# scale back to (and beyond) the round-10/B values.  Measured on v7, 256^2,
# `diagnostics/_round10_B_rederive.py`:
#     blank grey 1.72e-4,  blank white 3.52e-4,  structured 6.515 (32 peaks)
# geometric mean(blank white, structured) = sqrt(3.517e-4 * 6.515) = 0.0479
# i.e. the round-10/B value the v6 note dismissed as "stale".  The shipped
# 0.0066 sat 7.3x below the measured geometric mean and its recorded v6
# justification does not hold on v7.  The floor is set to the geometric mean,
# as this file's own documentation (and the U26 directive) requires:
# sqrt(3.517e-4 * 6.515) ~ 0.048.  136x above the blank-frame interior max,
# inert on every frame whose I_max exceeds 0.32 (the relative term governs),
# which is every occlusion scene in the U24 corpus; and above the blank-frame
# max for structure-free frames, which is the gate's whole purpose.
PEAK_MIN_INTEGRATION_ABS: float = 0.048
COCIRCULAR_WIDE_SIGMA_FRAC: float = 2.0
COCIRCULAR_WEIGHT: float = 0.25
ENDSTOP_MAX_RADIUS_FRAC: float = 2.0
CURVATURE_MAX_RADIUS_FRAC: float = 2.0

# --- [A5] Peak & chain anchor base ------------------------------------------
# Anchors for the field pipeline are computed from position in the field's
# peak list, so any lens can reference a peak without a lookup table.
# Well above any pipeline-internal anchor counter: the region allocator and
# the binder's synthetic anchors (hash_64 values) live in the tens of
# millions at the very highest image counts we have observed, so 1e9 keeps
# the two spaces disjoint by a wide margin.
PEAK_ANCHOR_BASE: int = 1_000_000_000
# Chain anchors are offset from the peak base by this much so that a chain
# ordinal can never collide with a peak index (peak indices are bounded by
# the frame's peak count, far below this).
CHAIN_ANCHOR_OFFSET: int = 100_000
# [Phase 1 / singular points] Corner anchors form a third disjoint band above
# the peak and chain bands so a corner ordinal cannot collide with either.
CORNER_ANCHOR_OFFSET: int = 200_000
# Minimum boundary turn for a chain vertex to be a corner rather than smooth
# curvature (radians). ~34 deg: a structural sharp/smooth separation, not a
# tuned detection threshold -- a right-angle corner is pi/2 (~1.57), a gentle
# bend is well under this.
CORNER_MIN_TURN: float = 0.60

# --- [ROUND 10 / ITEM E] Assembly contract runtime (additive) ---------------
# Pipeline-owned fast store retention: an episode may be dropped once every
# node it recorded is covered by a live concept whose support reached this.
ASSEMBLY_EPISODE_MIN_SUPPORT: int = 2
# Minimum member-set overlap for two live concepts to be recorded as competing
# assemblies (the overlap-link cache).
ASSEMBLY_OVERLAP_MIN: float = 0.25
# Label members MAY assist activation/completion during the build/test phase.
# DECLARED CHEAT: every number produced while a label member is present is
# tagged "label-assisted".
ASSEMBLY_LABEL_ASSIST: bool = True

# The PEAK_CHAIN field spine is the default contour representation.
CONTOUR_LENS_ENABLED: bool = False
