ARITY_MAX = 8
B_MAX = 64
DELTA_CAT = 0.05
DELTA_CHUNK = 0.05
DELTA_OP = 0.05
DELTA_THETA_CAT_MAX = 0.25
D_CHUNK_MAX = 4
ETA_CTX_OP = 0.05
ETA_FAT_DECAY_OP = 0.01
ETA_FAT_OP = 0.05
KAPPA_BIRTH = 0.1
KAPPA_CAT = 0.1
KAPPA_CHUNK_HOMEO = 0.05
KAPPA_EMA = 0.1
KAPPA_OP = 0.1
K_CATEGORIZE = 8
K_CHUNK_CAND = 8
K_CHUNK_PLAN = 8
K_CHUNK_STAR = 4
K_OP = 8
RF_MAX = 10.0
RHO_BIRTH = 0.05
RHO_CAT_TARGET = 0.1
RHO_CHUNK_TARGET = 0.1
RHO_OP_TARGET = 0.1
RHO_PROGRESS_MIN = 0.05
RHO_PROMO_RF = 0.5
SIGMA_MAX_OP = 2.0
TAU_EDGE_INACTIVE = 50
TAU_OP_COOLDOWN = 5
TAU_PROGRESS = 10
TAU_PROMO_MIN = 5
TAU_REVERT = 20
THETA_CATEGORIZE = 0.5
THETA_CATEGORIZE_MAX = 0.9
THETA_CATEGORIZE_MIN = 0.2
THETA_CHUNK_EMIT = 0.5
THETA_CHUNK_EMIT_MAX = 0.9
THETA_CHUNK_EMIT_MIN = 0.2
THETA_EDGE_ACTIVE = 0.15
THETA_FLOOR_OP = 0.1
THETA_HYST_OP = 0.05
THETA_PROMO_OP = 0.6
THETA_SIM = 0.65
W_CHUNK = 1.0
W_PLAN_CHUNK = 1.0
Y_FLOOR_OP = 0.05
_SIM_MEMO = {}
_SIM_MEMO_CAP = 4096
def _compat_key(a, b): return (getattr(a, "class_", ""), getattr(b, "class_", ""))
def _sig_compatible(a, b) -> bool: return getattr(a, "class_", None) == getattr(b, "class_", None)
def _sim_memo(a, b):
    if not _sig_compatible(a, b): return 0.0
    qa = getattr(a, "quantized", ()) or ()
    qb = getattr(b, "quantized", ()) or ()
    if not qa or len(qa) != len(qb): return 0.0
    diffs = [abs(float(x) - float(y)) / 8.0 for x, y in zip(qa, qb)]
    return max(0.0, 1.0 - sum(diffs) / len(diffs))
