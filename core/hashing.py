import hashlib, struct
from typing import Any

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

def heaviside(x: float) -> float:
    return 1.0 if x >= 0.0 else 0.0

def serialize_int(v: int) -> bytes:
    return b"i:" + str(int(v)).encode("utf-8") + b";"

def serialize_float(v: float) -> bytes:
    return b"f:" + struct.pack(">d", float(v))

def serialize_bytes(v: bytes) -> bytes:
    return b"b:" + len(v).to_bytes(4, "big") + bytes(v)

def serialize_tuple(v) -> bytes:
    return b"t:[" + b",".join(serialize_value(x) for x in v) + b"]"

def serialize_set(v) -> bytes:
    items = sorted(serialize_value(x) for x in v)
    return b"s:{" + b",".join(items) + b"}"

def serialize_value(v: Any) -> bytes:
    if v is None:
        return b"N"
    if isinstance(v, bool):
        return b"B1" if v else b"B0"
    if isinstance(v, int):
        return serialize_int(v)
    if isinstance(v, float):
        return serialize_float(v)
    if isinstance(v, str):
        b = v.encode("utf-8")
        return b"u:" + len(b).to_bytes(4, "big") + b
    if isinstance(v, (bytes, bytearray)):
        return serialize_bytes(bytes(v))
    if isinstance(v, (tuple, list)):
        return serialize_tuple(v)
    if isinstance(v, (set, frozenset)):
        return serialize_set(v)
    return b"r:" + repr(v).encode("utf-8")

def hash_128(*parts: bytes) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(p if isinstance(p, (bytes, bytearray)) else serialize_value(p))
    return h.digest()

def hash_64(*parts: bytes) -> int:
    return int.from_bytes(hash_128(*parts)[:8], "big", signed=False)

def hash_64_signed(*parts: bytes) -> int:
    return int.from_bytes(hash_128(*parts)[:8], "big", signed=True)

def _node_signature_hash(sigma: Any) -> bytes:
    return hash_128(
        serialize_value(getattr(sigma, "kind", "")),
        serialize_value(getattr(sigma, "class_", "")),
        serialize_value(getattr(sigma, "relation", "")),
        serialize_value(tuple(getattr(sigma, "quantized", ()) or ())),
    )

def _detection_sort_key(d: Any):
    return (
        str(getattr(d, "k", "")),
        int(getattr(d, "scale", 0)),
        _node_signature_hash(getattr(d, "sigma", None)),
        serialize_value(getattr(d, "alpha", None)),
    )
