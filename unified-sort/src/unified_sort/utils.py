from __future__ import annotations
import hashlib

def make_widget_key(prefix: str, path: str, digest_len: int = 8) -> str:
    return f"{prefix}_{hashlib.md5(path.encode('utf-8')).hexdigest()[:digest_len]}"
